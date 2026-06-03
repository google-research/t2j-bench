# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handles function calling for LLM flow."""

from __future__ import annotations

import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
import copy
import functools
import inspect
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any
from typing import AsyncGenerator
from typing import cast
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.platform import uuid as platform_uuid
from google.adk.tools.computer_use.computer_use_tool import ComputerUseTool
from google.genai import types

from ...agents.active_streaming_tool import ActiveStreamingTool
from ...agents.live_request_queue import LiveRequestQueue
from ...auth.auth_tool import AuthConfig
from ...auth.auth_tool import AuthToolArguments
from ...events.event import Event
from ...events.event_actions import EventActions
from ...telemetry.tracing import trace_merged_tool_calls
from ...telemetry.tracing import trace_tool_call
from ...telemetry.tracing import tracer
from ...tools.base_tool import BaseTool
from ...tools.tool_confirmation import ToolConfirmation
from ...tools.tool_context import ToolContext
from ...utils.context_utils import Aclosing

if TYPE_CHECKING:
  from ...agents.invocation_context import InvocationContext
  from ...agents.llm_agent import LlmAgent

AF_FUNCTION_CALL_ID_PREFIX = 'adk-'
REQUEST_EUC_FUNCTION_CALL_NAME = 'adk_request_credential'
REQUEST_CONFIRMATION_FUNCTION_CALL_NAME = 'adk_request_confirmation'
REQUEST_INPUT_FUNCTION_CALL_NAME = 'adk_request_input'

logger = logging.getLogger('google_adk.' + __name__)

_MODEL_TOOL_TEXT_MAX_ITEMS = 20
_MODEL_TOOL_TEXT_MAX_MATCHES = 8
_MODEL_TOOL_TEXT_MAX_LINES = 120
_MODEL_TOOL_TEXT_MAX_CHARS = 24000
_MODEL_TOOL_HEAD_TAIL_ITEMS = 10
_MODEL_TOOL_OUTPUT_CHAR_LIMIT = 500000
_MODEL_TOOL_OUTPUT_OFFLOAD_DIR = '.tmp/tool_outputs'


def _short_text(value: object) -> str:
  if value is None:
    return ''
  if isinstance(value, str):
    return value
  try:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
  except TypeError:
    return str(value)


def _append_if(lines: list[str], label: str, value: object) -> None:
  text = _short_text(value).strip()
  if text:
    lines.append(f'{label}: {text}')


def _head_tail_items(
    items: list[str],
    *,
    max_items: int,
    omitted_label: str,
) -> list[str]:
  if len(items) <= max_items:
    return items
  head_count = max_items // 2
  tail_count = max_items - head_count
  omitted = len(items) - head_count - tail_count
  return (
      items[:head_count]
      + [f'... omitted {omitted} {omitted_label}']
      + items[-tail_count:]
  )


def _truncate_text_for_model(value: object) -> str:
  text = _short_text(value)
  if not text:
    return ''
  lines = text.splitlines()
  omitted_lines = 0
  if len(lines) > _MODEL_TOOL_TEXT_MAX_LINES:
    omitted_lines = len(lines) - _MODEL_TOOL_TEXT_MAX_LINES
    head_count = max(_MODEL_TOOL_TEXT_MAX_LINES // 2, 1)
    tail_count = max(_MODEL_TOOL_TEXT_MAX_LINES - head_count - 1, 1)
    lines = (
        lines[:head_count]
        + [f'... omitted {omitted_lines} lines ...']
        + lines[-tail_count:]
    )
    text = '\n'.join(lines)
  if len(text) > _MODEL_TOOL_TEXT_MAX_CHARS:
    omitted_chars = len(text) - _MODEL_TOOL_TEXT_MAX_CHARS
    keep_head = max(_MODEL_TOOL_TEXT_MAX_CHARS // 2, 1)
    keep_tail = max(_MODEL_TOOL_TEXT_MAX_CHARS - keep_head - 32, 1)
    text = (
        text[:keep_head]
        + f'\n... omitted {omitted_chars} chars ...\n'
        + text[-keep_tail:]
    )
  return text


def _workspace_relative_display_path(path: Path, workspace_root: Path) -> str:
  try:
    return str(path.relative_to(workspace_root))
  except ValueError:
    return str(path)


def _offload_large_tool_output(
    tool_name: str,
    function_result: dict[str, object],
) -> dict[str, object]:
  output = function_result.get('output')
  if not isinstance(output, str):
    return function_result
  if len(output) <= _MODEL_TOOL_OUTPUT_CHAR_LIMIT:
    return function_result

  workspace_raw = os.environ.get('CODING_AGENT_WORKSPACE')
  workspace_root = (
      Path(workspace_raw).expanduser().resolve()
      if workspace_raw
      else Path.cwd().resolve()
  )
  offload_dir = workspace_root / _MODEL_TOOL_OUTPUT_OFFLOAD_DIR
  offload_dir.mkdir(parents=True, exist_ok=True)
  offload_path = offload_dir / (
      f'{tool_name}-{platform_uuid.new_uuid().replace("-", "")}.txt'
  )
  offload_path.write_text(output, encoding='utf-8')

  display_path = _workspace_relative_display_path(offload_path, workspace_root)
  omitted_chars = len(output) - _MODEL_TOOL_OUTPUT_CHAR_LIMIT
  keep_head = min(4000, _MODEL_TOOL_OUTPUT_CHAR_LIMIT // 2)
  keep_tail = min(4000, _MODEL_TOOL_OUTPUT_CHAR_LIMIT // 2)
  note = (
      f'[tool output truncated: original output exceeded '
      f'{_MODEL_TOOL_OUTPUT_CHAR_LIMIT} chars by {omitted_chars} chars. '
      f'Full output saved to {display_path}]'
  )
  available = max(_MODEL_TOOL_OUTPUT_CHAR_LIMIT - len(note) - 4, 0)
  if available <= 0:
    truncated_output = note
  else:
    head = min(keep_head, max(available // 2, 1))
    tail = min(keep_tail, max(available - head, 1))
    if len(output) <= head + tail:
      truncated_output = output[:available]
    else:
      truncated_output = (
          output[:head].rstrip()
          + '\n\n'
          + note
          + '\n\n'
          + output[-tail:].lstrip()
      )

  result = dict(function_result)
  result['output'] = truncated_output
  result['truncated'] = True
  result['full_output_path'] = display_path
  result['original_output_chars'] = len(output)
  result['output_char_limit'] = _MODEL_TOOL_OUTPUT_CHAR_LIMIT
  return result


def _render_tool_output(
    *,
    tool_name: str,
    status: str,
    key_facts: list[str] | None = None,
    content_blocks: list[str] | None = None,
    error_text: str | None = None,
) -> dict[str, object]:
  lines: list[str] = [f'Tool: {tool_name}', f'Status: {status}']
  if key_facts:
    filtered = [fact for fact in key_facts if fact and fact.strip()]
    if filtered:
      lines.append('Key facts:')
      for fact in filtered:
        lines.append(f'- {fact}')
  if content_blocks:
    filtered_blocks = [block for block in content_blocks if block and block.strip()]
    if filtered_blocks:
      lines.append('Content:')
      for index, block in enumerate(filtered_blocks):
        if index > 0:
          lines.append('')
        lines.append(block)
  if error_text and error_text.strip():
    lines.append('Error:')
    lines.append(error_text.strip())
  return {'output': '\n'.join(lines)}


def _context_usage_payload(
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> dict[str, object] | None:
  try:
    from ...apps.compaction import _latest_prompt_token_count
  except Exception:
    return None

  context_limit_raw = os.environ.get('CODING_AGENT_COMPACT_CONTEXT_LIMIT')
  try:
    context_limit = int(context_limit_raw) if context_limit_raw else 1048576
  except ValueError:
    context_limit = 1048576
  if context_limit <= 0:
    context_limit = 1048576

  prompt_tokens = _latest_prompt_token_count(
      tool_context._invocation_context.session.events,
      current_branch=getattr(invocation_context, 'branch', None),
      agent_name=invocation_context.agent.name,
  )
  if prompt_tokens is None:
    return None

  usage_percent = round((prompt_tokens / context_limit) * 100.0, 2)
  return {
      'context_usage_percent': usage_percent,
      'context_usage_prompt_tokens': prompt_tokens,
      'context_limit_tokens': context_limit,
  }


def _append_context_usage_to_result(
    result: dict[str, object],
    *,
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> dict[str, object]:
  usage_payload = _context_usage_payload(tool_context, invocation_context)
  if not usage_payload:
    return result

  updated = dict(result)
  updated.update(usage_payload)

  output = updated.get('output')
  usage_line = (
      f"Context usage: {usage_payload['context_usage_percent']}% "
      f"({usage_payload['context_usage_prompt_tokens']}/{usage_payload['context_limit_tokens']} prompt tokens)"
  )
  if isinstance(output, str) and output.strip():
    updated['output'] = output.rstrip() + '\n\n' + usage_line
  else:
    updated['output'] = usage_line
  return updated


def _format_match(match: object) -> str:
  if not isinstance(match, dict):
    return _short_text(match)
  path = match.get('path')
  line = match.get('line')
  header = []
  if path:
    header.append(str(path))
  if line is not None:
    header.append(f'line {line}')
  matched_keywords = match.get('matched_keywords')
  if isinstance(matched_keywords, list) and matched_keywords:
    header.append('keywords=' + ', '.join(str(x) for x in matched_keywords))
  header_text = ' | '.join(header)
  body = match.get('file_content') or match.get('context') or match.get('text')
  if header_text and body:
    return f'{header_text}\n{_short_text(body)}'
  if header_text:
    return header_text
  return _short_text(body)


def _format_read_file_output(function_result: dict[str, object], status: str, error_text: str | None) -> dict[str, object]:
  path = _short_text(function_result.get('path')).strip()
  start = function_result.get('start_line')
  end = function_result.get('end_line')
  content = _short_text(function_result.get('content')).strip()
  lines: list[str] = []
  if path:
    if start is not None or end is not None:
      lines.append(f'{path}:{start}-{end}')
    else:
      lines.append(path)
  if status != 'success':
    lines.append(f'status: {status}')
  if error_text:
    lines.append(error_text)
  if content:
    if lines:
      lines.append('')
    lines.append(content)
  return {'output': '\n'.join(lines).strip()}


def _format_search_output(tool_name: str, function_result: dict[str, object], status: str, error_text: str | None) -> dict[str, object]:
  path = _short_text(function_result.get('path')).strip()
  pattern = _short_text(function_result.get('pattern')).strip()
  keywords = function_result.get('keywords')
  summary_bits: list[str] = []
  if path:
    summary_bits.append(f'path={path}')
  if pattern:
    summary_bits.append(f'pattern={pattern}')
  if isinstance(keywords, list) and keywords:
    summary_bits.append('keywords=' + ', '.join(_short_text(x) for x in keywords))
  for key in ('matched_files', 'results_count', 'matches_count'):
    if key in function_result:
      summary_bits.append(f'{key}={function_result.get(key)}')
  if function_result.get('truncated') is not None:
    summary_bits.append(f'truncated={function_result.get("truncated")}')

  lines: list[str] = []
  if summary_bits:
    lines.append(f'{tool_name}: ' + ', '.join(summary_bits))
  elif status != 'success':
    lines.append(f'{tool_name}: status={status}')
  if error_text:
    lines.append(error_text)

  matches = function_result.get('matches')
  if isinstance(matches, list) and matches:
    if lines:
      lines.append('')
    rendered_matches = [_format_match(match) for match in matches]
    rendered_matches = _head_tail_items(
        rendered_matches,
        max_items=_MODEL_TOOL_TEXT_MAX_MATCHES,
        omitted_label='matches',
    )
    for index, match_text in enumerate(rendered_matches, start=1):
      if index > 1:
        lines.append('')
      lines.append(match_text)

  results = function_result.get('results')
  if isinstance(results, list) and results:
    if lines:
      lines.append('')
    rendered_results = [_format_match(result) for result in results]
    rendered_results = _head_tail_items(
        rendered_results,
        max_items=_MODEL_TOOL_TEXT_MAX_MATCHES,
        omitted_label='results',
    )
    for index, result_text in enumerate(rendered_results, start=1):
      if index > 1:
        lines.append('')
      lines.append(result_text)
  return {'output': '\n'.join(lines).strip()}


def _format_patch_output(tool_name: str, function_result: dict[str, object], status: str, error_text: str | None) -> dict[str, object]:
  path = _short_text(function_result.get('filePath') or function_result.get('path')).strip()
  action_type = _short_text(function_result.get('type')).strip() or (
      'update' if tool_name == 'patch_file' else 'write'
  )
  heading = (
      f'Created {path}'
      if action_type == 'create' and path
      else f'Updated {path}'
      if path
      else tool_name
  )
  key_facts: list[str] = []
  if function_result.get('match_strategy') not in (None, ''):
    key_facts.append(f"strategy={_short_text(function_result.get('match_strategy')).strip()}")
  if function_result.get('match_count') is not None:
    key_facts.append(f"matches={function_result.get('match_count')}")
  if function_result.get('line_count') is not None:
    key_facts.append(f"lines={function_result.get('line_count')}")
  if function_result.get('bytes_written') is not None:
    key_facts.append(f"bytes_written={function_result.get('bytes_written')}")
  if status != 'success':
    key_facts.append(f'status={status}')
  lines = [heading]
  if key_facts:
    lines.append(', '.join(key_facts))
  if error_text:
    lines.append(error_text)
  patch_text = _truncate_text_for_model(function_result.get('patch')).strip()
  if patch_text:
    lines.append('')
    lines.append('Patch:')
    lines.append(patch_text)
  corrected = _short_text(function_result.get('corrected_old_text')).strip()
  if corrected:
    lines.append('')
    lines.append('corrected_old_text:')
    lines.append(corrected)
  return {'output': '\n'.join(lines).strip()}


def _format_web_search_output(function_result: dict[str, object], status: str, error_text: str | None) -> dict[str, object]:
  query = _short_text(function_result.get('query')).strip()
  summary = _short_text(function_result.get('summary')).strip()
  lines: list[str] = []
  if query:
    lines.append(f'Web search for: {query}')
  if status != 'success':
    lines.append(f'status: {status}')
  if error_text:
    lines.append(error_text)
  if summary:
    if lines:
      lines.append('')
    lines.append(summary)
  sources = function_result.get('sources')
  if isinstance(sources, list) and sources:
    if lines:
      lines.append('')
    lines.append('Sources:')
    rendered_sources: list[str] = []
    for index, source in enumerate(sources, start=1):
      if isinstance(source, dict):
        title = _short_text(source.get('title')).strip() or 'Untitled'
        uri = _short_text(source.get('uri')).strip()
        if uri:
          rendered_sources.append(f'[{index}] {title} ({uri})')
        else:
          rendered_sources.append(f'[{index}] {title}')
      else:
        rendered_sources.append(f'[{index}] {_short_text(source)}')
    lines.extend(
        _head_tail_items(
            rendered_sources,
            max_items=_MODEL_TOOL_HEAD_TAIL_ITEMS,
            omitted_label='sources',
        )
    )
  return {'output': '\n'.join(lines).strip()}


def _format_web_fetch_output(function_result: dict[str, object], status: str, error_text: str | None) -> dict[str, object]:
  url = _short_text(function_result.get('url')).strip()
  final_url = _short_text(function_result.get('final_url')).strip()
  result = _short_text(function_result.get('result')).strip()
  lines: list[str] = []
  if url:
    lines.append(f'Web fetch: {url}')
  if final_url and final_url != url:
    lines.append(f'Final URL: {final_url}')
  meta_parts: list[str] = []
  for key in ('code', 'code_text', 'content_type', 'bytes'):
    value = function_result.get(key)
    if value not in (None, ''):
      meta_parts.append(f'{key}={_short_text(value).strip()}')
  if meta_parts:
    lines.append(', '.join(meta_parts))
  if status != 'success':
    lines.append(f'status: {status}')
  if error_text:
    lines.append(error_text)
  if result:
    if lines:
      lines.append('')
    lines.append(result)
  return {'output': '\n'.join(lines).strip()}


def _format_lsp_output(function_result: dict[str, object], status: str, error_text: str | None) -> dict[str, object]:
  operation = _short_text(function_result.get('operation')).strip()
  file_path = _short_text(function_result.get('file_path')).strip()
  query = _short_text(function_result.get('query')).strip()
  result = _short_text(function_result.get('result')).strip()
  lines: list[str] = []
  if operation:
    bits = [f'Python LSP: {operation}']
    if file_path:
      bits.append(f'file={file_path}')
    if query:
      bits.append(f'query={query}')
    lines.append(', '.join(bits))
  if function_result.get('result_count') is not None:
    lines.append(f"result_count={function_result.get('result_count')}")
  if status != 'success':
    lines.append(f'status: {status}')
  if error_text:
    lines.append(error_text)
  if result:
    if lines:
      lines.append('')
    lines.append(result)
  return {'output': '\n'.join(lines).strip()}


def _format_compact_memory_output(
    function_result: dict[str, object],
    status: str,
    error_text: str | None,
) -> dict[str, object]:
  lines: list[str] = []
  result_status = _short_text(function_result.get('status')).strip() or status
  lines.append(f'compact_memory: status={result_status}')
  for key in (
      'reason',
      'prompt',
      'current_usage_tokens',
      'compacted_event_count',
      'retained_recent_event_count',
      'archive_path',
  ):
    if key in function_result and function_result.get(key) not in (None, ''):
      lines.append(f'{key}={_short_text(function_result.get(key)).strip()}')
  output_text = _truncate_text_for_model(function_result.get('output')).strip()
  if output_text:
    lines.append('')
    lines.append(output_text)
  if error_text:
    lines.append('')
    lines.append(error_text.strip())
  return {'output': '\n'.join(lines).strip()}


def _format_evaluator_output(
    tool_name: str,
    function_result: dict[str, object],
    status: str,
    error_text: str | None,
) -> dict[str, object]:
  key_facts: list[str] = []
  for key in (
      'exit_code',
      'evaluator_log_path',
      'working_directory',
      'original_repository_path',
      'candidate_repository_path',
  ):
    if key in function_result and function_result.get(key) not in (None, ''):
      key_facts.append(f'{key}: {_short_text(function_result.get(key))}')

  content_blocks: list[str] = []
  output_text = _truncate_text_for_model(function_result.get('output')).strip()
  if output_text:
    content_blocks.append('Evaluator output:\n' + output_text)

  recent_artifacts = function_result.get('recent_artifacts')
  if isinstance(recent_artifacts, list) and recent_artifacts:
    artifact_lines = [f'- {_short_text(item)}' for item in recent_artifacts]
    artifact_lines = _head_tail_items(
        artifact_lines,
        max_items=_MODEL_TOOL_HEAD_TAIL_ITEMS,
        omitted_label='artifacts',
    )
    content_blocks.append('Recent artifacts:\n' + '\n'.join(artifact_lines))

  return _render_tool_output(
      tool_name=tool_name,
      status=status,
      key_facts=key_facts,
      content_blocks=content_blocks,
      error_text=error_text,
  )


def _format_tool_result_for_model(
    tool_name: str,
    function_result: dict[str, object],
) -> dict[str, object]:
  status = _short_text(function_result.get('status')).strip() or 'success'
  error_text = _short_text(function_result.get('error_message')).strip() or None

  if tool_name == 'read_file':
    return _format_read_file_output(function_result, status, error_text)

  if tool_name == 'list_files':
    key_facts: list[str] = []
    if function_result.get('path'):
      key_facts.append(f'path: {_short_text(function_result.get("path"))}')
    entries = function_result.get('entries')
    if isinstance(entries, list):
      key_facts.append(f'entries_count: {len(entries)}')
      if function_result.get('truncated') is not None:
        key_facts.append(f'truncated: {function_result.get("truncated")}')
    content_blocks = []
    if isinstance(entries, list) and entries:
      entry_lines = [f'- {entry}' for entry in entries]
      entry_lines = _head_tail_items(
          entry_lines,
          max_items=_MODEL_TOOL_HEAD_TAIL_ITEMS,
          omitted_label='entries',
      )
      content_blocks.append('\n'.join(entry_lines))
    return _render_tool_output(
        tool_name=tool_name,
        status=status,
        key_facts=key_facts,
        content_blocks=content_blocks,
        error_text=error_text,
    )

  if tool_name in ('search_text', 'keyword_search'):
    return _format_search_output(tool_name, function_result, status, error_text)

  if tool_name == 'google_web_search':
    return _format_web_search_output(function_result, status, error_text)

  if tool_name == 'web_fetch':
    return _format_web_fetch_output(function_result, status, error_text)

  if tool_name == 'lsp_tool':
    return _format_lsp_output(function_result, status, error_text)

  if tool_name == 'compact_memory':
    return _format_compact_memory_output(function_result, status, error_text)

  if tool_name == 'report_truncated_model_output':
    key_facts: list[str] = []
    if function_result.get('max_output_tokens') not in (None, ''):
      key_facts.append(f"max_output_tokens: {_short_text(function_result.get('max_output_tokens'))}")
    content_blocks: list[str] = []
    if function_result.get('error_message'):
      content_blocks.append(_short_text(function_result.get('error_message')).strip())
    if function_result.get('retry_guidance'):
      content_blocks.append('Retry guidance:\n' + _short_text(function_result.get('retry_guidance')).strip())
    return _render_tool_output(
        tool_name=tool_name,
        status=status,
        key_facts=key_facts,
        content_blocks=content_blocks,
        error_text=None,
    )

  if tool_name in ('run_evaluator_command', 'run_coding_agent'):
    return _format_evaluator_output(tool_name, function_result, status, error_text)

  if tool_name in ('bash_exec', 'read_command_session', 'wait'):
    key_facts: list[str] = []
    for key in ('exit_code', 'session_id', 'command'):
      if key in function_result and function_result.get(key) not in (None, ''):
        key_facts.append(f'{key}: {_short_text(function_result.get(key))}')
    if function_result.get('backgrounded') is not None:
      key_facts.append(f'backgrounded: {_short_text(function_result.get("backgrounded"))}')
    content_blocks: list[str] = []
    output_context_lines = function_result.get('output_context_lines')
    output_new_lines = function_result.get('output_new_lines')
    output_lines = function_result.get('output_lines')
    if isinstance(output_context_lines, list) and output_context_lines:
      content_blocks.append(
          'Recent output:\n' + '\n'.join(_short_text(line) for line in output_context_lines)
      )
    if isinstance(output_new_lines, list) and output_new_lines:
      content_blocks.append(
          'New output:\n' + '\n'.join(_short_text(line) for line in output_new_lines)
      )
    elif isinstance(output_lines, list) and output_lines:
      content_blocks.append(
          'Output:\n' + '\n'.join(_short_text(line) for line in output_lines)
      )
    if function_result.get('output'):
      content_blocks.append('Output:\n' + _truncate_text_for_model(function_result.get('output')))
    if function_result.get('message'):
      content_blocks.append(_short_text(function_result.get('message')).strip())
    return _render_tool_output(
        tool_name=tool_name,
        status=status,
        key_facts=key_facts,
        content_blocks=content_blocks,
        error_text=error_text,
    )

  if tool_name == 'check_running_sessions':
    key_facts: list[str] = []
    for key in ('running_count', 'returned_sessions', 'truncated'):
      if key in function_result and function_result.get(key) not in (None, ''):
        key_facts.append(f'{key}: {_short_text(function_result.get(key))}')
    content_blocks: list[str] = []
    sessions = function_result.get('sessions')
    if isinstance(sessions, list) and sessions:
      block_lines: list[str] = []
      for session in sessions:
        if not isinstance(session, dict):
          continue
        session_id = _short_text(session.get('session_id')).strip()
        command = _short_text(session.get('command')).strip()
        running_seconds = _short_text(session.get('running_seconds')).strip()
        pid = _short_text(session.get('pid')).strip()
        status_text = _short_text(session.get('command_status')).strip()
        parts = [part for part in (
            f'session_id={session_id}' if session_id else '',
            f'status={status_text}' if status_text else '',
            f'pid={pid}' if pid else '',
            f'running_seconds={running_seconds}' if running_seconds else '',
            f'command={command}' if command else '',
        ) if part]
        block_lines.append('- ' + ' | '.join(parts))
      block_lines = _head_tail_items(
          block_lines,
          max_items=_MODEL_TOOL_HEAD_TAIL_ITEMS,
          omitted_label='sessions',
      )
      if block_lines:
        content_blocks.append('\n'.join(block_lines))
    return _render_tool_output(
        tool_name=tool_name,
        status=status,
        key_facts=key_facts,
        content_blocks=content_blocks,
        error_text=error_text,
    )

  if tool_name == 'run_bash':
    content_blocks: list[str] = []
    output_context_lines = function_result.get('output_context_lines')
    output_new_lines = function_result.get('output_new_lines')
    output_lines = function_result.get('output_lines')
    if isinstance(output_context_lines, list) and output_context_lines:
      content_blocks.append(
          'Recent output:\n' + '\n'.join(_short_text(line) for line in output_context_lines)
      )
    if isinstance(output_new_lines, list) and output_new_lines:
      content_blocks.append(
          'New output:\n' + '\n'.join(_short_text(line) for line in output_new_lines)
      )
    elif isinstance(output_lines, list) and output_lines:
      content_blocks.append(
          'Output:\n' + '\n'.join(_short_text(line) for line in output_lines)
      )

    if function_result.get('backgrounded'):
      session_id = _short_text(function_result.get('session_id')).strip()
      message = _short_text(function_result.get('message')).strip()
      suffix_lines = []
      if message:
        suffix_lines.append(message)
      if session_id:
        suffix_lines.append(f'session_id: {session_id}')
      if suffix_lines:
        content_blocks.append('\n'.join(suffix_lines))

    if not content_blocks and error_text:
      content_blocks.append(error_text)

    return {'output': '\n\n'.join(block for block in content_blocks if block).strip()}

  if tool_name == 'python_exec':
    key_facts: list[str] = []
    for key in ('exit_code',):
      if key in function_result and function_result.get(key) not in (None, ''):
        key_facts.append(f'{key}: {_short_text(function_result.get(key))}')
    content_blocks: list[str] = []
    if function_result.get('stdout'):
      content_blocks.append('Stdout:\n' + _truncate_text_for_model(function_result.get('stdout')))
    if function_result.get('stderr'):
      content_blocks.append('Stderr:\n' + _truncate_text_for_model(function_result.get('stderr')))
    return _render_tool_output(
        tool_name=tool_name,
        status=status,
        key_facts=key_facts,
        content_blocks=content_blocks,
        error_text=error_text,
    )

  if tool_name in ('patch_file', 'write_file'):
    return _format_patch_output(tool_name, function_result, status, error_text)

  if tool_name in ('write_todos', 'read_todos'):
    todos = function_result.get('todos')
    if isinstance(todos, list):
      key_facts = [f'todos_count: {len(todos)}']
      if function_result.get('agent_name') not in (None, ''):
        key_facts.append(f'agent_name: {_short_text(function_result.get("agent_name"))}')
      if function_result.get('todo_file') not in (None, ''):
        key_facts.append(f'todo_file: {_short_text(function_result.get("todo_file"))}')
      block_lines: list[str] = []
      for todo in todos:
        if isinstance(todo, dict):
          status_text = _short_text(todo.get('status')).strip()
          task_text = _short_text(
              todo.get('description') or todo.get('content') or todo.get('task')
          ).strip()
          block_lines.append(f'- [{status_text}] {task_text}')
        else:
          block_lines.append(f'- {_short_text(todo)}')
      block_lines = _head_tail_items(
          block_lines,
          max_items=_MODEL_TOOL_HEAD_TAIL_ITEMS,
          omitted_label='todos',
      )
      return _render_tool_output(
          tool_name=tool_name,
          status=status,
          key_facts=key_facts,
          content_blocks=['\n'.join(block_lines)] if block_lines else None,
          error_text=error_text,
      )

  if tool_name in ('list_skills', 'activate_skill', 'terminate_command_session'):
    key_facts = []
    preferred = [
        'path',
        'session_id',
        'command_status',
        'backgrounded',
        'message',
        'summary',
        'result',
        'bytes_written',
        'running_seconds',
        'initial_wait_seconds',
    ]
    for key in preferred:
      if key in function_result:
        key_facts.append(f'{key}: {_short_text(function_result.get(key))}')
    content_blocks = []
    output_context_lines = function_result.get('output_context_lines')
    output_new_lines = function_result.get('output_new_lines')
    output_lines = function_result.get('output_lines')
    if isinstance(output_context_lines, list) and output_context_lines:
      content_blocks.append(
          'Recent output:\n' + '\n'.join(_short_text(line) for line in output_context_lines)
      )
    if isinstance(output_new_lines, list) and output_new_lines:
      content_blocks.append(
          'New output:\n' + '\n'.join(_short_text(line) for line in output_new_lines)
      )
    elif isinstance(output_lines, list) and output_lines:
      content_blocks.append(
          'Output:\n' + '\n'.join(_short_text(line) for line in output_lines)
      )
    return _render_tool_output(
        tool_name=tool_name,
        status=status,
        key_facts=key_facts,
        content_blocks=content_blocks,
        error_text=error_text,
    )

  preferred_keys = [
      'path',
      'message',
      'summary',
      'result',
      'content',
      'output',
      'bytes_written',
      'match_count',
  ]
  key_facts = []
  content_blocks = []
  for key in preferred_keys:
    if key in function_result:
      value = function_result.get(key)
      if key in ('content', 'output') and value:
        content_blocks.append(f'{key.capitalize()}:\n{_truncate_text_for_model(value)}')
      else:
        text = _short_text(value).strip()
        if text:
          key_facts.append(f'{key}: {text}')
  return _render_tool_output(
      tool_name=tool_name,
      status=status,
      key_facts=key_facts,
      content_blocks=content_blocks,
      error_text=error_text,
  )


def _print_tool_call_batch_counts() -> bool:
  value = os.environ.get('ADK_PRINT_TOOL_CALL_BATCH_COUNTS')
  if value is None:
    return False
  return value.strip().lower() in ('1', 'true', 'yes', 'on')

# Global thread pool executors for running tools in background threads.
# This prevents blocking tools from blocking the event loop in Live API mode.
# Key is max_workers, value is the executor.
_TOOL_THREAD_POOLS: dict[int, ThreadPoolExecutor] = {}
_TOOL_THREAD_POOL_LOCK = threading.Lock()


def _is_live_request_queue_annotation(param: inspect.Parameter) -> bool:
  """Check whether a parameter is annotated as LiveRequestQueue.

  Handles both the class itself and the string form produced by
  ``from __future__ import annotations``.
  """
  ann = param.annotation
  return ann is LiveRequestQueue or (
      isinstance(ann, str) and ann == 'LiveRequestQueue'
  )


def _get_tool_thread_pool(max_workers: int = 4) -> ThreadPoolExecutor:
  """Gets or creates a thread pool executor for tool execution.

  Args:
    max_workers: Maximum number of worker threads in the pool.

  Returns:
    A ThreadPoolExecutor with the specified max_workers.
  """
  if max_workers not in _TOOL_THREAD_POOLS:
    with _TOOL_THREAD_POOL_LOCK:
      if max_workers not in _TOOL_THREAD_POOLS:
        _TOOL_THREAD_POOLS[max_workers] = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix='adk_tool_executor'
        )
  return _TOOL_THREAD_POOLS[max_workers]


def _is_sync_tool(tool: BaseTool) -> bool:
  """Checks if a tool's underlying function is synchronous."""
  if not hasattr(tool, 'func'):
    return False
  func = tool.func
  return not (
      inspect.iscoroutinefunction(func)
      or inspect.isasyncgenfunction(func)
      or (
          hasattr(func, '__call__')
          and inspect.iscoroutinefunction(func.__call__)
      )
  )


async def _call_tool_in_thread_pool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    max_workers: int = 4,
) -> Any:
  """Runs a tool in a thread pool to avoid blocking the event loop.

  For sync tools, this runs the tool's function directly in a background thread.
  For async tools, this creates a new event loop in the background thread and
  runs the async function there. This helps catch blocking I/O (like time.sleep,
  network calls, file I/O) that was mistakenly used inside async functions.

  Note: Due to Python's GIL, this does NOT help with pure Python CPU-bound code.
  Thread pool only helps when the GIL is released (blocking I/O, C extensions).

  Args:
    tool: The tool to execute.
    args: Arguments to pass to the tool.
    tool_context: The tool context.
    max_workers: Maximum number of worker threads in the pool.

  Returns:
    The result of running the tool.
  """
  from ...tools.function_tool import FunctionTool

  loop = asyncio.get_running_loop()
  executor = _get_tool_thread_pool(max_workers)

  if _is_sync_tool(tool):
    # For sync FunctionTool, call the underlying function directly
    def run_sync_tool():
      if isinstance(tool, FunctionTool):
        args_to_call = tool._preprocess_args(args)
        signature = inspect.signature(tool.func)
        valid_params = {param for param in signature.parameters}
        if tool._context_param_name in valid_params:
          args_to_call[tool._context_param_name] = tool_context
        args_to_call = {
            k: v for k, v in args_to_call.items() if k in valid_params
        }
        return tool.func(**args_to_call)
      else:
        # For other sync tool types, we can't easily run them in thread pool
        return None

    result = await loop.run_in_executor(executor, run_sync_tool)
    if result is not None:
      return result
  else:
    # For async tools, run them in a new event loop in a background thread.
    # This helps when async functions contain blocking I/O (common user mistake)
    # that would otherwise block the main event loop.
    def run_async_tool_in_new_loop():
      # Create a new event loop for this thread
      return asyncio.run(tool.run_async(args=args, tool_context=tool_context))

    return await loop.run_in_executor(executor, run_async_tool_in_new_loop)

  # Fall back to normal async execution for non-FunctionTool sync tools
  return await tool.run_async(args=args, tool_context=tool_context)


def generate_client_function_call_id() -> str:
  return f'{AF_FUNCTION_CALL_ID_PREFIX}{platform_uuid.new_uuid()}'


def populate_client_function_call_id(model_response_event: Event) -> None:
  if not model_response_event.get_function_calls():
    return
  for function_call in model_response_event.get_function_calls():
    if not function_call.id:
      function_call.id = generate_client_function_call_id()


def remove_client_function_call_id(content: Optional[types.Content]) -> None:
  """Removes ADK-generated function call IDs from content before sending to LLM.

  Strips client-side function call/response IDs that start with 'adk-' prefix
  to avoid sending internal tracking IDs to the model.

  Args:
    content: Content containing function calls/responses to clean.
  """
  if content and content.parts:
    for part in content.parts:
      if (
          part.function_call
          and part.function_call.id
          and part.function_call.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_call.id = None
      if (
          part.function_response
          and part.function_response.id
          and part.function_response.id.startswith(AF_FUNCTION_CALL_ID_PREFIX)
      ):
        part.function_response.id = None


def get_long_running_function_calls(
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
) -> set[str]:
  long_running_tool_ids = set()
  for function_call in function_calls:
    if (
        function_call.name in tools_dict
        and tools_dict[function_call.name].is_long_running
    ):
      long_running_tool_ids.add(function_call.id)

  return long_running_tool_ids


def build_auth_request_event(
    invocation_context: InvocationContext,
    auth_requests: Dict[str, AuthConfig],
    *,
    author: Optional[str] = None,
    role: Optional[str] = None,
) -> Event:
  """Builds an auth request event with function calls for each auth request.

  This is a shared helper used by both tool-level auth (when a tool requests
  auth during execution) and toolset-level auth (before tool listing).

  Args:
    invocation_context: The invocation context.
    auth_requests: Dict mapping function_call_id to AuthConfig.
    author: The event author. Defaults to agent name.
    role: The content role. Defaults to None.

  Returns:
    Event with auth request function calls.
  """
  parts = []
  long_running_tool_ids = set()

  for function_call_id, auth_config in auth_requests.items():
    request_euc_function_call = types.FunctionCall(
        name=REQUEST_EUC_FUNCTION_CALL_NAME,
        id=generate_client_function_call_id(),
        args=AuthToolArguments(
            function_call_id=function_call_id,
            auth_config=auth_config,
        ).model_dump(exclude_none=True, by_alias=True),
    )
    long_running_tool_ids.add(request_euc_function_call.id)
    parts.append(types.Part(function_call=request_euc_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=author or invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(parts=parts, role=role),
      long_running_tool_ids=long_running_tool_ids,
  )


def generate_auth_event(
    invocation_context: InvocationContext,
    function_response_event: Event,
) -> Optional[Event]:
  """Generates an auth request event from a function response event.

  This is used for tool-level auth where a tool requests credentials during
  execution.

  Args:
    invocation_context: The invocation context.
    function_response_event: The function response event with auth requests.

  Returns:
    Event with auth request function calls, or None if no auth requested.
  """
  if not function_response_event.actions.requested_auth_configs:
    return None

  return build_auth_request_event(
      invocation_context,
      function_response_event.actions.requested_auth_configs,
      role=function_response_event.content.role,
  )


def generate_request_confirmation_event(
    invocation_context: InvocationContext,
    function_call_event: Event,
    function_response_event: Event,
) -> Optional[Event]:
  """Generates a request confirmation event from a function response event."""
  if not function_response_event.actions.requested_tool_confirmations:
    return None
  parts = []
  long_running_tool_ids = set()
  function_calls = function_call_event.get_function_calls()
  for (
      function_call_id,
      tool_confirmation,
  ) in function_response_event.actions.requested_tool_confirmations.items():
    original_function_call = next(
        (fc for fc in function_calls if fc.id == function_call_id), None
    )
    if not original_function_call:
      continue
    request_confirmation_function_call = types.FunctionCall(
        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        args={
            'originalFunctionCall': original_function_call.model_dump(
                exclude_none=True, by_alias=True
            ),
            'toolConfirmation': tool_confirmation.model_dump(
                by_alias=True, exclude_none=True
            ),
        },
    )
    request_confirmation_function_call.id = generate_client_function_call_id()
    long_running_tool_ids.add(request_confirmation_function_call.id)
    parts.append(types.Part(function_call=request_confirmation_function_call))

  return Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      branch=invocation_context.branch,
      content=types.Content(
          parts=parts, role=function_response_event.content.role
      ),
      long_running_tool_ids=long_running_tool_ids,
  )


async def handle_function_calls_async(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  function_calls = function_call_event.get_function_calls()
  return await handle_function_call_list_async(
      invocation_context,
      function_calls,
      tools_dict,
      filters,
      tool_confirmation_dict,
  )


async def handle_function_call_list_async(
    invocation_context: InvocationContext,
    function_calls: list[types.FunctionCall],
    tools_dict: dict[str, BaseTool],
    filters: Optional[set[str]] = None,
    tool_confirmation_dict: Optional[dict[str, ToolConfirmation]] = None,
) -> Optional[Event]:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = invocation_context.agent

  # Filter function calls
  filtered_calls = [
      fc for fc in function_calls if not filters or fc.id in filters
  ]

  if not filtered_calls:
    return None

  if _print_tool_call_batch_counts():
    tool_names = [fc.name or '<unknown>' for fc in filtered_calls]
    print(
        f'  [{invocation_context.agent.name} batch] '
        f'count={len(filtered_calls)} names={tool_names}',
        flush=True,
    )

  # Create tasks for parallel execution
  tasks = [
      asyncio.create_task(
          _execute_single_function_call_async(
              invocation_context,
              function_call,
              tools_dict,
              agent,
              tool_confirmation_dict[function_call.id]
              if tool_confirmation_dict
              else None,
          )
      )
      for function_call in filtered_calls
  ]

  # Wait for all tasks to complete
  function_response_events = await asyncio.gather(*tasks)

  # Filter out None results
  function_response_events = [
      event for event in function_response_events if event is not None
  ]

  if not function_response_events:
    return None

  merged_event = merge_parallel_function_response_events(
      function_response_events
  )

  if len(function_response_events) > 1:
    # this is needed for debug traces of parallel calls
    # individual response with tool.name is traced in __build_response_event
    # (we drop tool.name from span name here as this is merged event)
    with tracer.start_as_current_span('execute_tool (merged)'):
      trace_merged_tool_calls(
          response_event_id=merged_event.id,
          function_response_event=merged_event,
      )
  return merged_event


async def _execute_single_function_call_async(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    tool_confirmation: Optional[ToolConfirmation] = None,
) -> Optional[Event]:
  """Execute a single function call with thread safety for state modifications."""

  async def _run_on_tool_error_callbacks(
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict[str, Any]]:
    """Runs the on_tool_error_callbacks for the given tool."""
    error_response = (
        await invocation_context.plugin_manager.run_on_tool_error_callback(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
            error=error,
        )
    )
    if error_response is not None:
      return error_response

    for callback in agent.canonical_on_tool_error_callbacks:
      error_response = callback(
          tool=tool,
          args=tool_args,
          tool_context=tool_context,
          error=error,
      )
      if inspect.isawaitable(error_response):
        error_response = await error_response
      if error_response is not None:
        return error_response

    return None

  # Do not use "args" as the variable name, because it is a reserved keyword
  # in python debugger.
  # Make a deep copy to avoid being modified.
  function_args = (
      copy.deepcopy(function_call.args) if function_call.args else {}
  )

  tool_context = _create_tool_context(
      invocation_context, function_call, tool_confirmation
  )

  try:
    tool = _get_tool(function_call, tools_dict)
  except ValueError as tool_error:
    tool = BaseTool(name=function_call.name, description='Tool not found')
    error_response = await _run_on_tool_error_callbacks(
        tool=tool,
        tool_args=function_args,
        tool_context=tool_context,
        error=tool_error,
    )
    if error_response is not None:
      return __build_response_event(
          tool, error_response, tool_context, invocation_context
      )
    else:
      raise tool_error

  async def _run_with_trace():
    nonlocal function_args

    # Step 1: Check if plugin before_tool_callback overrides the function
    # response.
    function_response = (
        await invocation_context.plugin_manager.run_before_tool_callback(
            tool=tool, tool_args=function_args, tool_context=tool_context
        )
    )

    # Step 2: If no overrides are provided from the plugins, further run the
    # canonical callback.
    if function_response is None:
      for callback in agent.canonical_before_tool_callbacks:
        function_response = callback(
            tool=tool, args=function_args, tool_context=tool_context
        )
        if inspect.isawaitable(function_response):
          function_response = await function_response
        if function_response:
          break

    # Step 3: Otherwise, proceed calling the tool normally.
    if function_response is None:
      try:
        function_response = await __call_tool_async(
            tool, args=function_args, tool_context=tool_context
        )
      except Exception as tool_error:
        error_response = await _run_on_tool_error_callbacks(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            error=tool_error,
        )
        if error_response is not None:
          function_response = error_response
        else:
          raise tool_error

    # Step 4: Check if plugin after_tool_callback overrides the function
    # response.
    altered_function_response = (
        await invocation_context.plugin_manager.run_after_tool_callback(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            result=function_response,
        )
    )

    # Step 5: If no overrides are provided from the plugins, further run the
    # canonical after_tool_callbacks.
    if altered_function_response is None:
      for callback in agent.canonical_after_tool_callbacks:
        altered_function_response = callback(
            tool=tool,
            args=function_args,
            tool_context=tool_context,
            tool_response=function_response,
        )
        if inspect.isawaitable(altered_function_response):
          altered_function_response = await altered_function_response
        if altered_function_response:
          break

    # Step 6: If alternative response exists from after_tool_callback, use it
    # instead of the original function response.
    if altered_function_response is not None:
      function_response = altered_function_response

    if tool.is_long_running:
      # Allow long-running function to return None to not provide function
      # response.
      if not function_response:
        return None

    # Note: State deltas are not applied here - they are collected in
    # tool_context.actions.state_delta and applied later when the session
    # service processes the events

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    return function_response_event

  with tracer.start_as_current_span(f'execute_tool {tool.name}'):
    function_response_event = None
    caught_error = None
    try:
      function_response_event = await _run_with_trace()
      return function_response_event
    except Exception as e:
      caught_error = e
      raise
    finally:
      trace_tool_call(
          tool=tool,
          args=function_args,
          function_response_event=function_response_event,
          error=caught_error,
      )


async def handle_function_calls_live(
    invocation_context: InvocationContext,
    function_call_event: Event,
    tools_dict: dict[str, BaseTool],
) -> Event:
  """Calls the functions and returns the function response event."""
  from ...agents.llm_agent import LlmAgent

  agent = cast(LlmAgent, invocation_context.agent)
  function_calls = function_call_event.get_function_calls()

  if not function_calls:
    return None

  # Create async lock for active_streaming_tools modifications
  streaming_lock = asyncio.Lock()

  # Create tasks for parallel execution
  tasks = [
      asyncio.create_task(
          _execute_single_function_call_live(
              invocation_context,
              function_call,
              tools_dict,
              agent,
              streaming_lock,
          )
      )
      for function_call in function_calls
  ]

  # Wait for all tasks to complete
  function_response_events = await asyncio.gather(*tasks)

  # Filter out None results
  function_response_events = [
      event for event in function_response_events if event is not None
  ]

  if not function_response_events:
    return None

  merged_event = merge_parallel_function_response_events(
      function_response_events
  )
  if len(function_response_events) > 1:
    # this is needed for debug traces of parallel calls
    # individual response with tool.name is traced in __build_response_event
    # (we drop tool.name from span name here as this is merged event)
    with tracer.start_as_current_span('execute_tool (merged)'):
      trace_merged_tool_calls(
          response_event_id=merged_event.id,
          function_response_event=merged_event,
      )
  return merged_event


async def _execute_single_function_call_live(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    agent: LlmAgent,
    streaming_lock: asyncio.Lock,
) -> Optional[Event]:
  """Execute a single function call for live mode with thread safety."""

  async def _run_on_tool_error_callbacks(
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict[str, Any]]:
    """Runs the on_tool_error_callbacks for the given tool."""
    error_response = (
        await invocation_context.plugin_manager.run_on_tool_error_callback(
            tool=tool,
            tool_args=tool_args,
            tool_context=tool_context,
            error=error,
        )
    )
    if error_response is not None:
      return error_response

    for callback in agent.canonical_on_tool_error_callbacks:
      error_response = callback(
          tool=tool,
          args=tool_args,
          tool_context=tool_context,
          error=error,
      )
      if inspect.isawaitable(error_response):
        error_response = await error_response
      if error_response is not None:
        return error_response

    return None

  # Do not use "args" as the variable name, because it is a reserved keyword
  # in python debugger.
  # Make a deep copy to avoid being modified.
  function_args = (
      copy.deepcopy(function_call.args) if function_call.args else {}
  )

  tool_context = _create_tool_context(invocation_context, function_call)

  try:
    tool = _get_tool(function_call, tools_dict)
  except ValueError as tool_error:
    tool = BaseTool(name=function_call.name, description='Tool not found')
    error_response = await _run_on_tool_error_callbacks(
        tool=tool,
        tool_args=function_args,
        tool_context=tool_context,
        error=tool_error,
    )
    if error_response is not None:
      return __build_response_event(
          tool, error_response, tool_context, invocation_context
      )
    raise tool_error

  async def _run_with_trace():
    nonlocal function_args

    # Do not use "args" as the variable name, because it is a reserved keyword
    # in python debugger.
    # Make a deep copy to avoid being modified.
    function_response = None

    # Step 1: Check if plugin before_tool_callback overrides the function
    # response.
    function_response = (
        await invocation_context.plugin_manager.run_before_tool_callback(
            tool=tool, tool_args=function_args, tool_context=tool_context
        )
    )

    # Step 2: If no overrides are provided from the plugins, further run the
    # canonical callback.
    if function_response is None:
      for callback in agent.canonical_before_tool_callbacks:
        function_response = callback(
            tool=tool, args=function_args, tool_context=tool_context
        )
        if inspect.isawaitable(function_response):
          function_response = await function_response
        if function_response:
          break

    # Step 3: Otherwise, proceed calling the tool normally.
    if function_response is None:
      try:
        function_response = await _process_function_live_helper(
            tool,
            tool_context,
            function_call,
            function_args,
            invocation_context,
            streaming_lock,
        )
      except Exception as tool_error:
        error_response = await _run_on_tool_error_callbacks(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            error=tool_error,
        )
        if error_response is not None:
          function_response = error_response
        else:
          raise tool_error

    # Step 4: Check if plugin after_tool_callback overrides the function
    # response.
    altered_function_response = (
        await invocation_context.plugin_manager.run_after_tool_callback(
            tool=tool,
            tool_args=function_args,
            tool_context=tool_context,
            result=function_response,
        )
    )

    # Step 5: If no overrides are provided from the plugins, further run the
    # canonical after_tool_callbacks.
    if altered_function_response is None:
      for callback in agent.canonical_after_tool_callbacks:
        altered_function_response = callback(
            tool=tool,
            args=function_args,
            tool_context=tool_context,
            tool_response=function_response,
        )
        if inspect.isawaitable(altered_function_response):
          altered_function_response = await altered_function_response
        if altered_function_response:
          break

    # Step 6: If alternative response exists from after_tool_callback, use it
    # instead of the original function response.
    if altered_function_response is not None:
      function_response = altered_function_response

    if tool.is_long_running:
      # Allow async function to return None to not provide function response.
      if not function_response:
        return None

    # Note: State deltas are not applied here - they are collected in
    # tool_context.actions.state_delta and applied later when the session
    # service processes the events

    # Builds the function response event.
    function_response_event = __build_response_event(
        tool, function_response, tool_context, invocation_context
    )
    return function_response_event

  with tracer.start_as_current_span(f'execute_tool {tool.name}'):
    function_response_event = None
    caught_error = None
    try:
      function_response_event = await _run_with_trace()
      return function_response_event
    except Exception as e:
      caught_error = e
      raise
    finally:
      trace_tool_call(
          tool=tool,
          args=function_args,
          function_response_event=function_response_event,
          error=caught_error,
      )


async def _process_function_live_helper(
    tool,
    tool_context,
    function_call,
    function_args,
    invocation_context,
    streaming_lock: asyncio.Lock,
):
  function_response = None
  # Check if this is a stop_streaming function call
  if (
      function_call.name == 'stop_streaming'
      and 'function_name' in function_args
  ):
    function_name = function_args['function_name']
    # Thread-safe access to active_streaming_tools
    async with streaming_lock:
      active_tasks = invocation_context.active_streaming_tools
      if (
          active_tasks
          and function_name in active_tasks
          and active_tasks[function_name].task
          and not active_tasks[function_name].task.done()
      ):
        task = active_tasks[function_name].task
      else:
        task = None

    if task:
      task.cancel()
      try:
        # Wait for the task to be cancelled
        await asyncio.wait_for(task, timeout=1.0)
      except (asyncio.CancelledError, asyncio.TimeoutError):
        # Log the specific condition
        if task.cancelled():
          logging.info('Task %s was cancelled successfully', function_name)
        elif task.done():
          logging.info('Task %s completed during cancellation', function_name)
        else:
          logging.warning(
              'Task %s might still be running after cancellation timeout',
              function_name,
          )
          function_response = {
              'status': f'The task is not cancelled yet for {function_name}.'
          }
      if not function_response:
        # Clean up the reference under lock
        async with streaming_lock:
          if (
              invocation_context.active_streaming_tools
              and function_name in invocation_context.active_streaming_tools
          ):
            invocation_context.active_streaming_tools[function_name].task = None
            invocation_context.active_streaming_tools[function_name].stream = (
                None
            )

        function_response = {
            'status': f'Successfully stopped streaming function {function_name}'
        }
    else:
      function_response = {
          'status': f'No active streaming function named {function_name} found'
      }
  elif hasattr(tool, 'func') and inspect.isasyncgenfunction(tool.func):
    # for streaming tool use case
    # we require the function to be an async generator function
    async def run_tool_and_update_queue(tool, function_args, tool_context):
      try:
        async with Aclosing(
            __call_tool_live(
                tool=tool,
                args=function_args,
                tool_context=tool_context,
                invocation_context=invocation_context,
            )
        ) as agen:
          async for result in agen:
            updated_content = types.Content(
                role='user',
                parts=[
                    types.Part.from_text(
                        text=f'Function {tool.name} returned: {result}'
                    )
                ],
            )
            invocation_context.live_request_queue.send_content(updated_content)
      except asyncio.CancelledError:
        raise  # Re-raise to properly propagate the cancellation

    task = asyncio.create_task(
        run_tool_and_update_queue(tool, function_args, tool_context)
    )

    async with streaming_lock:

      if invocation_context.active_streaming_tools is None:
        invocation_context.active_streaming_tools = {}
      if tool.name in invocation_context.active_streaming_tools:
        invocation_context.active_streaming_tools[tool.name].task = task
      else:
        # Register the streaming tool lazily when the model calls it.
        invocation_context.active_streaming_tools[tool.name] = (
            ActiveStreamingTool(task=task)
        )
        logger.debug('Lazily registered streaming tool: %s', tool.name)

      # For input-streaming tools (those with `input_stream:
      # LiveRequestQueue`), create a dedicated LiveRequestQueue so
      # _send_to_model starts duplicating data to it. This also
      # handles re-invocation after stop_streaming reset .stream
      # to None.
      sig = inspect.signature(tool.func)
      if (
          'input_stream' in sig.parameters
          and _is_live_request_queue_annotation(sig.parameters['input_stream'])
      ):
        invocation_context.active_streaming_tools[tool.name].stream = (
            LiveRequestQueue()
        )

    # Immediately return a pending response.
    # This is required by current live model.
    function_response = {
        'status': (
            'The function is running asynchronously and the results are'
            ' pending.'
        )
    }
  else:
    # Check if we should run tools in thread pool to avoid blocking event loop
    thread_pool_config = invocation_context.run_config.tool_thread_pool_config
    if thread_pool_config is not None:
      function_response = await _call_tool_in_thread_pool(
          tool,
          args=function_args,
          tool_context=tool_context,
          max_workers=thread_pool_config.max_workers,
      )
    else:
      function_response = await __call_tool_async(
          tool, args=function_args, tool_context=tool_context
      )
  return function_response


def _get_tool(
    function_call: types.FunctionCall, tools_dict: dict[str, BaseTool]
):
  """Returns the tool corresponding to the function call."""
  if function_call.name not in tools_dict:
    available = list(tools_dict.keys())
    error_msg = (
        f"Tool '{function_call.name}' not found.\nAvailable tools:"
        f" {', '.join(available)}\n\nPossible causes:\n  1. LLM hallucinated"
        ' the function name - review agent instruction clarity\n  2. Tool not'
        ' registered - verify agent.tools list\n  3. Name mismatch - check for'
        ' typos\n\nSuggested fixes:\n  - Review agent instruction to ensure'
        ' tool usage is clear\n  - Verify tool is included in agent.tools'
        ' list\n  - Check for typos in function name'
    )
    raise ValueError(error_msg)

  return tools_dict[function_call.name]


def _create_tool_context(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tool_confirmation: Optional[ToolConfirmation] = None,
):
  """Creates a ToolContext object."""
  return ToolContext(
      invocation_context=invocation_context,
      function_call_id=function_call.id,
      tool_confirmation=tool_confirmation,
  )


def _get_tool_and_context(
    invocation_context: InvocationContext,
    function_call: types.FunctionCall,
    tools_dict: dict[str, BaseTool],
    tool_confirmation: Optional[ToolConfirmation] = None,
):
  """Returns the tool and tool context corresponding to the function call."""
  tool = _get_tool(function_call, tools_dict)
  tool_context = _create_tool_context(
      invocation_context,
      function_call,
      tool_confirmation,
  )

  return (tool, tool_context)


def _try_decode_computer_use_image(
    tool: BaseTool,
    function_result: dict[str, object],
) -> Optional[list[types.FunctionResponsePart]]:
  """Decodes the image from the function result for a computer use tool.

  Args:
    tool: The tool that produced the function result.
    function_result: The dictionary containing the function's result. This
      dictionary may be modified in-place to remove the 'image' key if an image
      is successfully decoded.

  Returns:
    A list containing a `types.FunctionResponsePart` with the decoded image
    data, or None if no image was found or decoding failed.
  """

  if not isinstance(tool, ComputerUseTool) or not isinstance(
      function_result, dict
  ):
    return None

  if (
      'image' not in function_result
      or 'data' not in function_result['image']
      or 'mimetype' not in function_result['image']
  ):
    return None

  try:
    image_data = base64.b64decode(function_result['image']['data'])
    mime_type = function_result['image']['mimetype']

    part = types.FunctionResponsePart.from_bytes(
        data=image_data, mime_type=mime_type
    )

    del function_result['image']
    return [part]
  except (binascii.Error, ValueError):
    logger.exception('Failed to decode image from computer use tool')
    return None


async def __call_tool_live(
    tool: BaseTool,
    args: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> AsyncGenerator[Event, None]:
  """Calls the tool asynchronously (awaiting the coroutine)."""
  async with Aclosing(
      tool._call_live(
          args=args,
          tool_context=tool_context,
          invocation_context=invocation_context,
      )
  ) as agen:
    async for item in agen:
      yield item


async def __call_tool_async(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> Any:
  """Calls the tool."""
  return await tool.run_async(args=args, tool_context=tool_context)


def __build_response_event(
    tool: BaseTool,
    function_result: dict[str, object],
    tool_context: ToolContext,
    invocation_context: InvocationContext,
) -> Event:
  # Specs requires the result to be a dict.
  if not isinstance(function_result, dict):
    function_result = {'result': function_result}

  if tool.name != 'format_codebase_output':
    function_result = _format_tool_result_for_model(tool.name, function_result)
    function_result = _offload_large_tool_output(tool.name, function_result)
  function_result = _append_context_usage_to_result(
      function_result,
      tool_context=tool_context,
      invocation_context=invocation_context,
  )

  function_response_parts = None
  if isinstance(tool, ComputerUseTool):
    function_response_parts = _try_decode_computer_use_image(
        tool, function_result
    )

  part_function_response = types.Part.from_function_response(
      name=tool.name,
      response=function_result,
      parts=function_response_parts,
  )
  part_function_response.function_response.id = tool_context.function_call_id

  content = types.Content(
      role='user',
      parts=[part_function_response],
  )

  function_response_event = Event(
      invocation_id=invocation_context.invocation_id,
      author=invocation_context.agent.name,
      content=content,
      actions=tool_context.actions,
      branch=invocation_context.branch,
  )

  return function_response_event


def deep_merge_dicts(d1: dict, d2: dict) -> dict:
  """Recursively merges d2 into d1."""
  for key, value in d2.items():
    if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
      d1[key] = deep_merge_dicts(d1[key], value)
    else:
      d1[key] = value
  return d1


def merge_parallel_function_response_events(
    function_response_events: list['Event'],
) -> 'Event':
  if not function_response_events:
    raise ValueError('No function response events provided.')

  if len(function_response_events) == 1:
    return function_response_events[0]
  merged_parts = []
  for event in function_response_events:
    if event.content:
      for part in event.content.parts or []:
        merged_parts.append(part)

  # Use the first event as the "base" for common attributes
  base_event = function_response_events[0]

  # Merge actions from all events
  merged_actions_data: dict[str, Any] = {}
  for event in function_response_events:
    if event.actions:
      # Use `by_alias=True` because it converts the model to a dictionary while respecting field aliases, ensuring that the enum fields are correctly handled without creating a duplicate.
      merged_actions_data = deep_merge_dicts(
          merged_actions_data,
          event.actions.model_dump(exclude_none=True, by_alias=True),
      )

  merged_actions = EventActions.model_validate(merged_actions_data)

  # Create the new merged event
  merged_event = Event(
      invocation_id=base_event.invocation_id,
      author=base_event.author,
      branch=base_event.branch,
      content=types.Content(role='user', parts=merged_parts),
      actions=merged_actions,  # Optionally merge actions if required
  )

  # Use the base_event as the timestamp
  merged_event.timestamp = base_event.timestamp
  return merged_event


def find_event_by_function_call_id(
    events: list[Event],
    function_call_id: str,
) -> Optional[Event]:
  """Finds the function call event that matches the function call id."""
  for event in reversed(events):
    for function_call in event.get_function_calls():
      if function_call.id == function_call_id:
        return event
  return None


def find_matching_function_call(
    events: list[Event],
) -> Optional[Event]:
  """Finds the function call event that matches the function response id of the last event."""
  if not events:
    return None

  last_event = events[-1]
  function_responses = last_event.get_function_responses()
  if not function_responses:
    return None

  return find_event_by_function_call_id(events[:-1], function_responses[0].id)
