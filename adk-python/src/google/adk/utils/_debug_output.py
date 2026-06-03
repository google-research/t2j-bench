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

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..events.event import Event

# Constants for debug output truncation
_ARGS_MAX_LEN = 50  # Keep arg previews short for readability
_RESPONSE_MAX_LEN = 100  # Show more of response for context
_CODE_OUTPUT_MAX_LEN = 100  # Code execution output preview length
_COMMAND_WRAP_WIDTH = 100
_MAX_PREVIEW_LINES = 5
_PREFERRED_RESULT_KEYS = [
    'status',
    'error_message',
    'path',
    'truncated',
    'entries_count',
    'results_count',
    'matches_count',
    'returned_results',
    'matched_files',
    'match_strategy',
    'match_count',
    'expected_replacements',
    'correction_used',
    'bytes_written',
    'exit_code',
    'command_status',
    'is_running',
    'session_id',
]
_PREFERRED_ARG_KEYS = [
    'path',
    'target_folder',
    'pattern',
    'keywords',
    'expected_replacements',
    'match_mode',
    'start_line',
    'end_line',
]
_printed_block = False


def _truncate(text: str, max_len: int) -> str:
  """Truncate text to max length, appending '...' if truncated.

  Args:
      text: The text to truncate.
      max_len: Maximum length before truncation.

  Returns:
      The truncated text with '...' appended if it exceeds max_len.
  """
  return text[:max_len] + '...' if len(text) > max_len else text


def _truncate_lines(text: str, max_lines: int = _MAX_PREVIEW_LINES) -> str:
  lines = text.splitlines()
  if len(lines) <= max_lines:
    return text
  omitted = len(lines) - max_lines
  return '\n'.join(lines[:max_lines] + [f'...<omitted {omitted} lines>'])


def _format_command(command: str, *, indent: str = '  ') -> str:
  if not command:
    return ''
  wrapped_lines: list[str] = []
  for line in command.splitlines() or ['']:
    filled = textwrap.fill(
        line,
        width=_COMMAND_WRAP_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    wrapped_lines.append(filled if filled else indent)
  return '\n'.join(wrapped_lines)


def _indent_block(text: str, prefix: str = '  ') -> str:
  return '\n'.join(
      f'{prefix}{line}' if line else prefix.rstrip()
      for line in text.splitlines()
  )


def _format_usage_metadata(event: Event) -> str:
  usage = getattr(event, 'usage_metadata', None)
  if not usage:
    return ''
  parts: list[str] = []
  if getattr(usage, 'prompt_token_count', None) is not None:
    parts.append(f'prompt={usage.prompt_token_count}')
  if getattr(usage, 'cached_content_token_count', None) is not None:
    parts.append(f'cached={usage.cached_content_token_count}')
  if getattr(usage, 'candidates_token_count', None) is not None:
    parts.append(f'output={usage.candidates_token_count}')
  if getattr(usage, 'thoughts_token_count', None) is not None:
    parts.append(f'thoughts={usage.thoughts_token_count}')
  if getattr(usage, 'total_token_count', None) is not None:
    parts.append(f'total={usage.total_token_count}')
  if not parts:
    return ''
  return '[Token usage] ' + ' '.join(parts)


def _serialize_preview(value: object, *, max_len: int) -> str:
  if isinstance(value, str):
    serialized = value
  else:
    try:
      serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
      serialized = str(value)
  serialized = _truncate_lines(serialized)
  return _truncate(serialized, max_len)


def _preview_scalar(value: object) -> str:
  if isinstance(value, str):
    single_line = value.replace('\n', '\\n')
    return _truncate(single_line, _RESPONSE_MAX_LEN)
  if isinstance(value, (int, float, bool)) or value is None:
    return str(value)
  return _serialize_preview(value, max_len=_RESPONSE_MAX_LEN)


def _summarize_dict(
    data: dict[object, object],
    *,
    preferred_keys: list[str],
) -> list[str]:
  lines: list[str] = []
  used_keys: set[str] = set()
  for key in preferred_keys:
    if key in data:
      lines.append(f'{key}: {_preview_scalar(data[key])}')
      used_keys.add(key)

  remaining_items = [
      (str(key), value)
      for key, value in data.items()
      if isinstance(key, str) and key not in used_keys
  ]
  for key, value in sorted(remaining_items):
    lines.append(f'{key}: {_preview_scalar(value)}')
  return lines


def _format_tool_call(name: str | None, args: object) -> str:
  tool_name = name or '<unknown>'
  if isinstance(args, dict):
    command = args.get('command')
    if isinstance(command, str):
      remaining = {k: v for k, v in args.items() if k != 'command'}
      header = f'[Calling tool: {tool_name}]'
      sections = [header, 'command:', _format_command(command)]
      if remaining:
        sections.append('args:')
        sections.append(_serialize_preview(remaining, max_len=_ARGS_MAX_LEN))
      return _truncate_lines('\n'.join(sections))
    summary_lines = _summarize_dict(args, preferred_keys=_PREFERRED_ARG_KEYS)
    if summary_lines:
      return _truncate_lines('\n'.join(
          [f'[Calling tool: {tool_name}]'] + summary_lines
      ))
  return _truncate_lines(
      f'[Calling tool: {tool_name}({_serialize_preview(args, max_len=_ARGS_MAX_LEN)})]'
  )


def _format_tool_result(response: object) -> str:
  if isinstance(response, dict):
    command = response.get('command')
    output_lines = response.get('output_lines')
    stdout_lines = response.get('stdout_lines')
    stderr_lines = response.get('stderr_lines')
    stdout = response.get('stdout')
    stderr = response.get('stderr')
    sections: list[str] = ['[Tool result]']
    if isinstance(command, str):
      sections.extend(['command:', _format_command(command)])
    if isinstance(output_lines, list):
      output_text = '\n'.join(str(line) for line in output_lines)
      sections.extend(['output:', _truncate_lines(output_text)])
    if isinstance(stdout_lines, list):
      stdout_text = '\n'.join(str(line) for line in stdout_lines)
      sections.extend(['stdout:', _truncate_lines(stdout_text)])
    elif isinstance(stdout, str) and stdout:
      sections.extend(['stdout:', _truncate_lines(stdout)])
    if isinstance(stderr_lines, list):
      stderr_text = '\n'.join(str(line) for line in stderr_lines)
      sections.extend(['stderr:', _truncate_lines(stderr_text)])
    if isinstance(stderr, str) and stderr:
      sections.extend(['stderr:', _truncate_lines(stderr)])
    remaining = {
        k: v for k, v in response.items()
        if k not in {'command', 'output_lines', 'stdout_lines', 'stderr_lines', 'stdout', 'stderr'}
    }
    if isinstance(remaining.get('entries'), list):
      entries = remaining['entries']
      remaining['entries_count'] = len(entries)
    if isinstance(remaining.get('results'), list):
      results = remaining['results']
      remaining['results_count'] = len(results)
    if isinstance(remaining.get('matches'), list):
      matches = remaining['matches']
      remaining['matches_count'] = len(matches)
    if remaining:
      sections.extend(_summarize_dict(remaining, preferred_keys=_PREFERRED_RESULT_KEYS))
    return '\n'.join(section for section in sections if section)
  return _truncate_lines(
      f'[Tool result: {_serialize_preview(response, max_len=_RESPONSE_MAX_LEN)}]'
  )


def print_event(event: Event, *, verbose: bool = False) -> None:
  """Print an event to stdout in a user-friendly format.

  Args:
      event: The event to print.
      verbose: If True, shows detailed tool calls and responses. If False,
          shows only text responses for cleaner output.
  """
  if not event.content or not event.content.parts:
    return

  global _printed_block

  # Collect consecutive text parts to avoid repeating author prefix
  text_buffer: list[str] = []
  header_printed = False

  def print_header() -> None:
    nonlocal header_printed
    global _printed_block
    if _printed_block:
      print()
    print(f'[{event.author}]')
    usage_line = _format_usage_metadata(event)
    if usage_line:
      print(f'  {usage_line}')
    header_printed = True
    _printed_block = True

  def flush_text() -> None:
    """Flush accumulated text parts as a single output."""
    global _printed_block
    if text_buffer:
      combined_text = ''.join(text_buffer)
      if not header_printed:
        print_header()
      print(_indent_block(combined_text))
      text_buffer.clear()
      _printed_block = True

  for part in event.content.parts:
    # Text parts are always shown regardless of verbose setting
    # because they contain the actual agent responses users expect
    if part.text:
      text_buffer.append(part.text)
    else:
      # Flush any accumulated text before handling non-text parts
      flush_text()

      # Non-text parts (tool calls, code, etc.) are hidden by default
      # to reduce clutter and show only what matters: the final results
      if verbose:
        # Tool invocations show the behind-the-scenes processing
        if part.function_call:
          formatted = _format_tool_call(
              part.function_call.name, part.function_call.args
          )
          if not header_printed:
            print_header()
          else:
            print()
            print(f'[{event.author}]')
          print(_indent_block(formatted))
          _printed_block = True
        # Handle function response parts (tool results)
        elif part.function_response:
          formatted = _format_tool_result(part.function_response.response)
          if not header_printed:
            print_header()
          else:
            print()
            print(f'[{event.author}]')
          print(_indent_block(formatted))
          _printed_block = True
        # Handle executable code parts
        elif part.executable_code:
          lang = part.executable_code.language or 'code'
          if not header_printed:
            print_header()
          else:
            print()
            print(f'[{event.author}]')
          print(f'  [Executing {lang} code...]')
          _printed_block = True
        # Handle code execution result parts
        elif part.code_execution_result:
          output = part.code_execution_result.output or 'result'
          if not header_printed:
            print_header()
          else:
            print()
            print(f'[{event.author}]')
          print(
              f'  [Code output:'
              f' {_truncate(str(output), _CODE_OUTPUT_MAX_LEN)}]'
          )
          _printed_block = True
        # Handle inline data (images, files)
        elif part.inline_data:
          mime_type = part.inline_data.mime_type or 'data'
          if not header_printed:
            print_header()
          else:
            print()
            print(f'[{event.author}]')
          print(f'  [Inline data: {mime_type}]')
          _printed_block = True
        # Handle file data
        elif part.file_data:
          uri = part.file_data.file_uri or 'file'
          if not header_printed:
            print_header()
          else:
            print()
            print(f'[{event.author}]')
          print(f'  [File: {uri}]')
          _printed_block = True

  # Flush any remaining text at the end
  flush_text()
