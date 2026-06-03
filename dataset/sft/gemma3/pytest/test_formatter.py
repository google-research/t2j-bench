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

"""Tests verifying jax_repo formatters match torch_orig."""
import pytest


class TestEmptyFormatter:
    def test_apply_returns_slots_unchanged(self, torch_formatter, jax_formatter):
        slots = [{"bos_token"}, "hello"]
        torch_f = torch_formatter.EmptyFormatter(slots=slots)
        jax_f = jax_formatter.EmptyFormatter(slots=slots)
        assert torch_f.apply() == jax_f.apply()

    def test_apply_with_kwargs_ignores_them(self, torch_formatter, jax_formatter):
        slots = ["raw text"]
        torch_f = torch_formatter.EmptyFormatter(slots=slots)
        jax_f = jax_formatter.EmptyFormatter(slots=slots)
        assert torch_f.apply(content="ignored") == jax_f.apply(content="ignored")


class TestStringFormatter:
    def test_content_replacement(self, torch_formatter, jax_formatter):
        slots = ["<start>{{content}}<end>"]
        torch_f = torch_formatter.StringFormatter(slots=slots)
        jax_f = jax_formatter.StringFormatter(slots=slots)
        assert torch_f.apply(content="hello world") == jax_f.apply(content="hello world")

    def test_no_placeholder(self, torch_formatter, jax_formatter):
        slots = ["no placeholder here"]
        torch_f = torch_formatter.StringFormatter(slots=slots)
        jax_f = jax_formatter.StringFormatter(slots=slots)
        assert torch_f.apply(content="unused") == jax_f.apply(content="unused")

    def test_multiple_slots(self, torch_formatter, jax_formatter):
        slots = ["prefix {{content}}", {"bos_token"}, " suffix"]
        torch_f = torch_formatter.StringFormatter(slots=slots)
        jax_f = jax_formatter.StringFormatter(slots=slots)
        result_t = torch_f.apply(content="MSG")
        result_j = jax_f.apply(content="MSG")
        assert result_t == result_j
        assert result_t[0] == "prefix MSG"
        assert result_t[1] == {"bos_token"}

    def test_empty_content(self, torch_formatter, jax_formatter):
        slots = ["begin {{content}} end"]
        torch_f = torch_formatter.StringFormatter(slots=slots)
        jax_f = jax_formatter.StringFormatter(slots=slots)
        assert torch_f.apply(content="") == jax_f.apply(content="")

    def test_gemma3_user_format(self, torch_formatter, jax_formatter):
        slots = ["<start_of_turn>user\n{{content}}<end_of_turn>\n<start_of_turn>model\n"]
        torch_f = torch_formatter.StringFormatter(slots=slots)
        jax_f = jax_formatter.StringFormatter(slots=slots)
        result_t = torch_f.apply(content="What is this?")
        result_j = jax_f.apply(content="What is this?")
        assert result_t == result_j
        assert "<start_of_turn>user\nWhat is this?<end_of_turn>" in result_t[0]

    def test_gemma3_assistant_format(self, torch_formatter, jax_formatter):
        slots = ["{{content}}<end_of_turn>\n"]
        torch_f = torch_formatter.StringFormatter(slots=slots)
        jax_f = jax_formatter.StringFormatter(slots=slots)
        result_t = torch_f.apply(content="This is a cat.")
        result_j = jax_f.apply(content="This is a cat.")
        assert result_t == result_j


class TestFunctionFormatter:
    def test_basic_function_formatting(self, torch_formatter, jax_formatter):
        from collections import namedtuple
        FunctionCall = namedtuple("FunctionCall", ["name", "arguments"])
        fns = [FunctionCall("search", '{"query": "cats"}')]

        torch_f = torch_formatter.FunctionFormatter(slots=[])
        jax_f = jax_formatter.FunctionFormatter(slots=[])
        result_t = torch_f.apply(functions=fns)
        result_j = jax_f.apply(functions=fns)
        assert result_t == result_j

    def test_no_functions_falls_back_to_content(self, torch_formatter, jax_formatter):
        torch_f = torch_formatter.FunctionFormatter(slots=[])
        jax_f = jax_formatter.FunctionFormatter(slots=[])
        result_t = torch_f.apply(content="fallback text")
        result_j = jax_f.apply(content="fallback text")
        assert result_t == result_j == ["fallback text"]


class TestToolFormatter:
    def test_json_tools(self, torch_formatter, jax_formatter):
        tools = [{"name": "search", "description": "Search the web"}]
        import json
        tools_str = json.dumps(tools)

        torch_f = torch_formatter.ToolFormatter(slots=[])
        jax_f = jax_formatter.ToolFormatter(slots=[])
        result_t = torch_f.apply(content=tools_str)
        result_j = jax_f.apply(content=tools_str)
        assert result_t == result_j

    def test_plain_string_passthrough(self, torch_formatter, jax_formatter):
        torch_f = torch_formatter.ToolFormatter(slots=[])
        jax_f = jax_formatter.ToolFormatter(slots=[])
        result_t = torch_f.apply(content="not json")
        result_j = jax_f.apply(content="not json")
        assert result_t == result_j == ["not json"]
