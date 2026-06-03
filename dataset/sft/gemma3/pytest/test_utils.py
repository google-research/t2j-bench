# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests verifying jax_repo utility functions and classes match torch_orig."""
import pytest
import torch
import numpy as np


class TestActivationFunctions:
    """Verify ACT2FN registry and activation function outputs match."""

    def test_act2fn_keys_match(self, torch_utils, jax_utils):
        assert set(torch_utils.ACT2FN.keys()) == set(jax_utils.ACT2FN.keys())

    @pytest.mark.parametrize("act_name", ["gelu", "silu", "swish", "relu", "gelu_pytorch_tanh", "gelu_new", "quick_gelu"])
    def test_activation_outputs_match(self, torch_utils, jax_utils, act_name):
        x = torch.randn(4, 16)
        torch_out = torch_utils.ACT2FN[act_name](x)
        jax_out = jax_utils.ACT2FN[act_name](x)
        assert torch.allclose(torch_out, jax_out, atol=1e-6)

    @pytest.mark.parametrize("act_name", ["gelu", "silu", "relu", "gelu_pytorch_tanh", "quick_gelu"])
    def test_activation_on_zeros(self, torch_utils, jax_utils, act_name):
        x = torch.zeros(4, 8)
        torch_out = torch_utils.ACT2FN[act_name](x)
        jax_out = jax_utils.ACT2FN[act_name](x)
        assert torch.allclose(torch_out, jax_out, atol=1e-7)

    def test_relu_negative_values(self, torch_utils, jax_utils):
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        torch_out = torch_utils.ACT2FN["relu"](x)
        jax_out = jax_utils.ACT2FN["relu"](x)
        assert torch.equal(torch_out, jax_out)
        assert torch_out[0] == 0.0
        assert torch_out[4] == 1.0


class TestModelOutput:
    """Verify ModelOutput base class behaves identically."""

    def test_getitem_by_string(self, torch_utils, jax_utils):
        torch_out = torch_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0]),
            past_key_values=None,
        )
        jax_out = jax_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0]),
            past_key_values=None,
        )

        assert torch.equal(torch_out["last_hidden_state"], jax_out["last_hidden_state"])
        assert torch_out["past_key_values"] == jax_out["past_key_values"]

    def test_getitem_by_index(self, torch_utils, jax_utils):
        torch_out = torch_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0, 2.0]),
        )
        jax_out = jax_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0, 2.0]),
        )

        # Index 0 = last_hidden_state
        assert torch.equal(torch_out[0], jax_out[0])

    def test_keys_and_values(self, torch_utils, jax_utils):
        torch_out = torch_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0]),
        )
        jax_out = jax_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0]),
        )

        assert set(torch_out.keys()) == set(jax_out.keys())

    def test_contains(self, torch_utils, jax_utils):
        torch_out = torch_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0]),
        )
        jax_out = jax_utils.BaseModelOutputWithPast(
            last_hidden_state=torch.tensor([1.0]),
        )

        assert ("last_hidden_state" in torch_out) == ("last_hidden_state" in jax_out) == True
        assert ("nonexistent" in torch_out) == ("nonexistent" in jax_out) == False

    def test_get_with_default(self, torch_utils, jax_utils):
        torch_out = torch_utils.BaseModelOutputWithPast()
        jax_out = jax_utils.BaseModelOutputWithPast()

        assert torch_out.get("last_hidden_state") == jax_out.get("last_hidden_state") == None
        assert torch_out.get("missing", 42) == jax_out.get("missing", 42) == 42


class TestBaseModelOutputWithPooling:
    def test_fields_match(self, torch_utils, jax_utils):
        t = torch_utils.BaseModelOutputWithPooling(
            last_hidden_state=torch.randn(2, 3),
            pooler_output=torch.randn(2),
        )
        j = jax_utils.BaseModelOutputWithPooling(
            last_hidden_state=t.last_hidden_state,
            pooler_output=t.pooler_output,
        )

        assert torch.equal(t.last_hidden_state, j.last_hidden_state)
        assert torch.equal(t.pooler_output, j.pooler_output)


class TestCausalLMOutputWithPast:
    def test_fields_match(self, torch_utils, jax_utils):
        loss = torch.tensor(0.5)
        logits = torch.randn(1, 10, 128)

        t = torch_utils.CausalLMOutputWithPast(loss=loss, logits=logits)
        j = jax_utils.CausalLMOutputWithPast(loss=loss, logits=logits)

        assert torch.equal(t.loss, j.loss)
        assert torch.equal(t.logits, j.logits)


class TestBatchFeature:
    def test_basic_creation(self, torch_utils, jax_utils):
        data = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        t_bf = torch_utils.BatchFeature(data)
        j_bf = jax_utils.BatchFeature(data)

        assert t_bf["input_ids"] == j_bf["input_ids"]
        assert t_bf["attention_mask"] == j_bf["attention_mask"]

    def test_convert_to_tensors(self, torch_utils, jax_utils):
        data = {"values": np.array([[1.0, 2.0], [3.0, 4.0]])}
        t_bf = torch_utils.BatchFeature(data, tensor_type="pt")
        j_bf = jax_utils.BatchFeature(data, tensor_type="pt")

        # Both should have converted numpy arrays to tensors
        assert isinstance(t_bf["values"], torch.Tensor)
        assert isinstance(j_bf["values"], torch.Tensor)
        assert torch.equal(t_bf["values"], j_bf["values"])

    def test_to_device(self, torch_utils, jax_utils):
        data = {"x": torch.randn(2, 3), "y": "not a tensor"}
        t_bf = torch_utils.BatchFeature(data)
        j_bf = jax_utils.BatchFeature(data)

        t_moved = t_bf.to(device=torch.device("cpu"))
        j_moved = j_bf.to(device=torch.device("cpu"))

        assert isinstance(t_moved["x"], torch.Tensor)
        assert isinstance(j_moved["x"], torch.Tensor)
        assert t_moved["y"] == j_moved["y"] == "not a tensor"


class TestConstants:
    """Verify constants match between repos."""

    def test_ignore_index(self, torch_constants, jax_constants):
        assert torch_constants.IGNORE_INDEX == jax_constants.IGNORE_INDEX == -100

    def test_data_config(self, torch_constants, jax_constants):
        assert torch_constants.DATA_CONFIG == jax_constants.DATA_CONFIG == "dataset_info.json"

    def test_placeholders(self, torch_constants, jax_constants):
        assert torch_constants.IMAGE_PLACEHOLDER == jax_constants.IMAGE_PLACEHOLDER == "<image>"
        assert torch_constants.VIDEO_PLACEHOLDER == jax_constants.VIDEO_PLACEHOLDER == "<video>"
        assert torch_constants.AUDIO_PLACEHOLDER == jax_constants.AUDIO_PLACEHOLDER == "<audio>"

    def test_fileext2type(self, torch_constants, jax_constants):
        assert torch_constants.FILEEXT2TYPE == jax_constants.FILEEXT2TYPE


class TestToolUtils:
    """Verify tool utilities match."""

    def test_function_call_namedtuple(self, torch_tool_utils, jax_tool_utils):
        t_fc = torch_tool_utils.FunctionCall(name="search", arguments='{"q": "cats"}')
        j_fc = jax_tool_utils.FunctionCall(name="search", arguments='{"q": "cats"}')
        assert t_fc.name == j_fc.name
        assert t_fc.arguments == j_fc.arguments

    def test_default_function_formatter(self, torch_tool_utils, jax_tool_utils):
        fns = [
            torch_tool_utils.FunctionCall("search", '{"q": "cats"}'),
            torch_tool_utils.FunctionCall("count", '{"n": 5}'),
        ]
        t_utils = torch_tool_utils.get_tool_utils()
        j_fns = [
            jax_tool_utils.FunctionCall("search", '{"q": "cats"}'),
            jax_tool_utils.FunctionCall("count", '{"n": 5}'),
        ]
        j_utils = jax_tool_utils.get_tool_utils()

        assert t_utils.function_formatter(fns) == j_utils.function_formatter(j_fns)

    def test_default_tool_formatter(self, torch_tool_utils, jax_tool_utils):
        tools = [{"name": "search", "description": "Search the web"}]
        t_utils = torch_tool_utils.get_tool_utils()
        j_utils = jax_tool_utils.get_tool_utils()

        assert t_utils.tool_formatter(tools) == j_utils.tool_formatter(tools)

    def test_tool_extractor(self, torch_tool_utils, jax_tool_utils):
        content = "Action: search\nAction Input: {\"q\": \"cats\"}\nAction: count\nAction Input: {\"n\": 5}"
        t_utils = torch_tool_utils.get_tool_utils()
        j_utils = jax_tool_utils.get_tool_utils()

        t_fns = t_utils.tool_extractor(content)
        j_fns = j_utils.tool_extractor(content)

        assert len(t_fns) == len(j_fns) == 2
        for t, j in zip(t_fns, j_fns):
            assert t.name == j.name
            assert t.arguments == j.arguments


class TestMMPlugin:
    """Verify multimodal plugin matches."""

    def test_plugin_registry(self, torch_mm_plugin, jax_mm_plugin):
        assert set(torch_mm_plugin.PLUGINS.keys()) == set(jax_mm_plugin.PLUGINS.keys())

    def test_base_plugin_passthrough(self, torch_mm_plugin, jax_mm_plugin):
        t_plugin = torch_mm_plugin.get_mm_plugin("base")
        j_plugin = jax_mm_plugin.get_mm_plugin("base")

        messages = [{"role": "user", "content": "Hi"}]
        assert t_plugin.process_messages(messages) == j_plugin.process_messages(messages)
        assert t_plugin.get_mm_inputs() == j_plugin.get_mm_inputs() == {}

    def test_gemma3_plugin_process_messages_no_images(self, torch_mm_plugin, jax_mm_plugin):
        t_plugin = torch_mm_plugin.get_mm_plugin("gemma3", image_token="<img>")
        j_plugin = jax_mm_plugin.get_mm_plugin("gemma3", image_token="<img>")

        messages = [{"role": "user", "content": "Hi"}]
        # Without images/processor, should passthrough
        t_result = t_plugin.process_messages(messages)
        j_result = j_plugin.process_messages(messages)
        assert t_result == j_result == messages

    def test_get_gemma3_token_type_ids_list(self, torch_mm_plugin, jax_mm_plugin):
        input_ids = [1, 2, 99, 3, 99, 4]
        image_token_id = 99

        t_ttids = torch_mm_plugin._get_gemma3_token_type_ids(input_ids, image_token_id)
        j_ttids = jax_mm_plugin._get_gemma3_token_type_ids(input_ids, image_token_id)

        assert t_ttids == j_ttids == [0, 0, 1, 0, 1, 0]

    def test_get_gemma3_token_type_ids_tensor(self, torch_mm_plugin, jax_mm_plugin):
        input_ids = torch.tensor([1, 2, 99, 3, 99, 4])
        image_token_id = 99

        t_ttids = torch_mm_plugin._get_gemma3_token_type_ids(input_ids, image_token_id)
        j_ttids = jax_mm_plugin._get_gemma3_token_type_ids(input_ids, image_token_id)

        assert torch.equal(t_ttids, j_ttids)
        expected = torch.tensor([0, 0, 1, 0, 1, 0])
        assert torch.equal(t_ttids, expected)
