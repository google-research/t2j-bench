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


def report_truncated_model_output(
    error_message: str,
    max_output_tokens: int | None = None,
) -> dict:
    """Return a structured error when a model-generated tool call was truncated.

    Use this synthetic tool result to tell the model that its previous tool call
    exceeded the current output budget and needs to be retried with a shorter
    response or split into smaller steps.

    Example input:
        error_message="Tool call arguments were truncated while streaming and could not be parsed as valid JSON."
        max_output_tokens=32768

    Example output:
        {
            "status": "error",
            "error_message": "Tool call arguments were truncated while streaming and could not be parsed as valid JSON.",
            "max_output_tokens": 32768,
            "retry_guidance": "Retry with a shorter tool call. Split large file writes into smaller edits or multiple steps."
        }
    """
    guidance = (
        "Retry with a shorter tool call. Split large file writes into smaller edits "
        "or multiple steps."
    )
    return {
        "status": "error",
        "error_message": error_message,
        "max_output_tokens": max_output_tokens,
        "retry_guidance": guidance,
        "output": (
            f"{error_message}\n\n"
            f"Retry guidance: {guidance}"
            + (
                f"\nCurrent max_output_tokens: {max_output_tokens}"
                if isinstance(max_output_tokens, int)
                else ""
            )
        ),
    }
