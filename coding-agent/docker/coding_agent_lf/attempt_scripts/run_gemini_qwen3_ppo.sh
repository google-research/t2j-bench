#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_gemini_instruction_attempt.sh" ppo qwen3_vl qwen3_ppo "$@"
