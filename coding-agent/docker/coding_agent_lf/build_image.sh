#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/.artifacts"
ENV_TARBALL="${ARTIFACT_DIR}/lf-env.tar.gz"
EDITABLE_SRC_DIR="${ARTIFACT_DIR}/editable_sources"
IMAGE_TAG="${1:-coding-agent-lf:latest}"
USER_HOME="${HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# adk-python is vendored in the repo (custom fork). Default to the in-repo copy.
ADK_SRC="${CODING_AGENT_ADK_SRC:-${REPO_ROOT}/adk-python/src}"
LLAMAFACTORY_SRC="${CODING_AGENT_LLAMAFACTORY_SRC:-${CODING_AGENT_LLAMAFCTORY_SRC:-${USER_HOME}/LlamaFactory/src}}"
CONDA_PACK_BIN="${CODING_AGENT_CONDA_PACK_BIN:-}"

if [[ -z "${CONDA_PACK_BIN}" ]]; then
  if command -v conda-pack >/dev/null 2>&1; then
    CONDA_PACK_BIN="$(command -v conda-pack)"
  else
    CONDA_PACK_BIN="${USER_HOME}/miniconda3/bin/conda-pack"
  fi
fi

resolve_docker_cmd() {
  if docker ps >/dev/null 2>&1; then
    echo "docker"
    return 0
  fi
  if sudo -n docker ps >/dev/null 2>&1; then
    echo "sudo -n docker"
    return 0
  fi
  echo "Unable to access Docker daemon with either 'docker' or 'sudo -n docker'." >&2
  return 1
}

DOCKER_CMD="$(resolve_docker_cmd)"

mkdir -p "${ARTIFACT_DIR}"
mkdir -p "${EDITABLE_SRC_DIR}"

if [[ ! -d "${ADK_SRC}" ]]; then
  echo "google_adk source not found: ${ADK_SRC}" >&2
  exit 1
fi

if [[ ! -d "${LLAMAFACTORY_SRC}" ]]; then
  echo "LlamaFactory source not found: ${LLAMAFACTORY_SRC}" >&2
  exit 1
fi

if [[ ! -x "${CONDA_PACK_BIN}" ]]; then
  echo "conda-pack binary not found or not executable: ${CONDA_PACK_BIN}" >&2
  exit 1
fi

echo "[1/4] Syncing editable package sources"
rm -rf "${EDITABLE_SRC_DIR}/google_adk" "${EDITABLE_SRC_DIR}/llamafactory"
cp -R "${ADK_SRC}" "${EDITABLE_SRC_DIR}/google_adk"
cp -R "${LLAMAFACTORY_SRC}" "${EDITABLE_SRC_DIR}/llamafactory"

echo "[2/4] Packing conda env 'lf' into ${ENV_TARBALL}"
rm -f "${ENV_TARBALL}"
"${CONDA_PACK_BIN}" \
  -n lf \
  -o "${ENV_TARBALL}" \
  --ignore-editable-packages \
  --ignore-missing-files

echo "[3/4] Building docker image ${IMAGE_TAG}"
${DOCKER_CMD} build -t "${IMAGE_TAG}" "${SCRIPT_DIR}"

echo "[4/4] Done"
echo "Image: ${IMAGE_TAG}"
echo "Env tarball: ${ENV_TARBALL}"
