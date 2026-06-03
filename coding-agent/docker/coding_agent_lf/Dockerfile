FROM node:22-bookworm-slim AS node_cli

RUN npm install -g @openai/codex @google/gemini-cli @anthropic-ai/claude-code

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOST_HOME=/home/coding_agent
ENV LF_ENV_PATH=/home/coding_agent/miniconda3/envs/lf

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bzip2 \
    ca-certificates \
    curl \
    git \
    procps \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${LF_ENV_PATH}" \
    "${HOST_HOME}/adk-python" \
    "${HOST_HOME}/LlamaFactory" \
    "${HOST_HOME}/.cache" \
    "${HOST_HOME}/.codex" \
    "${HOST_HOME}/.claude" \
    "${HOST_HOME}/.config" \
    /opt/conda/envs \
    /attempt_workspace \
    /attempt_meta \
    /workspace

RUN chmod 0777 "${HOST_HOME}" /attempt_workspace /attempt_meta "${HOST_HOME}/.claude"

COPY --from=node_cli /usr/local/ /usr/local/
COPY .artifacts/lf-env.tar.gz /tmp/lf-env.tar.gz
COPY .artifacts/editable_sources/google_adk ${HOST_HOME}/adk-python/src
COPY .artifacts/editable_sources/llamafactory ${HOST_HOME}/LlamaFactory/src

RUN tar -xzf /tmp/lf-env.tar.gz -C "${LF_ENV_PATH}" \
    && rm /tmp/lf-env.tar.gz \
    && PATH="${LF_ENV_PATH}/bin:$PATH" "${LF_ENV_PATH}/bin/conda-unpack" \
    && ln -s "${LF_ENV_PATH}" /opt/conda/envs/lf

RUN "${LF_ENV_PATH}/bin/python" -m pip install --no-cache-dir "litellm>=1.75.5,<2.0.0"

ENV PATH=${LF_ENV_PATH}/bin:$PATH
ENV CONDA_PREFIX=${LF_ENV_PATH}
ENV CONDA_DEFAULT_ENV=lf
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=${HOST_HOME}/adk-python/src:${HOST_HOME}/LlamaFactory/src${PYTHONPATH:+:$PYTHONPATH}

WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
