FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      curl \
      git \
      gnupg \
      jq \
      ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex @anthropic-ai/claude-code @mariozechner/pi-coding-agent \
    && pi install npm:pi-subagents

COPY pi-extensions /opt/pi-extensions
RUN node /opt/pi-extensions/patch-bedrock-mantle.mjs \
    && node /opt/pi-extensions/patch-pi-subagents-final-output.js --required

WORKDIR /benchmark
CMD ["sleep", "infinity"]
