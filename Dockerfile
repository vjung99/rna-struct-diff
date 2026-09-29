FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04
ENV WANDB_API_KEY=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
RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev git build-essential && rm -rf /var/lib/apt/lists/* && ln -sf python3.12 /usr/bin/python3
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY . /app
RUN uv venv && uv sync
