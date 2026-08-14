# syntax=docker/dockerfile:1
# (needed for the --mount=type=secret build-secret syntax used below)

# python:3.9 (used by the sibling NLLB container) is too old here: transformers' GitHub
# main branch (required below) depends on safetensors>=0.8.0, which needs Python >=3.10.
FROM python:3.11

# Optional: HF_TOKEN, if you have one (docker build --secret id=hf_token,env=HF_TOKEN ...;
# buildDocker.ps1 and docker-compose.yml already do this for you). openai/privacy-filter
# is public, so this is NEVER required -- without it, Hugging Face Hub requests are just
# unauthenticated (lower rate limits, and transformers logs a "You are sending
# unauthenticated requests" warning at build time here and again at container startup in
# server.py).
#
# Deliberately a BuildKit secret mount, not ARG/ENV: those get recorded in the image's
# build history/config even when only used transiently -- an ARG's value can still show
# up in `docker history` for the layer that used it, and ENV would additionally bake it
# permanently into the shipped image's config. A secret mount exists only for the single
# RUN instruction that references it below (via /run/secrets/hf_token, in a tmpfs), and
# is never written to any layer or the final image -- the officially recommended way to
# handle build-time secrets. Requires BuildKit (the default builder in modern Docker
# Desktop). If no secret is supplied, the mount is simply absent (not an error) -- see the
# `2>/dev/null || true` fallback below.

# git is needed because requirements.txt installs transformers from GitHub source
# (see the note there for why -- no released PyPI version supports this model yet).
# tesseract-ocr is needed for scanned/flattened PDF pages that have no real text layer
# (see ENABLE_OCR in settings.py).
RUN apt-get update && apt-get install -y --no-install-recommends git tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch from its CPU-only wheel index -- matches this project's DEVICE=-1
# default in settings.py, and skips ~1-2GB of CUDA runtime libraries that would otherwise
# get bundled in even though nothing on the default CPU path uses them.
# To switch to a GPU build instead: change DEVICE to a GPU index in settings.py, drop the
# --index-url below (a plain `pip install torch torchvision torchaudio` pulls a
# CUDA-enabled build from PyPI), and run the container with `--gpus "device=0"` (or
# whichever GPU index you want).
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies -- separate layer so it's cached independently of app
# code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy settings first so import_model.py can read MODEL_NAME
COPY settings.py .
COPY import_model.py .

# Bakes the model + tokenizer into the image at build time (via the HF cache), so the
# container needs no network access at runtime. openai/privacy-filter is a public model
# (Apache 2.0, no gating notice on its model card) -- HF_TOKEN above is optional/for rate
# limits only, never required. If this step ever fails with an authentication/403 error,
# that would mean the model turned out to be gated after all -- stop and ask before relying
# on a token to make that error go away.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" python import_model.py

COPY . .

# Example (cpu, matches default DEVICE=-1 in settings.py):
#   docker run -p 8142:8142 removepii
# Example (gpu, after setting DEVICE=0 in settings.py):
#   docker run --gpus "device=0" -p 8142:8142 removepii
CMD ["python", "server.py"]
