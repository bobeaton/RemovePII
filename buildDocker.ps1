$env:DOCKER_BUILDKIT = "1"  # ensure BuildKit is used (needed for --secret below); default in modern Docker Desktop anyway
docker build --secret id=hf_token,env=HF_TOKEN -t removepii .
docker run -e HF_TOKEN=$env:HF_TOKEN -p 8142:8142 removepii
# HF_TOKEN is optional (openai/privacy-filter is public) -- this just picks it up from
# your shell environment if you have one set, to avoid Hugging Face Hub rate limits and
# the "unauthenticated requests" warning. Fine to leave unset.
# Passed at build time via a BuildKit secret mount (never written to any image layer --
# see the Dockerfile) and separately at container runtime via -e (server.py re-loads the
# model on every startup).
#
# to view the redactor in a web browser, in another powershell window run:
# Start-Process http://localhost:8142/
#
# Alternatively, build and run via docker-compose (also picks up HF_TOKEN from your
# shell environment automatically, see docker-compose.yml):
# docker-compose up --build
