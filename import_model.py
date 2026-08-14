# This file exists because:
# 1. Documentation for using transformers often shows how to download models in Python
# 2. We want the download of the model to happen as part of the building of the image, not when it runs
#
# openai/privacy-filter is a public model (Apache 2.0), so HF_TOKEN is NOT required to
# fetch it -- but if one is set (via `docker build --build-arg HF_TOKEN=...`, see
# Dockerfile), pass it along: it authenticates the request, which raises rate limits and
# quiets transformers' "You are sending unauthenticated requests" warning. If this ever
# fails with an authentication/403 error even with a token set, the model turned out to
# be gated after all -- stop and ask rather than assuming a token will fix it.
#
# trust_remote_code=True is required: transformers' built-in model registry does not
# (yet) recognize this model's architecture ("openai_privacy_filter"), so the model repo
# ships its own modeling code, which we allow it to load and run here.

import os

from settings import MODEL_NAME
from transformers import AutoTokenizer, AutoModelForTokenClassification

hf_token = os.environ.get("HF_TOKEN")
model_kwargs = {"token": hf_token} if hf_token else {}

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, trust_remote_code=True, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, **model_kwargs)
