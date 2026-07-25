# App layer only — dependencies live in the base image built from Dockerfile.base
# (python:3.12-slim + curl + pip install). deploy-dev.sh passes BASE pinned to a
# hash of requirements.txt, so a code-only change rebuilds just the COPY below.
ARG BASE=acrehcddev.azurecr.io/ehcd-chatbot-base:latest
FROM ${BASE}

# WORKDIR is inherited from BASE.
WORKDIR /app

COPY . .

ARG APP_ENV=dev
RUN cp env-${APP_ENV} .env

EXPOSE 8080

# Single uvicorn worker on purpose: FAISS lives in process memory and the index
# rebuild runs on background threads.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
