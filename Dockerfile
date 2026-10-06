ARG BASE=acrehcddev.azurecr.io/ehcd-chatbot-base:latest
FROM ${BASE}

# WORKDIR is inherited from BASE.
WORKDIR /app

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data \
    HOME=/app/data \
    TIKTOKEN_CACHE_DIR=/app/data/tiktoken

# Run unprivileged. /app deliberately stays root-owned: the application only reads from
# it, so leaving it unwritable by appuser is the point, not an oversight. Only /app/data
# is writable, and in Kubernetes it is a mounted volume that replaces what is baked here.
RUN groupadd --system --gid 1000 appuser \
 && useradd --system --uid 1000 --gid 1000 --no-create-home appuser \
 && mkdir -p /app/data/tiktoken \
 && chown -R 1000:1000 /app/data
USER 1000:1000

EXPOSE 8080

# Single uvicorn worker on purpose: FAISS lives in process memory and the index
# rebuild runs on background threads.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
