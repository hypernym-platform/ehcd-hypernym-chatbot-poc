
ARG BASE=acrehcddev.azurecr.io/ehcd-chatbot-base:latest
FROM ${BASE}

# WORKDIR is inherited from BASE.
WORKDIR /app

COPY . .

# No env file is baked in — config comes from the k8s Secret ehcd-chatbot-secrets,
# which CI renders from env-dev.tmpl + the repo's GitHub Environment. Keeps the
# image environment-agnostic and keeps credentials out of registry layers.
# (`load_dotenv()` in app.py is override=False, so a missing .env is a no-op.)

EXPOSE 8080

# Single uvicorn worker on purpose: FAISS lives in process memory and the index
# rebuild runs on background threads.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
