
ARG BASE=acrehcddev.azurecr.io/ehcd-chatbot-base:latest
FROM ${BASE}

# WORKDIR is inherited from BASE.
WORKDIR /app

COPY . .

ARG APP_ENV=dev
#RUN cp env-${APP_ENV} .env

EXPOSE 8080

# Single uvicorn worker on purpose: FAISS lives in process memory and the index
# rebuild runs on background threads.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
