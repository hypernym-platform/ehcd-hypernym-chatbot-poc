
FROM python:3.12-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# The container runs with readOnlyRootFilesystem, so nothing may be written into the
# image at runtime. Everything writable lives under DATA_ROOT, which Kubernetes mounts
# as an emptyDir at /app/data (see k8s_yamls/*/codebase_yamls/ehcd-chatbot.yaml).
#
#   PYTHONDONTWRITEBYTECODE  stops the interpreter dropping __pycache__ beside the sources
#   DATA_ROOT                app.py / doc.py / edu_pg.py / policy.py all derive paths from it
#   HOME                     appuser has no home dir; without this, ~/.cache writes hit /
#   TIKTOKEN_CACHE_DIR       tiktoken otherwise caches BPE files under /tmp
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_ROOT=/app/data \
    HOME=/app/data \
    TIKTOKEN_CACHE_DIR=/app/data/tiktoken

# Set a working directory inside the container
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Run unprivileged. /app deliberately stays root-owned: the application only reads from
# it, so leaving it unwritable by appuser is the point, not an oversight. Only /app/data
# is writable, and in Kubernetes it is a mounted volume that replaces what is baked here.
RUN groupadd --system --gid 1000 appuser \
 && useradd --system --uid 1000 --gid 1000 --no-create-home appuser \
 && mkdir -p /app/data/tiktoken \
 && chown -R 1000:1000 /app/data
USER 1000:1000

# Expose the port the app runs on
EXPOSE 8080

# Command to run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
