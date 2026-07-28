#!/bin/bash
# Build + deploy ehcd-chatbot to the EHCD dev AKS cluster.
# Run from the repo root after `git pull` on `development`. No docker and no ACR
# password: ACR builds server-side, `az aks command invoke` rolls out inside the
# private cluster.
#
# LOGIN: unattended if a service-principal creds file exists (see CREDS below),
# otherwise it falls back to your interactive `az login` session.
set -euo pipefail

SUBSCRIPTION=b4b57807-43c9-4904-9c06-5fe1267e5be2
ACR=acrehcddev
IMAGE=ehcd-chatbot:dev-v1.0
BASE_REPO=ehcd-chatbot-base
RG=rg-ehcd-dev
CLUSTER=aks-ehcd-dev
NS=dev
DEPLOY=ehcd-chatbot

# Service-principal creds, kept OUTSIDE the repo, chmod 600. Override with
# EHCD_DEPLOY_CREDS=/path/to/file. Must define AZURE_CLIENT_ID, AZURE_TENANT_ID
# and one of AZURE_CLIENT_CERT (path to a .pem) or AZURE_CLIENT_SECRET.
CREDS="${EHCD_DEPLOY_CREDS:-$HOME/.config/ehcd/deploy-sp.env}"

cd "$(dirname "$0")"

if [ -f "$CREDS" ]; then
  set -a; . "$CREDS"; set +a
  : "${AZURE_CLIENT_ID:?missing in $CREDS}" "${AZURE_TENANT_ID:?missing in $CREDS}"
  CRED="${AZURE_CLIENT_CERT:-${AZURE_CLIENT_SECRET:?set AZURE_CLIENT_CERT or AZURE_CLIENT_SECRET in $CREDS}}"
  # Own config dir so this login never disturbs your interactive `az` session.
  export AZURE_CONFIG_DIR="$HOME/.azure-ehcd-deploy"
  # Reuse the cached SP token; only re-login when it is gone or expired.
  if ! az account get-access-token -o none 2>/dev/null; then
    # A cert avoids putting the credential in the process list, unlike -p <secret>.
    az login --service-principal -u "$AZURE_CLIENT_ID" --tenant "$AZURE_TENANT_ID" \
      -p "$CRED" --only-show-errors -o none
  fi
else
  echo "no creds file at $CREDS -> using interactive session (HOME=$HOME)"
  az account show -o none 2>/dev/null || { echo "no cached session, logging in"; az login -o none; }
fi
az account set --subscription "$SUBSCRIPTION"
echo "identity: $(az account show --query user.name -o tsv) / sub $(az account show --query name -o tsv)"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "development" ] || echo "WARNING: on branch '$BRANCH', dev builds from 'development'"
echo "building $IMAGE from $BRANCH @ $(git rev-parse --short HEAD)"

# Dependencies live in a separate base image tagged with a hash of requirements.txt.
# Unchanged requirements -> the tag already exists -> skip the slow pip install (faiss,
# langchain, tiktoken) and the app build is just a COPY on top.
BASE_TAG="reqs-$(sha256sum requirements.txt | cut -c1-12)"
BASE_IMAGE="$ACR.azurecr.io/$BASE_REPO:$BASE_TAG"

if az acr repository show-tags -n "$ACR" --repository "$BASE_REPO" -o tsv 2>/dev/null | grep -qx "$BASE_TAG"; then
  echo "deps base image current: $BASE_TAG"
else
  echo "requirements.txt changed -> building deps base $BASE_TAG"
  az acr build -r "$ACR" -t "$BASE_REPO:$BASE_TAG" -f Dockerfile.base --platform linux/amd64 .
fi

# The image bakes no env file — all config comes from the k8s Secret
# ehcd-chatbot-secrets, so the same image is environment-agnostic.
az acr build -r "$ACR" -t "$IMAGE" \
  --build-arg BASE="$BASE_IMAGE" --platform linux/amd64 .

# Optional local equivalent of CI's "Render env file" step: if you keep the dev
# secret values in a local file (chmod 600, OUTSIDE the repo), this re-renders
# env-dev.tmpl and re-applies the Secret. Without the file the existing Secret is
# left untouched — CI is the normal way to change it.
SECRETS="${EHCD_CHATBOT_SECRETS:-$HOME/.config/ehcd/chatbot-dev.secrets.env}"
if [ -f "$SECRETS" ]; then
  echo "rendering k8s Secret from env-dev.tmpl + $SECRETS"
  RENDERED=$(mktemp); trap 'rm -f "$RENDERED"' EXIT
  # NOT `source`: an Azure connection string contains `;`, which bash reads as a
  # command separator and silently truncates the value at. Parse literally instead.
  ( while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in ''|'#'*) continue ;; esac
      key=${line%%=*}
      case "$key" in *[!A-Za-z0-9_]*|'') continue ;; esac
      printf -v "$key" '%s' "${line#*=}"; export "$key"
    done < "$SECRETS"
    envsubst '${PG_PASS} ${AZURE_OPENAI_API_KEY} ${AZURE_BLOB_CONN_STR} ${JWT_SECRET} ${ELEVENLABS_API_KEY} ${ELEVENLABS_AGENT_ID} ${ELEVENLABS_CUSTOMLLM_SHARED_SECRET}' \
      < env-dev.tmpl | grep -vE '^[[:space:]]*(#|$)' > "$RENDERED" )
  az aks command invoke -g "$RG" -n "$CLUSTER" -o tsv --file "$RENDERED" \
    --command "kubectl -n $NS create secret generic ehcd-chatbot-secrets --from-env-file=$(basename "$RENDERED") --dry-run=client -o yaml | kubectl apply -f -"
else
  echo "no secrets file at $SECRETS -> leaving Secret ehcd-chatbot-secrets as-is"
fi

# Tag is fixed and mutable, so the rollout is what picks up the new image
# (imagePullPolicy: Always in the Deployment).
az aks command invoke -g "$RG" -n "$CLUSTER" -o tsv \
  --command "kubectl -n $NS rollout restart deploy/$DEPLOY && kubectl -n $NS rollout status deploy/$DEPLOY --timeout=300s"

echo "done: https://dev-api.20.233.49.25.sslip.io/chatbot/health"
