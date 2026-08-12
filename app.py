import os
import json
import re
import time
import asyncio
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

import psycopg2
from psycopg2 import InterfaceError, OperationalError
from psycopg2.pool import ThreadedConnectionPool

import redis
import jwt
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    StreamingResponse,
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from openai import AzureOpenAI

# FAISS + embeddings (still needed for policy searches)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

from doc import Documents
from education import TabularConfig
from edu_pg import load_excel_to_sqlite
from policy import update_policy_index_if_changed, PolicyConfig
from emb_pace import PacedEmbeddings

# Modular imports
from rbac import fetch_user_profile
from tools import build_available_tools, run_chatbot_graph
from chart_engine import detect_chart_opportunity

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG & LOGGING
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="EHCD Hypernym Chatbot")
app.add_middleware(SessionMiddleware, secret_key="fs78sf7s8d6v7sdy7sdbds7v")

# Static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Redis (for chat history)
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
)


@dataclass(frozen=True)
class CFG:
    # Postgres
    PG_HOST: str = os.getenv("PG_HOST", "127.0.0.1")
    PG_DB: str = os.getenv("PG_DB", "postgres")
    PG_USER: str = os.getenv("PG_USER", "postgres")
    PG_PASS: str = os.getenv("PG_PASS", "wgRV|X&77:8#")
    PG_PORT: int = int(os.getenv("PG_PORT", "5432"))

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv(
        "ENDPOINT_URL", "https://app-openai-uae.cognitiveservices.azure.com/"
    )
    AZURE_OPENAI_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
    )
    AZURE_EMBED_DEPLOYMENT: str = os.getenv(
        "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
    )

    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    USER_ID_CLAIM: str = os.getenv("USER_ID_CLAIM", "sub/user_id/id")

    # Local storage
    ROOT: str = os.getenv("DATA_ROOT", "./data")
    DOC_DIR: str = os.path.join(ROOT, "docs")
    HASH_DIR: str = os.path.join(ROOT, "hashes")
    FAISS_DIR: str = os.path.join(ROOT, "faiss")

    # Chunking (for education & policy)
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 120


cfg = CFG()

# ────────────────────────────────────────────────────────────────────────────────
# Education Tabular (Excel) Config
# ────────────────────────────────────────────────────────────────────────────────
EDU_CFG = TabularConfig(
    tabular_dir=os.path.join(cfg.DOC_DIR, "tabular"),
    faiss_dir=os.path.join(cfg.FAISS_DIR, "education_tabular"),
    hash_json=os.path.join(cfg.HASH_DIR, "education_tabular.sha.json"),
    blob_conn_str=os.getenv("AZURE_BLOB_CONN_STR", ""),
    blob_container=os.getenv("AZURE_BLOB_CONTAINER", ""),
    blob_prefix=os.getenv("AZURE_BLOB_PREFIX", ""),
    throttle_seconds=600,
)

POLICY_CFG = PolicyConfig(
    faiss_dir=os.path.join(cfg.FAISS_DIR, "policy"),
    hash_json=os.path.join(cfg.HASH_DIR, "policy.sha.json"),
    blob_conn_str=os.getenv("AZURE_BLOB_CONN_STR", ""),
    blob_container=os.getenv("AZURE_BLOB_CONTAINER", ""),
    blob_prefix="policies/",
    local_dir=os.path.join(cfg.DOC_DIR, "policies"),
    throttle_seconds=600,
)

os.makedirs(cfg.DOC_DIR, exist_ok=True)
os.makedirs(cfg.HASH_DIR, exist_ok=True)
os.makedirs(cfg.FAISS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Azure OpenAI clients
# ────────────────────────────────────────────────────────────────────────────────
client = AzureOpenAI(
    azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
    api_key=cfg.AZURE_OPENAI_KEY,
    api_version=cfg.AZURE_OPENAI_API_VERSION,
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=cfg.AZURE_EMBED_DEPLOYMENT,
    openai_api_key=cfg.AZURE_OPENAI_KEY,
    azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
    openai_api_version=cfg.AZURE_OPENAI_API_VERSION,
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
)
emb = PacedEmbeddings(embeddings, tpm_limit=200_000, batch_size=32)

# Document management (optional UI)
documents = Documents()
documents.save_local_files_to_db()

# ────────────────────────────────────────────────────────────────────────────────
# DATABASE CONNECTION POOL
# ────────────────────────────────────────────────────────────────────────────────
_pool = None


def _init_pg_pool():
    global _pool
    if _pool is not None:
        return
    _pool = ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        host=cfg.PG_HOST,
        dbname=cfg.PG_DB,
        user=cfg.PG_USER,
        password=cfg.PG_PASS,
        port=cfg.PG_PORT,
        sslmode="require",
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        application_name="ehcd-hypernym-chatbot",
    )
    logger.info("[PG] Connection pool initialized.")


# Try to connect at startup, but don't crash if PG is temporarily unreachable
try:
    _init_pg_pool()
except Exception as e:
    logger.warning(f"[PG] Pool init failed at startup (will retry on first request): {e}")


@contextmanager
def pg_conn():
    """Get a connection from the pool with auto-commit/rollback."""
    global _pool
    if _pool is None:
        _init_pg_pool()
    conn = _pool.getconn()
    try:
        if conn.closed:
            _pool.putconn(conn, close=True)
            conn = _pool.getconn()
        yield conn
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except (OperationalError, InterfaceError):
            pass
        if isinstance(e, (OperationalError, InterfaceError)):
            _pool.putconn(conn, close=True)
            conn = None
        raise
    finally:
        if conn is not None:
            _pool.putconn(conn)


# ────────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY (Redis)
# ────────────────────────────────────────────────────────────────────────────────
MAX_HISTORY_MESSAGES = 40

def get_conversation_history(user_key: str) -> list:
    redis_key = f"user_{user_key}_history"

    try:
        h = redis_client.get(redis_key)

        logger.info(
            "[CHAT_HISTORY][GET] key=%s exists=%s",
            redis_key,
            h is not None
        )

        if not h:
            logger.info("[CHAT_HISTORY][GET] No history found.")
            return []

        history = json.loads(h)

        logger.info(
            "[CHAT_HISTORY][GET] Loaded %d messages",
            len(history)
        )

        logger.info(
            "[CHAT_HISTORY][GET] History=%s",
            history
        )

        return history

    except Exception as e:
        logger.error(
            "[CHAT_HISTORY][GET] Failed for key=%s: %s",
            redis_key,
            e,
            exc_info=True
        )
        return []


def save_conversation_history(user_key: str, history: list):
    redis_key = f"user_{user_key}_history"

    try:
        trimmed = (
            history[-MAX_HISTORY_MESSAGES:]
            if len(history) > MAX_HISTORY_MESSAGES
            else history
        )

        redis_client.set(
            redis_key,
            json.dumps(trimmed, ensure_ascii=False),
            ex=3600
        )

        logger.info(
            "[CHAT_HISTORY][SAVE] key=%s messages=%d",
            redis_key,
            len(trimmed)
        )

        logger.info(
            "[CHAT_HISTORY][SAVE] History=%s",
            trimmed
        )

    except Exception as e:
        logger.error(
            "[CHAT_HISTORY][SAVE] Failed for key=%s: %s",
            redis_key,
            e,
            exc_info=True
        )

#def get_conversation_history(user_key: str) -> list:
#    h = redis_client.get(f"user_{user_key}_history")
#    return json.loads(h) if h else []


#def save_conversation_history(user_key: str, history: list):
#    trimmed = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
#    redis_client.set(f"user_{user_key}_history", json.dumps(trimmed), ex=3600)

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"[ \t]+")


def _strip_html_for_history(text: str) -> str:
    """
    Strip HTML markup before persisting an assistant turn to Redis history.

    We store/replay conversation history verbatim into future LLM calls (see
    run_chatbot_graph), so a raw HTML table (<table><tr><th>...) costs real
    tokens on every follow-up turn for pure formatting with no extra meaning,
    and any malformed/dangling tags in a stored response get replayed back to
    the model as if they were its own prior output. Stripping tags here keeps
    the actual data (names, numbers, values) available for the model to
    resolve follow-ups like "put them in bullets" against, without the
    markup overhead or corruption risk. The original HTML is still streamed
    to the client unchanged — only what gets persisted/replayed changes.
    """
    if not text:
        return text
    stripped = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def _strip_html_incremental(chunk: str, pending_tag: str) -> tuple[str, str]:
    """
    Strip HTML tags from streamed chunks while preserving partial tags across chunk boundaries.
    """
    if not chunk:
        return "", pending_tag

    data = f"{pending_tag}{chunk}" if pending_tag else chunk
    out_chars = []
    in_tag = False
    tag_start = -1

    for idx, ch in enumerate(data):
        if in_tag:
            if ch == ">":
                in_tag = False
                tag_start = -1
            continue

        if ch == "<":
            in_tag = True
            tag_start = idx
            continue

        out_chars.append(ch)

    next_pending_tag = data[tag_start:] if in_tag and tag_start != -1 else ""
    return "".join(out_chars), next_pending_tag



# ────────────────────────────────────────────────────────────────────────────────
# BACKGROUND INDEX BUILDERS (education & policy only)
# ────────────────────────────────────────────────────────────────────────────────
def background_education_rebuilder():
    while True:
        try:
            changed = load_excel_to_sqlite(EDU_CFG)
            if changed:
                logger.info("[EducationRebuilder] Education SQLite tables reloaded.")
        except Exception as e:
            logger.error(f"[EducationRebuilder] Failed: {e}")
        time.sleep(7200)


def background_policy_rebuilder():
    while True:
        try:
            update_policy_index_if_changed(POLICY_CFG, splitter, emb)
        except Exception as e:
            logger.error(f"[PolicyRebuilder] Failed: {e}")
        time.sleep(86400 * 3)


# ────────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ────────────────────────────────────────────────────────────────────────────────
# JWT VERIFICATION
# ────────────────────────────────────────────────────────────────────────────────
# The main backend handles login and issues JWTs.
# This chatbot service ONLY verifies the token and extracts user_id.
# Both services must share the same JWT_SECRET (set via env var).
# ────────────────────────────────────────────────────────────────────────────────
security = HTTPBearer()


def _create_test_token(user_id: int) -> str:
    """Generate a test JWT token for development/testing."""
    from datetime import timezone
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=cfg.JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, cfg.JWT_SECRET, algorithm=cfg.JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict:
    """Decode and verify a JWT token. Raises on invalid/expired."""
    return jwt.decode(
        token, cfg.JWT_SECRET, algorithms=[cfg.JWT_ALGORITHM],
        options={"verify_sub": False},
    )


def _extract_user_id(payload: dict) -> int | None:
    """
    Extract user ID from JWT payload using configured claim chain.
    Example USER_ID_CLAIM: "sub/user_id/id"
    """
    claim_chain = [c.strip() for c in (cfg.USER_ID_CLAIM or "").split("/") if c.strip()]
    if not claim_chain:
        claim_chain = ["sub", "user_id", "id"]

    for claim in claim_chain:
        val = payload.get(claim)
        if val is not None and val != "":
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None


# ── Test endpoint: generate a token for testing (disable in production) ──
# @app.post("/api/test/token")
# async def generate_test_token(user_id: int = 1):
#     token = _create_test_token(user_id)
#     return {"access_token": token, "user_id": user_id}


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> int:
    """FastAPI dependency: extract user_id from Bearer token."""
    try:
        payload = _decode_jwt(credentials.credentials)
        user_id = _extract_user_id(payload)
        if user_id is None:
            raise HTTPException(status_code=401, detail="Token missing user_id")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.error(f"[JWT] InvalidTokenError: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ────────────────────────────────────────────────────────────────────────────────
# MAIN API ENDPOINT (LangGraph-based)
# ────────────────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    conversation_id: str = "default"


@app.post("/api/query")
async def handle_query(
    payload: QueryRequest,
    user_id: int = Depends(get_current_user),
):
    query = payload.query.strip()
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    rbac_user_id = user_id
    conv_id = payload.conversation_id.strip()
    history_key = f"uid:{rbac_user_id}:conv:{conv_id}"

    conversation_history = get_conversation_history(history_key)
    conversation_history.append({"role": "user", "content": query})

    # Fetch user info and available tools in one connection
    with pg_conn() as conn:
        user_profile = fetch_user_profile(conn, rbac_user_id)
        user_name = (
            user_profile.get("full_name_en")
            or user_profile.get("full_name_ar")
            or "Unknown User"
        )
        user_role = user_profile.get("designation") or ""
        user_email = user_profile.get("email") or ""
        user_contact_no = user_profile.get("contact_no") or ""
        available_tools = build_available_tools(conn, rbac_user_id)

    def generate():
        assistant_response = ""
        tool_results_for_chart = []
        pending_tag = ""

        try:
            for chunk in run_chatbot_graph(
                query=query,
                conversation_history=conversation_history,
                available_tools=available_tools,
                user_id=rbac_user_id,
                user_name=user_name,
                user_role=user_role,
                user_email=user_email,
                user_contact_no=user_contact_no,
                client=client,
                model=cfg.AZURE_OPENAI_DEPLOYMENT,
                pg_conn_fn=pg_conn,
                policy_cfg=POLICY_CFG,
                emb=emb,
                tool_results_collector=tool_results_for_chart,
            ):
                assistant_response += chunk
                plain_text_chunk, pending_tag = _strip_html_incremental(chunk, pending_tag)
                if plain_text_chunk:
                    yield plain_text_chunk

        except Exception as e:
            logger.error(f"Error in tool response generation: {e}")
            error_msg = (
                "I encountered an error processing your request. "
                "Please try again or rephrase your question."
            )
            assistant_response = error_msg
            yield error_msg

        # Save conversation history (plain-text — see _strip_html_for_history)
        conversation_history.append(
            {"role": "assistant", "content": _strip_html_for_history(assistant_response)}
        )
        save_conversation_history(history_key, conversation_history)

        # Detect chart opportunity and build final payload
        chart_data = None
        try:
            chart_data = detect_chart_opportunity(
                query, tool_results_for_chart, assistant_response
            )
        except Exception as e:
            logger.error(f"Chart detection error: {e}")

        # Send the final <replace> payload
        if assistant_response:
            if chart_data:
                final_payload = json.dumps(
                    {"html": assistant_response, "chart_data": chart_data},
                    ensure_ascii=False,
                    default=str,
                )
                yield f"<replace>{final_payload}</replace>"
            else:
                yield f"<replace>{assistant_response}</replace>"

    return StreamingResponse(generate(), media_type="text/html")


# ────────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT (for Flutter / mobile clients)
# ────────────────────────────────────────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """
    WebSocket for real-time streaming chat.

    Client flow:
    1. Connect to ws://host/ws/chat
    2. Send JSON: {"token": "<jwt>", "query": "...", "conversation_id": "default"}
    3. Receive streamed JSON messages:
       - {"type": "chunk", "content": "..."} — partial plain text
       - {"type": "done",  "html": "...", "chart_data": {...} | null} — final result
       - {"type": "error", "message": "..."} — on failure
    4. Send another query or close connection
    """
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()

            # ── Auth via JWT token ──
            token = data.get("token", "")
            try:
                jwt_payload = _decode_jwt(token)
                rbac_user_id = _extract_user_id(jwt_payload)
                if rbac_user_id is None:
                    await ws.send_json({"type": "error", "message": "Token missing user_id"})
                    continue
            except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
                await ws.send_json({"type": "error", "message": f"Auth failed: {e}"})
                continue

            query = (data.get("query") or "").strip()
            if not query:
                await ws.send_json({"type": "error", "message": "Empty query"})
                continue

            conv_id = (data.get("conversation_id") or "default").strip()
            history_key = f"uid:{rbac_user_id}:conv:{conv_id}"

            conversation_history = get_conversation_history(history_key)
            conversation_history.append({"role": "user", "content": query})

            # Fetch user info
            with pg_conn() as conn:
                user_profile = fetch_user_profile(conn, rbac_user_id)
                user_name = (
                    user_profile.get("full_name_en")
                    or user_profile.get("full_name_ar")
                    or "Unknown User"
                )
                user_role = user_profile.get("designation") or ""
                user_email = user_profile.get("email") or ""
                user_contact_no = user_profile.get("contact_no") or ""
                available_tools = build_available_tools(conn, rbac_user_id)

            # Stream response using async queue to avoid blocking the event loop
            assistant_response = ""
            tool_results_for_chart = []
            pending_tag = ""
            async_q = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _run_graph_sync():
                """Run the blocking LangGraph generator in a thread, pushing chunks to async queue."""
                try:
                    for chunk in run_chatbot_graph(
                        query=query,
                        conversation_history=conversation_history,
                        available_tools=available_tools,
                        user_id=rbac_user_id,
                        user_name=user_name,
                        user_role=user_role,
                        user_email=user_email,
                        user_contact_no=user_contact_no,
                        client=client,
                        model=cfg.AZURE_OPENAI_DEPLOYMENT,
                        pg_conn_fn=pg_conn,
                        policy_cfg=POLICY_CFG,
                        emb=emb,
                        tool_results_collector=tool_results_for_chart,
                    ):
                        loop.call_soon_threadsafe(async_q.put_nowait, ("chunk", chunk))
                except Exception as e:
                    logger.error(f"WS tool response error: {e}")
                    loop.call_soon_threadsafe(async_q.put_nowait, ("error", str(e)))
                finally:
                    loop.call_soon_threadsafe(async_q.put_nowait, ("done", None))

            loop.run_in_executor(None, _run_graph_sync)

            # Consume chunks asynchronously — each chunk is sent immediately
            while True:
                msg_type, msg_data = await async_q.get()
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    assistant_response = "I encountered an error processing your request."
                    await ws.send_json({"type": "error", "message": assistant_response})
                    break
                elif msg_type == "chunk":
                    assistant_response += msg_data
                    plain, pending_tag = _strip_html_incremental(msg_data, pending_tag)
                    if plain:
                        await ws.send_json({"type": "chunk", "content": plain})

            # Save history (plain-text — see _strip_html_for_history)
            conversation_history.append(
                {"role": "assistant", "content": _strip_html_for_history(assistant_response)}
            )
            save_conversation_history(history_key, conversation_history)

            # Chart detection
            chart_data = None
            try:
                chart_data = detect_chart_opportunity(
                    query, tool_results_for_chart, assistant_response
                )
            except Exception as e:
                logger.error(f"WS chart detection error: {e}")

            # Send final result
            await ws.send_json({
                "type": "done",
                "html": assistant_response,
                "chart_data": json.loads(
                    json.dumps(chart_data, default=str, ensure_ascii=False)
                ) if chart_data else None,
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────────
# LOGIN / SESSIONS (kept for testing UI)
# ────────────────────────────────────────────────────────────────────────────────
credentials = {"hypernym1": "hyper@chatbot", "hypernym2": "hyper@chatbot"}
SESSION_LIFETIME = timedelta(hours=1)


def init_db():
    conn = sqlite3.connect("sessions.db")
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS active_sessions
           (username TEXT PRIMARY KEY, last_active TIMESTAMP)"""
    )
    conn.commit()
    conn.close()


def add_session(username):
    conn = sqlite3.connect("sessions.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO active_sessions (username, last_active) VALUES (?, ?)",
        (username, datetime.now()),
    )
    conn.commit()
    conn.close()


def remove_session(username):
    conn = sqlite3.connect("sessions.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM active_sessions WHERE username = ?", (username,))
    conn.commit()
    conn.close()


def cleanup_expired_sessions():
    expiration_time = datetime.now() - SESSION_LIFETIME
    conn = sqlite3.connect("sessions.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM active_sessions WHERE last_active < ?", (expiration_time,))
    conn.commit()
    conn.close()


def count_active_sessions():
    cleanup_expired_sessions()
    conn = sqlite3.connect("sessions.db")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM active_sessions")
    c = cur.fetchone()[0]
    conn.close()
    return c


def is_user_logged_in(username):
    cleanup_expired_sessions()
    conn = sqlite3.connect("sessions.db")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM active_sessions WHERE username = ?", (username,))
    r = cur.fetchone()
    conn.close()
    return r is not None


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error_message": None})


@app.post("/", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    if count_active_sessions() >= 2:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error_message": "Maximum number of users are currently logged in. Please wait."},
        )

    if credentials.get(username) != password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error_message": "Invalid credentials. Please try again."},
        )

    if is_user_logged_in(username):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error_message": "This user is already logged in from another session."},
        )

    request.session["username"] = username
    add_session(username)
    return RedirectResponse(url="/home", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    username = request.session.pop("username", None)
    if username:
        remove_session(username)
    return RedirectResponse(url="/", status_code=303)


@app.get("/home", response_class=HTMLResponse)
async def index(request: Request):
    username = request.session.get("username")
    if not username or not is_user_logged_in(username):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request})


# ────────────────────────────────────────────────────────────────────────────────
# DOCS ROUTES
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/documents", response_class=HTMLResponse)
async def list_documents_page(request: Request):
    username = request.session.get("username")
    if not username or not is_user_logged_in(username):
        return RedirectResponse(url="/", status_code=303)
    if username != "hypernym1":
        return HTMLResponse("Unauthorized", status_code=401)
    document_list = documents.fetch_documents()
    return templates.TemplateResponse(
        "documents.html", {"request": request, "documents": document_list}
    )


@app.get("/download/{document_id}")
async def download_document(document_id: int):
    document = documents.get_document_path(document_id)
    if document:
        document_name, file_path = document
        return FileResponse(file_path, filename=document_name)
    return HTMLResponse("Document not found", status_code=404)


# ────────────────────────────────────────────────────────────────────────────────
# STARTUP
# ────────────────────────────────────────────────────────────────────────────────
init_db()


def _startup_edu_load():
    try:
        load_excel_to_sqlite(EDU_CFG)
        logger.info("[Startup] Education SQLite tables initialized.")
    except Exception as e:
        logger.warning(f"[Startup] Education table init failed (will retry): {e}")


threading.Thread(target=_startup_edu_load, daemon=True).start()
threading.Thread(target=background_education_rebuilder, daemon=True).start()
threading.Thread(target=background_policy_rebuilder, daemon=True).start()

# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting EHCD Chatbot (LangGraph + FastAPI)")
    uvicorn.run(app, host="0.0.0.0", port=8080)
