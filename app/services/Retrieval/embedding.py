import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from typing import TypeVar

import logfire
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

_T = TypeVar("_T")

BATCH_SIZE = 50
_GEMINI_RPM_LIMIT = 90
_GEMINI_TPM_LIMIT = 27_000
_GEMINI_RPD_LIMIT = 900
_GEMINI_BATCH_TOKEN_LIMIT = 24_000
_GEMINI_USAGE_DB = Path(__file__).resolve().parents[3] / "processed_data" / "gemini_usage.sqlite3"
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "fallback"


def _estimated_tokens(texts: list[str]) -> int:
    """Estimate input tokens conservatively for local quota pacing."""
    return sum(max(1, math.ceil(len(text) / 3)) for text in texts)


def _wait_for_gemini_quota(estimated_tokens: int) -> None:
    """Reserve quota under conservative rolling-minute and rolling-day limits."""
    if estimated_tokens > _GEMINI_TPM_LIMIT:
        raise ValueError(
            f"Gemini request is estimated at {estimated_tokens} tokens, "
            f"above the {_GEMINI_TPM_LIMIT}-token local per-minute limit."
        )

    _GEMINI_USAGE_DB.parent.mkdir(parents=True, exist_ok=True)
    while True:
        now = time.time()
        wait_seconds = 0.0
        with closing(sqlite3.connect(_GEMINI_USAGE_DB, timeout=30)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS gemini_requests "
                "(created_at REAL NOT NULL, estimated_tokens INTEGER NOT NULL)"
            )
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM gemini_requests WHERE created_at <= ?",
                (now - 86_400,),
            )
            daily = connection.execute(
                "SELECT created_at FROM gemini_requests "
                "WHERE created_at > ? ORDER BY created_at",
                (now - 86_400,),
            ).fetchall()
            recent = connection.execute(
                "SELECT created_at, estimated_tokens FROM gemini_requests "
                "WHERE created_at > ? ORDER BY created_at",
                (now - 60,),
            ).fetchall()

            if len(daily) >= _GEMINI_RPD_LIMIT:
                wait_seconds = max(0.1, daily[0][0] + 86_400 - now)
            elif len(recent) >= _GEMINI_RPM_LIMIT:
                wait_seconds = max(0.1, recent[0][0] + 60 - now)
            else:
                recent_tokens = sum(row[1] for row in recent)
                if recent_tokens + estimated_tokens > _GEMINI_TPM_LIMIT:
                    for created_at, tokens in recent:
                        recent_tokens -= tokens
                        if recent_tokens + estimated_tokens <= _GEMINI_TPM_LIMIT:
                            wait_seconds = max(0.1, created_at + 60 - now)
                            break

            if wait_seconds == 0:
                connection.execute(
                    "INSERT INTO gemini_requests VALUES (?, ?)",
                    (now, estimated_tokens),
                )
                connection.commit()
                return

        logfire.info(
            f"Waiting {wait_seconds:.1f}s to stay within Gemini embedding quota."
        )
        time.sleep(wait_seconds)


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    retry_delay = re.search(
        r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s",
        str(error),
        re.IGNORECASE,
    )
    if retry_delay:
        return float(retry_delay.group(1))
    return min(60.0, 2**attempt)


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(value in message for value in ("429", "rate", "quota", "resource_exhausted"))


def _gemini_call(call: Callable[[], _T], texts: list[str]) -> _T:
    estimated_tokens = _estimated_tokens(texts)
    for attempt in range(5):
        _wait_for_gemini_quota(estimated_tokens)
        try:
            return call()
        except Exception as error:
            if not _is_rate_limit_error(error) or attempt == 4:
                logfire.error(f"Gemini embedding failed: {error}")
                raise

            wait_seconds = _retry_delay_seconds(error, attempt)
            if wait_seconds > 300:
                raise RuntimeError(
                    f"Gemini requested a {wait_seconds:.0f}s retry delay; "
                    "stopping rather than blocking ingestion for more than 5 minutes."
                ) from error
            logfire.warning(
                f"Gemini rate limit hit — retrying in {wait_seconds:g}s "
                f"(attempt {attempt + 1}/5)."
            )
            time.sleep(wait_seconds)


def _batches(texts: list[str]) -> Iterator[list[str]]:
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        text_tokens = _estimated_tokens([text])
        if text_tokens > _GEMINI_BATCH_TOKEN_LIMIT:
            raise ValueError(
                f"Text is estimated at {text_tokens} tokens, above the "
                f"{_GEMINI_BATCH_TOKEN_LIMIT}-token Gemini batch limit."
            )
        if batch and (
            len(batch) >= BATCH_SIZE
            or batch_tokens + text_tokens > _GEMINI_BATCH_TOKEN_LIMIT
        ):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += text_tokens
    if batch:
        yield batch


# ── Model initialisation ───────────────────────────────────────────────────────

def _probe_gemini():
    """Try one embed call to verify Gemini is reachable. Returns model or None."""
    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.GEMINI_API_KEY,
        )
        _gemini_call(lambda: model.embed_query("probe"), ["probe"])
        logfire.info("Gemini embeddings ready (gemini-embedding-2-preview, 3072-dim).")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed: {e}. Will use sentence-transformers fallback.")
        return None


def _load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading sentence-transformers fallback (all-mpnet-base-v2, 768-dim).")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    """Initialise embedding model once per process. Called lazily on first use."""
    global _active_model, _model_type
    if _active_model is not None:
        return

    gemini = _probe_gemini()
    if gemini:
        _active_model = gemini
        _model_type = "gemini"
    else:
        _active_model = _load_fallback()
        _model_type = "fallback"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """Return the vector dimension for the active model. Call after _init()."""
    _init()
    return _GEMINI_DIM if _model_type == "gemini" else _FALLBACK_DIM


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type == "gemini":
        return _gemini_call(lambda: _active_model.embed_documents(batch), batch)
    else:
        return _active_model.encode(batch, show_progress_bar=False).tolist()


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _gemini_call(lambda: _active_model.embed_query(query), [query])
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    all_embeddings: list[list[float]] = []
    for i, batch in enumerate(_batches(texts)):
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))
    return all_embeddings