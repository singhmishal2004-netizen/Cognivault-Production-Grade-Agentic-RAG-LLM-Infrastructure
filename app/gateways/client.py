from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


PRIMARY_MODEL = f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_ID
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + optional saved config). The @virtual-key/model-name format is Portkey-specific — Groq's own client
      does not understand it. Portkey routes the configured model through the selected virtual key.
    """
    headers = createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        metadata={
            "feature": feature,
            "_user": "rag-system",
            "environment": "production"
        },
    )
    if settings.PORTKEY_CONFIG_ID:
        headers = createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            },
        )

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PRIMARY_MODEL,
        temperature=0,
        default_headers=headers,
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"