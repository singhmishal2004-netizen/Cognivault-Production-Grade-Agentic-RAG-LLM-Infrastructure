import re

import logfire
from nemoguardrails import RailsConfig, LLMRails

from app.gateways.client import get_langchain_llm
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

_PROMPT_OVERRIDE_PATTERNS = (
    re.compile(
        r"\b(?:rewrite|change|replace|modify|reveal|show|print|repeat|quote|disclose)"
        r"\b.{0,80}\b(?:system|developer|hidden)\s+(?:prompt|instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,60}"
        r"\b(?:previous|prior|system|developer)\s+(?:instructions?|rules?|prompts?)\b",
        re.IGNORECASE,
    ),
)

_PROMPT_OVERRIDE_RESPONSE = (
    "I can’t rewrite or reveal internal instructions. "
    "I can help with supported questions about Kubernetes, Intel hardware, or enterprise networking."
)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the same Portkey-backed model as the planner and responder.
    """
    global _rails

    guard_llm = get_langchain_llm(feature="guardrails")

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info("🛡️ NeMo Guardrails initialised (Portkey-backed model).")
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if any(pattern.search(message) for pattern in _PROMPT_OVERRIDE_PATTERNS):
        logfire.warning("Prompt-override request blocked by input guardrail.")
        return True, _PROMPT_OVERRIDE_RESPONSE

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        normalized_content = content.casefold()
        fired = any(
            indicator.casefold() in normalized_content
            for indicator in RAIL_INDICATORS
        )

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None