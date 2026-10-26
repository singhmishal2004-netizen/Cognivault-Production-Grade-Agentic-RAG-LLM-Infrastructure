import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT: str | None = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION_NAME", "cognivault")

    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY: str | None = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL: str | None = os.getenv("GROQ_MODEL")
    PORTKEY_API_KEY: str | None = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID: str | None = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG: str = os.getenv("GROQ_SLUG", "grok-api-key")
    GROQ_SLUG_2: str = os.getenv("GROQ_SLUG_2", "groq-fallback-api-key")


settings = Settings()
