import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    anthropic_api_key: str = os.environ["ANTHROPIC_API_KEY"]
    voyage_api_key: str = os.environ["VOYAGE_API_KEY"]
    top_k: int = int(os.environ.get("RAG_TOP_K", 4))
    grounding_threshold: float = float(os.environ.get("RAG_GROUNDING_THRESHOLD", 0.5))
    allowed_origins: list[str] = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")


settings = Settings()
