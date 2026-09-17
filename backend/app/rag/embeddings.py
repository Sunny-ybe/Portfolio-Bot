import voyageai

from app.config import settings

_client = voyageai.Client(api_key=settings.voyage_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    result = _client.embed(texts, model="voyage-3", input_type="document")
    return result.embeddings


def embed_query(text: str) -> list[float]:
    result = _client.embed([text], model="voyage-3", input_type="query")
    return result.embeddings[0]
