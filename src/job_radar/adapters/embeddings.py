import httpx

from job_radar.config import settings


async def embed(text: str) -> list[float]:
    payload = {"model": settings.embedding_model, "input": text}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url=f"{settings.ollama_base_url}/api/embed", json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"][0]
