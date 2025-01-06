from typing import List
import openai
from ..core.config import get_settings

settings = get_settings()

class EmbeddingService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
    
    async def create_embedding(self, text: str) -> List[float]:
        response = await openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding