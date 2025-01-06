from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import uuid
from ...core.config import get_settings
from ...models.dynamodb import CompanyKnowledge
from ...services.embedding_service import EmbeddingService
from ...schemas.knowledge import KnowledgeResponse

router = APIRouter()
settings = get_settings()                                                                                                                                                                                                                                                                                                                                                                                                      

@router.post("/upload", response_model=KnowledgeResponse)
async def upload_knowledge(
    file: UploadFile = File(...),
    
    embedding_service: EmbeddingService = Depends()
):
    try:
        content = await file.read()
        content_text = content.decode()
        
        # Generate embeddings
        embedding = await embedding_service.create_embedding(content_text)
        
        knowledge = CompanyKnowledge(
            id=str(uuid.uuid4()),
            content=content_text,
            embedding=embedding,
            metadata={"filename": file.filename}
        )
        knowledge.save()
        
        return {
            "id": knowledge.id,
            "message": "Knowledge base updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))