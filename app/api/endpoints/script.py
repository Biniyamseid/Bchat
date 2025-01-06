from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import uuid
from ...core.config import get_settings
from ...models.dynamodb import ChatbotScript
from ...services.pdf_service import PDFService
from ...schemas.script import ScriptResponse

router = APIRouter()
settings = get_settings()

@router.post("/upload", response_model=ScriptResponse)
async def upload_script(
    file: UploadFile = File(...),
    pdf_service: PDFService = Depends()
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        content = await pdf_service.extract_text(file)
        script = ChatbotScript(
            id=str(uuid.uuid4()),
            content=content
        )
        script.save()
        
        return {
            "id": script.id,
            "message": "Script uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from ...models.dynamodb import ChatbotScript
from ...services.pdf_service import PDFService
from ...schemas.script import ScriptResponse
import uuid

# router = APIRouter()

@router.post("/upload1", response_model=ScriptResponse)
async def upload_script(
    file: UploadFile = File(...),
    pdf_service: PDFService = Depends()
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        content = await pdf_service.extract_text(file)
        script = ChatbotScript(
            id=str(uuid.uuid4()),
            content=content
        )
        script.save()
        
        return {
            "id": script.id,
            "message": "Script uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
import logging

logger = logging.getLogger(__name__)

@router.post("/upload3", response_model=ScriptResponse)
async def upload_script(
    file: UploadFile = File(...),
    pdf_service: PDFService = Depends()
):
    logger.info(f"Received file: {file.filename}")
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        content = await pdf_service.extract_text(file)
        logger.info(f"Extracted content: {content[:100]}")  # Log first 100 characters of content
        script = ChatbotScript(
            id=str(uuid.uuid4()),
            content=content,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        script.save()
        
        return {
            "id": script.id,
            "content": script.content,
            "created_at": script.created_at,
            "updated_at": script.updated_at,
            "message": "Script uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading script: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/script/{script_id}", response_model=ScriptResponse)
async def get_script(script_id: str):
    try:
        script = ChatbotScript.get(script_id)
        if not script:
            raise HTTPException(status_code=404, detail="Script not found")
        
        return {
            "id": script.id,
            "content": script.content,
            "created_at": script.created_at,
            "updated_at": script.updated_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    