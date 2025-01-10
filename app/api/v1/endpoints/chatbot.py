from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ....schemas.chatbot import ChatbotCreate, ChatbotResponse, ChatbotUpdate
from ....services.chatbot import ChatbotService
from ....core.security import get_current_user

router = APIRouter()

@router.get("/", response_model=List[ChatbotResponse])
async def get_all_chatbots(
    current_user = Depends(get_current_user),
    service: ChatbotService = Depends()
):
    """Get all chatbots"""
    return await service.get_all()

@router.get("/{chatbot_id}", response_model=ChatbotResponse)
async def get_chatbot(
    chatbot_id: str,
    current_user = Depends(get_current_user),
    service: ChatbotService = Depends()
):
    """Get a specific chatbot by ID"""
    return await service.get_by_id(chatbot_id)

@router.post("/", response_model=ChatbotResponse)
async def create_chatbot(
    chatbot: ChatbotCreate,
    current_user = Depends(get_current_user),
    service: ChatbotService = Depends()
):
    """Create a new chatbot"""
    return await service.create(chatbot)

@router.get("/{chatbot_id}/lead-lists")
async def get_chatbot_lead_lists(
    chatbot_id: str,
    current_user = Depends(get_current_user),
    service: ChatbotService = Depends()
):
    """Get all lead lists associated with a chatbot"""
    return await service.get_lead_lists(chatbot_id) 