from fastapi import APIRouter
from .endpoints import chat, campaign, chatbot, leads, sms

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(campaign.router, prefix="/campaign", tags=["campaign"])
api_router.include_router(chatbot.router, prefix="/chatbot", tags=["chatbot"])
api_router.include_router(leads.router, prefix="/leads", tags=["leads"])
api_router.include_router(sms.router, prefix="/sms", tags=["sms"]) 