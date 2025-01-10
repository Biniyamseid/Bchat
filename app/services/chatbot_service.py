from typing import List
from fastapi import HTTPException
from ..db.repositories.chatbot_repository import ChatbotRepository
from ..schemas.chatbot import ChatbotCreate, ChatbotResponse

class ChatbotService:
    def __init__(self):
        self.repository = ChatbotRepository()

    async def get_all_chatbots(self) -> List[ChatbotResponse]:
        try:
            return await self.repository.get_all()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_lead_lists(self, chatbot_id: str):
        try:
            return await self.repository.get_lead_lists(chatbot_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e)) 