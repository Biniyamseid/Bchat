from typing import List, Optional
from fastapi import HTTPException
from ..repositories.chat import ChatRepository
from ..schemas.chat import ChatMessageCreate, ChatSessionCreate, ChatResponse
from ..core.exceptions import NotFoundException

class ChatService:
    def __init__(self):
        self.repository = ChatRepository()

    async def create_session(self, data: ChatSessionCreate) -> ChatResponse:
        try:
            session = await self.repository.create(**data.dict())
            return ChatResponse(**session.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def add_message(
        self,
        session_id: str,
        message: ChatMessageCreate
    ) -> ChatResponse:
        try:
            session = await self.repository.add_message(
                session_id=session_id,
                role=message.role,
                content=message.content
            )
            if not session:
                raise NotFoundException("Chat session not found")
            return ChatResponse(**session.to_dict())
        except NotFoundException as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_chat_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[dict]:
        try:
            return await self.repository.get_chat_history(user_id, limit)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 