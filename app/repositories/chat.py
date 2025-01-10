from typing import List, Optional
from ..db.models.chat import ChatSession
from .base import BaseRepository
from datetime import datetime, timezone

class ChatRepository(BaseRepository[ChatSession]):
    def __init__(self):
        super().__init__(ChatSession)

    async def get_user_sessions(self, user_id: str) -> List[ChatSession]:
        return [
            session for session in ChatSession.query(
                user_id,
                scan_index_forward=False
            )
        ]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> Optional[ChatSession]:
        session = await self.get_by_id(session_id)
        if session:
            session.add_message(role, content)
            return session
        return None

    async def get_chat_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[dict]:
        sessions = await self.get_user_sessions(user_id)
        history = []
        for session in sessions[:limit]:
            history.extend(session.messages)
        return [msg.as_dict() for msg in sorted(
            history,
            key=lambda x: x.timestamp
        )] 