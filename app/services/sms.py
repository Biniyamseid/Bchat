from typing import Optional, Dict
from fastapi import HTTPException
from ..repositories.chat import ChatRepository
from ..repositories.lead import LeadRepository
from ..core.config import settings
from ..utils.ai import OpenAIClient
from datetime import datetime

class SMSService:
    def __init__(self):
        self.chat_repository = ChatRepository()
        self.lead_repository = LeadRepository()
        self.ai_client = OpenAIClient()

    async def process_incoming_sms(
        self,
        phone_number: str,
        message: str
    ) -> Dict:
        try:
            # Get or create lead
            lead = await self.lead_repository.get_by_phone(phone_number)
            if not lead:
                raise HTTPException(status_code=404, detail="Lead not found")

            # Get or create chat session
            session = await self.chat_repository.get_or_create_session(
                user_id=phone_number,
                title=f"SMS Chat with {lead.name}"
            )

            # Add user message
            session.add_message("user", message)

            # Get chatbot response
            response = await self.ai_client.get_response(
                message=message,
                lead_info=lead.to_dict(),
                chat_history=session.messages
            )

            # Add AI response
            session.add_message("assistant", response)

            return {
                "session_id": session.id,
                "response": response,
                "lead": lead.to_dict()
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def send_sms(
        self,
        phone_number: str,
        message: str,
        session_id: Optional[str] = None
    ) -> Dict:
        try:
           
            if session_id:
                session = await self.chat_repository.get_by_id(session_id)
                if session:
                    session.add_message("assistant", message)

            return {
                "status": "sent",
                "to": phone_number,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 