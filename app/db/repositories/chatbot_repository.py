from typing import List, Optional
from ..models.chatbot import Chatbot
from ..models.lead import LeadInformationList

class ChatbotRepository:
    async def get_all(self) -> List[Chatbot]:
        return [chatbot for chatbot in Chatbot.scan()]

    async def get_by_id(self, chatbot_id: str) -> Optional[Chatbot]:
        try:
            return Chatbot.get(hash_key=chatbot_id)
        except Chatbot.DoesNotExist:
            return None

    async def get_lead_lists(self, chatbot_id: str):
        chatbot = await self.get_by_id(chatbot_id)
        if not chatbot:
            raise ValueError("Chatbot not found")
            
        lead_lists = []
        if chatbot.lead_information_list_ids:
            for list_id in chatbot.lead_information_list_ids:
                try:
                    lead_list = LeadInformationList.get(hash_key=list_id)
                    lead_lists.append(lead_list)
                except LeadInformationList.DoesNotExist:
                    continue
        return lead_lists 