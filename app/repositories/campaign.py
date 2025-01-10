from typing import List, Optional
from ..db.models.campaign import Campaign
from ..db.models.lead import LeadInformationList
from ..db.models.chatbot import Chatbot
from .base import BaseRepository

class CampaignRepository(BaseRepository[Campaign]):
    def __init__(self):
        super().__init__(Campaign)

    async def create_campaign(self, data: dict) -> Campaign:
        # Validate relationships
        try:
            chatbot = Chatbot.get(data['chatbot_id'])
            leads_list = LeadInformationList.get(data['leads_list_id'])
        except (Chatbot.DoesNotExist, LeadInformationList.DoesNotExist):
            raise ValueError("Invalid chatbot_id or leads_list_id")

        campaign = await self.create(**data)
        
        # Update chatbot's lead list IDs
        if leads_list.id not in chatbot.lead_information_list_ids:
            chatbot.lead_information_list_ids.append(leads_list.id)
            chatbot.save()

        return campaign

    async def update_status(self, campaign_id: str, status: str) -> Campaign:
        campaign = await self.get_by_id(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")
        
        campaign.status = status
        campaign.save()
        return campaign

    async def update_metrics(self, campaign_id: str, metrics: dict) -> Campaign:
        campaign = await self.get_by_id(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")
        
        for key, value in metrics.items():
            setattr(campaign.metrics, key, value)
        campaign.save()
        return campaign 