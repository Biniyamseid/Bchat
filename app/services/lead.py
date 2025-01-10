from typing import List, Optional
from fastapi import HTTPException
from ..repositories.lead import LeadRepository
from ..schemas.lead import LeadCreate, LeadResponse, LeadListCreate
from ..core.exceptions import NotFoundException

class LeadService:
    def __init__(self):
        self.repository = LeadRepository()

    async def create_lead(self, data: LeadCreate) -> LeadResponse:
        try:
            # Check if lead already exists
            existing_lead = await self.repository.get_by_phone(data.phone_number)
            if existing_lead:
                raise HTTPException(
                    status_code=400,
                    detail="Lead with this phone number already exists"
                )

            lead = await self.repository.create(**data.dict())
            return LeadResponse(**lead.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def create_lead_list(self, data: LeadListCreate) -> dict:
        try:
            lead_list = await self.repository.create_lead_list(
                name=data.name,
                lead_ids=data.lead_ids
            )
            return lead_list.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_leads_by_list(self, list_id: str) -> List[LeadResponse]:
        try:
            leads = await self.repository.get_leads_by_list(list_id)
            return [LeadResponse(**lead.to_dict()) for lead in leads]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def update_lead_chatbot(
        self,
        phone_number: str,
        chatbot_id: str
    ) -> LeadResponse:
        try:
            lead = await self.repository.get_by_phone(phone_number)
            if not lead:
                raise NotFoundException("Lead not found")

            lead.chatbot_id = chatbot_id
            lead.save()
            return LeadResponse(**lead.to_dict())
        except NotFoundException as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 