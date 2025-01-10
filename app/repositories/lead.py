from typing import List, Optional
from ..db.models.lead import LeadInformation, LeadInformationList
from .base import BaseRepository
import uuid

class LeadRepository(BaseRepository[LeadInformation]):
    def __init__(self):
        super().__init__(LeadInformation)

    async def get_by_phone(self, phone_number: str) -> Optional[LeadInformation]:
        try:
            return next(LeadInformation.query(
                phone_number,
                limit=1,
                scan_index_forward=False
            ))
        except StopIteration:
            return None

    async def create_lead_list(self, name: str, lead_ids: List[str] = None) -> LeadInformationList:
        lead_list = LeadInformationList(
            id=str(uuid.uuid4()),
            name=name,
            lead_information_ids=lead_ids or []
        )
        lead_list.save()
        return lead_list

    async def add_to_list(self, list_id: str, lead_id: str) -> bool:
        try:
            lead_list = LeadInformationList.get(list_id)
            if lead_id not in lead_list.lead_information_ids:
                lead_list.lead_information_ids.append(lead_id)
                lead_list.save()
            return True
        except LeadInformationList.DoesNotExist:
            return False

    async def get_leads_by_list(self, list_id: str) -> List[LeadInformation]:
        try:
            lead_list = LeadInformationList.get(list_id)
            leads = []
            for lead_id in lead_list.lead_information_ids:
                try:
                    lead = await self.get_by_phone(lead_id)
                    if lead:
                        leads.append(lead)
                except Exception:
                    continue
            return leads
        except LeadInformationList.DoesNotExist:
            return [] 