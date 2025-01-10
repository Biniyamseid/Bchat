from typing import List, Optional
from fastapi import HTTPException
from ..repositories.campaign import CampaignRepository
from ..schemas.campaign import CampaignCreate, CampaignResponse, CampaignUpdate
from ..core.exceptions import NotFoundException

class CampaignService:
    def __init__(self):
        self.repository = CampaignRepository()

    async def create_campaign(self, data: CampaignCreate) -> CampaignResponse:
        try:
            campaign = await self.repository.create_campaign(data.dict())
            return CampaignResponse(**campaign.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_campaign(self, campaign_id: str) -> CampaignResponse:
        campaign = await self.repository.get_by_id(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return CampaignResponse(**campaign.to_dict())

    async def update_campaign_status(
        self,
        campaign_id: str,
        status: str
    ) -> CampaignResponse:
        try:
            campaign = await self.repository.update_status(campaign_id, status)
            return CampaignResponse(**campaign.to_dict())
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_campaign_analytics(self, campaign_id: str) -> dict:
        campaign = await self.repository.get_by_id(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        
        return {
            "campaign_id": campaign.id,
            "metrics": campaign.metrics.as_dict() if campaign.metrics else {},
            "status": campaign.status,
            "created_at": campaign.created_at.isoformat(),
            "updated_at": campaign.updated_at.isoformat()
        } 