from fastapi import APIRouter, HTTPException, Depends
from typing import List
from ....schemas.lead import (
    LeadCreate,
    LeadResponse,
    LeadUpdate,
    LeadListCreate,
    LeadListResponse
)
from ....services.lead import LeadService
from ....core.security import get_current_user

router = APIRouter()

@router.post("/", response_model=LeadResponse)
async def create_lead(
    lead: LeadCreate,
    current_user = Depends(get_current_user),
    service: LeadService = Depends()
):
    """Create a new lead"""
    return await service.create_lead(lead)

@router.post("/lists", response_model=LeadListResponse)
async def create_lead_list(
    data: LeadListCreate,
    current_user = Depends(get_current_user),
    service: LeadService = Depends()
):
    """Create a new lead list"""
    return await service.create_lead_list(data)

@router.get("/lists/{list_id}/leads", response_model=List[LeadResponse])
async def get_leads_by_list(
    list_id: str,
    current_user = Depends(get_current_user),
    service: LeadService = Depends()
):
    """Get all leads in a list"""
    return await service.get_leads_by_list(list_id)

@router.patch("/{phone_number}/chatbot", response_model=LeadResponse)
async def update_lead_chatbot(
    phone_number: str,
    chatbot_id: str,
    current_user = Depends(get_current_user),
    service: LeadService = Depends()
):
    """Update a lead's assigned chatbot"""
    return await service.update_lead_chatbot(phone_number, chatbot_id) 