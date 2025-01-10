from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class LeadBase(BaseModel):
    name: str
    email: EmailStr
    phone_number: str
    detail: Optional[str] = None

class LeadCreate(LeadBase):
    chatbot_id: Optional[str] = None

class LeadUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    detail: Optional[str] = None
    chatbot_id: Optional[str] = None

class LeadResponse(LeadBase):
    chatbot_id: Optional[str]
    created_at: datetime
    updated_at: datetime

class LeadListCreate(BaseModel):
    name: str
    lead_ids: Optional[List[str]] = None

class LeadListResponse(BaseModel):
    id: str
    name: str
    lead_information_ids: List[str]
    created_at: datetime
    updated_at: datetime 