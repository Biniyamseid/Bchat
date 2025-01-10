from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

class CampaignSchedule(BaseModel):
    start_date: str
    end_date: str
    time_zone: str
    daily_start_time: str
    daily_end_time: str

class CampaignCreate(BaseModel):
    name: str
    chatbot_id: str
    leads_list_id: str
    status: str = "active"
    schedule: Optional[CampaignSchedule] = None

class CampaignUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    schedule: Optional[CampaignSchedule] = None

class CampaignResponse(BaseModel):
    id: str
    name: str
    chatbot_id: str
    leads_list_id: str
    status: str
    schedule: Optional[Dict] = None
    metrics: Optional[Dict] = None
    created_at: datetime
    updated_at: datetime 