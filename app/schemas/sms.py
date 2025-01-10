from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class SMSRequest(BaseModel):
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    message: str
    session_id: Optional[str] = None

class SMSResponse(BaseModel):
    session_id: Optional[str]
    response: str
    status: str = "success"
    timestamp: datetime = datetime.utcnow()
    lead: Optional[dict] = None 