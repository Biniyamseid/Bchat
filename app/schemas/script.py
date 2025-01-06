
    
    
from pydantic import BaseModel
from datetime import datetime

class ScriptResponse(BaseModel):
    id: str
    content: str
    created_at: datetime
    updated_at: datetime
    message: str = "Script retrieved successfully"