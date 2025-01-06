from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    

class ChatRequest(BaseModel):
    session_id: Optional[str]
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str

class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    current_status: str
    user_id: str
    session_id: str
    context: Dict[str, Any] = {}

class LeadInformationResponse(BaseModel):
    phone_number: str
    name: str
    email: str
    detail: Optional[str] = None
    chatbot_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str

class AgentState(BaseModel):
    messages: List[Dict[str, str]]
    current_status: str
    user_id: str
    session_id: str
    context: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str

class MessageRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None