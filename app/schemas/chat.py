from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime

class MessageBase(BaseModel):
    role: str
    content: str

class ChatMessageCreate(MessageBase):
    pass

class ChatMessage(MessageBase):
    id: str
    timestamp: datetime

class ChatSessionCreate(BaseModel):
    user_id: str
    title: str = "New Chat"

class ChatResponse(BaseModel):
    id: str
    user_id: str
    title: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime
    
    
class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    current_status: str
    user_id: str
    session_id: str
    context: Dict[str, Any] = {}