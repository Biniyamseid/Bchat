from typing import List
import uuid
from pynamodb.attributes import (
    UnicodeAttribute, 
    ListAttribute,
    MapAttribute,
    UTCDateTimeAttribute
)
from datetime import datetime, timezone
from ..base import BaseModel

class MessageAttribute(MapAttribute):
    id = UnicodeAttribute()
    role = UnicodeAttribute()
    content = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute()

class ChatSession(BaseModel):
    class Meta:
        table_name = 'chat_sessions'
    
    id = UnicodeAttribute(default=lambda: str(uuid.uuid4()))
    user_id = UnicodeAttribute(hash_key=True)
    created_at = UTCDateTimeAttribute(range_key=True, default=lambda: datetime.now(timezone.utc))
    title = UnicodeAttribute()
    messages = ListAttribute(of=MessageAttribute, default=list)

    def add_message(self, role: str, content: str):
        message = MessageAttribute(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc)
        )
        self.messages.append(message)
        self.save()
        return message

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "messages": [msg.as_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        } 