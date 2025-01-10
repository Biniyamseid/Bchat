import asyncio
from typing import Dict
import uuid
from pynamodb.models import Model
from pynamodb.attributes import (
    UnicodeAttribute,
    UTCDateTimeAttribute,
    MapAttribute,
    ListAttribute,
)
from pynamodb.exceptions import TableError, PynamoDBException
from pynamodb.connection import Connection
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

from pynamodb.models import Model
from pynamodb.attributes import (
    UnicodeAttribute,
    UTCDateTimeAttribute,
    ListAttribute,
    NumberAttribute,
    MapAttribute,
)
from datetime import datetime
import os
# Load environment variables from .env file
load_dotenv()


async def is_name_unique(model_class, name, exclude_id=None):
    filter_condition = model_class.name == name
    if exclude_id:
        filter_condition &= model_class.id != exclude_id
    
    def scan_items():
        items = model_class.scan(filter_condition, limit=1)
        try:
            next(items)
            return False  # Name already exists
        except StopIteration:
            return True  # Name is unique
    
    return await asyncio.to_thread(scan_items)





    
class SessionTable(Model):
    class Meta:
        table_name = 'SessionTable'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    session_id = UnicodeAttribute(hash_key=True)  # Hash key
    user_id = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)
    metadata = MapAttribute(null=True)
    data = MapAttribute(null=True)
    
    
class Chatbot(Model):
    class Meta:
        table_name = 'chatbots'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    chatbot_script_id = UnicodeAttribute()
    # lead_information_list_ids = ListAttribute(null=True, default=list)  # Changed to ListAttribute
    lead_information_list_ids = ListAttribute(of=UnicodeAttribute, null=True, default=list)
    instructions = UnicodeAttribute(null=True)
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)
    
    # Personal Details
    first_name = UnicodeAttribute(null=True)
    last_name = UnicodeAttribute(null=True)
    job_title = UnicodeAttribute(null=True)
    personality = UnicodeAttribute(null=True)
    
    # Company and Role Details
    company_name = UnicodeAttribute(null=True)
    company_address = UnicodeAttribute(null=True)
    department = UnicodeAttribute(null=True)
    industry = UnicodeAttribute(null=True)
    year_established = NumberAttribute(null=True)
    company_email = UnicodeAttribute(null=True)
    phone_number = UnicodeAttribute(null=True)
    company_website = UnicodeAttribute(null=True)
    job_description = UnicodeAttribute(null=True)
    priorities = ListAttribute(of=UnicodeAttribute, null=True)
    opinions = ListAttribute(of=UnicodeAttribute, null=True)
    ai_creativity = NumberAttribute(null=True)
    scorecard = MapAttribute(null=True)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'lead_information_list_ids': list(self.lead_information_list_ids) if self.lead_information_list_ids else [],
            'chatbot_script_id': self.chatbot_script_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }




class Message(MapAttribute):
    id = UnicodeAttribute()
    role = UnicodeAttribute()
    content = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute()
    
    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()  # Convert datetime to string
        }
        
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, UTCDateTimeAttribute, ListAttribute
from datetime import datetime, timezone
import os


class MessageAttribute(MapAttribute):
    id = UnicodeAttribute()
    role = UnicodeAttribute()
    content = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute()
class ChatSession(Model):
    class Meta:
        table_name = 'chat_sessions'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    id = UnicodeAttribute(default=lambda: str(uuid.uuid4()))
    user_id = UnicodeAttribute(hash_key=True)
    created_at = UTCDateTimeAttribute(range_key=True, default=lambda: datetime.now(timezone.utc))
    title = UnicodeAttribute()
    messages = ListAttribute(of=MessageAttribute, default=list)
    updated_at = UTCDateTimeAttribute(default=lambda: datetime.now(timezone.utc))

    def add_message(self, message: Dict):
        """Add a message to the session"""
        if self.messages is None:
            self.messages = []
            
        # Create a MessageAttribute instance
        message_attr = MessageAttribute(
            id=str(uuid.uuid4()),
            role=message.get("role", "user"),
            content=message.get("content", ""),
            timestamp=datetime.now(timezone.utc)
        )
        self.messages.append(message_attr)
        self.updated_at = datetime.now(timezone.utc)
        
    def get_messages(self):
        """Get all messages in the session"""
        return [
            {
                "id": msg.id if isinstance(msg, MessageAttribute) else msg.get("id"),
                "role": msg.role if isinstance(msg, MessageAttribute) else msg.get("role"),
                "content": msg.content if isinstance(msg, MessageAttribute) else msg.get("content"),
                "timestamp": msg.timestamp.isoformat() if isinstance(msg, MessageAttribute) else msg.get("timestamp")
            }
            for msg in self.messages
        ] if self.messages else []

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "title": self.title,
            "messages": self.get_messages(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class LeadInformationList(Model):
    class Meta:
        table_name = 'lead_information_lists'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    lead_information_ids = ListAttribute(of=UnicodeAttribute)  # Field to store LeadInformation IDs
    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)



class LeadInformation(Model):
    class Meta:
        table_name = 'lead_information'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    phone_number = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    email = UnicodeAttribute()
    detail = UnicodeAttribute(null=True)
    chatbot_id = UnicodeAttribute(null=True)
    created_at = UTCDateTimeAttribute(range_key=True, default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)

    @classmethod
    async def async_query(cls, hash_key, **kwargs):
        async for item in super(LeadInformation, cls).async_query(hash_key, **kwargs):
            yield cls.from_raw_data(item)
    
    def to_dict(self):
        return {
            "phone_number": self.phone_number,
            "name": self.name,
            "email": self.email,
            "detail": self.detail,
            "chatbot_id": self.chatbot_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }



    
    
from pynamodb.models import Model
from pynamodb.attributes import (
    UnicodeAttribute,
    UTCDateTimeAttribute,
)

class ChatbotScript(Model):
    class Meta:
        table_name = 'chatbot_scripts'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    id = UnicodeAttribute(hash_key=True)
    content = UnicodeAttribute()
    created_at = UTCDateTimeAttribute(range_key=True, default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)

class CompanyKnowledge(Model):
    class Meta:
        table_name = 'company_knowledge'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    id = UnicodeAttribute(hash_key=True)
    created_at = UTCDateTimeAttribute(range_key=True, default=datetime.utcnow)
    content = UnicodeAttribute()
    embedding = ListAttribute()
    metadata = MapAttribute()


class Campaign(Model):
    class Meta:
        table_name = 'smsAI_campaigns'
        region = os.getenv("AWS_REGION")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    chatbot_id = UnicodeAttribute()
    leads_list_id = UnicodeAttribute()
    status = UnicodeAttribute()
    schedule = MapAttribute()
    created_at = UnicodeAttribute()
    updated_at = UnicodeAttribute()

def health_check():
    print("!!!!!!!!!!!!!!!!!!!!1 Checking DynamoDB connection...")
    try:
        conn = Connection(
            region=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            
        )
        
        print(conn)
        tables = conn.list_tables().get("TableNames", [])
        for model in [ChatSession, ChatbotScript, CompanyKnowledge,LeadInformation,Chatbot,LeadInformationList,Campaign,SessionTable]:
            if model.Meta.table_name not in tables:
                model.create_table(
                    read_capacity_units=5,
                    write_capacity_units=5,
                    wait=True,
                )
                print(f"Table '{model.Meta.table_name}' was created successfully")
        return {
            "status": "success",
            "message": f"Connection to DynamoDB is successful. Existing tables: {tables}",
        }
    except TableError as e:
        return {"status": "error", "message": f"Table error: {str(e)}"}
    except PynamoDBException as e:
        return {"status": "error", "message": str(e)}
    
    
# uncomment it to check the connection
health_check()
def add_chat_session(item: dict):
    try:
        item_id = item.get("id")
        if not item_id:
            raise ValueError("Missing 'id' in item")

        try:
            record = ChatSession.get(hash_key=item_id)
            existing_messages = record.messages
        except ChatSession.DoesNotExist:
            existing_messages = []

        new_messages = item.get("messages", [])
        updated_messages = existing_messages + new_messages

        record = ChatSession(
            id=item_id,
            user_id=item.get("user_id"),
            title=item.get("title"),
            messages=updated_messages,
            created_at=item.get("created_at", datetime.utcnow()),
            updated_at=datetime.utcnow()
        )
        record.save()
        return {"status": "Item added", "item": item}
    except PynamoDBException as e:
        raise Exception(str(e))
    
# health_check()

def get_chat_session(item_id: str):
    try:
        record = ChatSession.get(hash_key=item_id)
        return {"id": record.id, "user_id": record.user_id, "title": record.title, "messages": record.messages}
    except ChatSession.DoesNotExist:
        raise ValueError("Item not found")
    except PynamoDBException as e:
        raise Exception(str(e))

def update_chat_session(item_id: str, new_data: dict):
    try:
        record = ChatSession.get(hash_key=item_id)
        
        if "user_id" in new_data:
            record.user_id = new_data["user_id"]
        if "title" in new_data:
            record.title = new_data["title"]
        
        new_messages = new_data.get("messages", [])
        record.messages.extend(new_messages)
        
        
        
        
        record.updated_at = datetime.utcnow()
        record.save()
        return {"status": "Item updated", "item": {"id": record.id, "user_id": record.user_id, "title": record.title, "messages": record.messages}}
    except ChatSession.DoesNotExist:
        raise ValueError("Item not found")
    except PynamoDBException as e:
        raise Exception(str(e))







