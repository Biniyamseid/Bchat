from typing import List, Optional
from pynamodb.attributes import (
    UnicodeAttribute, 
    ListAttribute, 
    NumberAttribute,
    MapAttribute
)
from ..base import BaseModel

class Chatbot(BaseModel):
    class Meta:
        table_name = 'chatbots'
    
    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    chatbot_script_id = UnicodeAttribute()
    lead_information_list_ids = ListAttribute(of=UnicodeAttribute, null=True, default=list)
    instructions = UnicodeAttribute(null=True)
    
    # Personal Details
    first_name = UnicodeAttribute(null=True)
    last_name = UnicodeAttribute(null=True)
    job_title = UnicodeAttribute(null=True)
    personality = UnicodeAttribute(null=True)
    
    # Company Details
    company_name = UnicodeAttribute(null=True)
    company_address = UnicodeAttribute(null=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'lead_information_list_ids': list(self.lead_information_list_ids) if self.lead_information_list_ids else [],
            'chatbot_script_id': self.chatbot_script_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        } 