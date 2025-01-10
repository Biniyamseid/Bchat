from pynamodb.attributes import UnicodeAttribute, ListAttribute
from ..base import BaseModel

class LeadInformation(BaseModel):
    class Meta:
        table_name = 'lead_information'
    
    phone_number = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    email = UnicodeAttribute()
    detail = UnicodeAttribute(null=True)
    chatbot_id = UnicodeAttribute(null=True)

    def to_dict(self):
        return {
            "phone_number": self.phone_number,
            "name": self.name,
            "email": self.email,
            "detail": self.detail,
            "chatbot_id": self.chatbot_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class LeadInformationList(BaseModel):
    class Meta:
        table_name = 'lead_information_lists'
    
    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    lead_information_ids = ListAttribute(of=UnicodeAttribute)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "lead_information_ids": list(self.lead_information_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        } 