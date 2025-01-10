from pynamodb.attributes import UnicodeAttribute, MapAttribute, NumberAttribute
from ..base import BaseModel

class CampaignMetrics(MapAttribute):
    total_messages = NumberAttribute(default=0)
    responses = NumberAttribute(default=0)
    successful_conversations = NumberAttribute(default=0)
    failed_conversations = NumberAttribute(default=0)
    total_leads = NumberAttribute(default=0)
    processed_leads = NumberAttribute(default=0)

class CampaignSchedule(MapAttribute):
    start_date = UnicodeAttribute()
    end_date = UnicodeAttribute()
    time_zone = UnicodeAttribute()
    daily_start_time = UnicodeAttribute()
    daily_end_time = UnicodeAttribute()

class Campaign(BaseModel):
    class Meta:
        table_name = 'smsAI_campaigns'
    
    id = UnicodeAttribute(hash_key=True)
    name = UnicodeAttribute()
    chatbot_id = UnicodeAttribute()
    leads_list_id = UnicodeAttribute()
    status = UnicodeAttribute()  # active, paused, completed, failed
    schedule = CampaignSchedule(null=True)
    metrics = CampaignMetrics(null=True, default=CampaignMetrics)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'chatbot_id': self.chatbot_id,
            'leads_list_id': self.leads_list_id,
            'status': self.status,
            'schedule': self.schedule.as_dict() if self.schedule else None,
            'metrics': self.metrics.as_dict() if self.metrics else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        } 