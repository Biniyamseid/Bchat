from datetime import datetime
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, UTCDateTimeAttribute
from ..core.config import settings

class BaseModel(Model):
    """Base model for all DynamoDB models"""
    class Meta:
        abstract = True
        region = settings.AWS_REGION
        aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY

    created_at = UTCDateTimeAttribute(default=datetime.utcnow)
    updated_at = UTCDateTimeAttribute(default=datetime.utcnow)

    def save(self, **kwargs):
        self.updated_at = datetime.utcnow()
        super().save(**kwargs)

    def to_dict(self):
        """Convert model to dictionary"""
        raise NotImplementedError 