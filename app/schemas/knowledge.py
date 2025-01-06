from pydantic import BaseModel

class KnowledgeResponse(BaseModel):
    id: str
    message: str