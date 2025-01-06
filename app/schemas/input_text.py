from pydantic import BaseModel

class InputText(BaseModel):
    sms: str