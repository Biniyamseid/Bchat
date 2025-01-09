from typing import Optional
from pydantic_settings import BaseSettings
import os
from functools import lru_cache


class Settings(BaseSettings):
    PROJECT_NAME: str = "Agentic Chatbot"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    TEXTGRID_ACCOUNT_SID: str = "KxyPVgnXwhFE48n6dcYCcA=="
    TEXTGRID_AUTH_TOKEN: str = "98D0D330FC624212A1DF514A7A9BA2BA"
    TEXTGRID_PHONE_NUMBER_SID: Optional[str] = None
    WEBHOOK_URL: Optional[str] = "https://519044fb025af0.lhr.life/api/v1/sms/receive-sms"

    
    
    # AWS Configuration
    AWS_REGION: str = "us-east-2"
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    # OpenAI Configuration
    OPENAI_API_KEY:str = os.getenv("OPENAI_API_KEY")
    # Security
    SECRET_KEY: str = "hello"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()


