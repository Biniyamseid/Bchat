from fastapi import APIRouter

from app.api.endpoints import trial
from ..endpoints import chat, script, knowledge,customer,experiment,smschat

api_router = APIRouter()

api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(script.router, prefix="/script", tags=["script"])
api_router.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
api_router.include_router(customer.router, prefix="/customer", tags=["customer"])
api_router.include_router(experiment.router, prefix="/experiment", tags=["experimental"])
api_router.include_router(trial.router, prefix="/trial", tags=["experimental"])
api_router.include_router(smschat.router, prefix="/sms", tags=["sms-chat"])