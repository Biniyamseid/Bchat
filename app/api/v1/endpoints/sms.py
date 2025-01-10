from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from ....schemas.sms import SMSRequest, SMSResponse
from ....services.sms import SMSService
from ....core.security import get_current_user

router = APIRouter()

@router.post("/incoming", response_model=SMSResponse)
async def handle_incoming_sms(
    request: SMSRequest,
    service: SMSService = Depends()
):
    """Handle incoming SMS messages"""
    try:
        return await service.process_incoming_sms(
            phone_number=request.from_number,
            message=request.message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/send", response_model=SMSResponse)
async def send_sms(
    request: SMSRequest,
    current_user = Depends(get_current_user),
    service: SMSService = Depends()
):
    """Send SMS messages"""
    try:
        return await service.send_sms(
            phone_number=request.to_number,
            message=request.message,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat-history/{phone_number}")
async def get_sms_chat_history(
    phone_number: str,
    current_user = Depends(get_current_user),
    service: SMSService = Depends()
):
    """Get SMS chat history for a phone number"""
    try:
        return await service.get_chat_history(phone_number)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 