from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import pandas as pd
from io import BytesIO
import uuid
from ...models.dynamodb import ChatSession, Chatbot, LeadInformation, LeadInformationList
Dict = dict

router = APIRouter()

@router.post("/upload-customer-data")
async def upload_customer_data(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        content = await file.read()
        df = pd.read_excel(BytesIO(content))
        
        required_columns = ["Name", "Email", "Phone"]
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"File missing required columns: {', '.join(required_columns)}")
        id = str(uuid.uuid4())
        for _, row in df.iterrows():
            
            customer = LeadInformation(
                id=id,
                name=row["Name"],
                email=row["Email"],
                phone_number=row["Phone"]
            )
            customer.save()
        
        return {"message": "Customer data uploaded successfully",
                'id': id,                                                                                                                                                            
                
                'data': df.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/customer-data", response_model=List[dict])
async def get_customer_data():
    try:
        customers = LeadInformation.scan()
        return [
            {
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
                "phone_number": customer.phone_number,
                "created_at": customer.created_at
            }
            for customer in customers
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   
    
    
    
@router.get("/chatbot/{chatbot_id}/lead-lists", response_model=List[dict])
async def get_lead_lists_by_chatbot(chatbot_id: str):
    """
    Get all lead information lists associated with a chatbot
    """
    try:
        # Get the chatbot
        try:
            chatbot = Chatbot.get(hash_key=chatbot_id)
        except Chatbot.DoesNotExist:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Get all lead lists associated with the chatbot
        lead_lists = []
        if chatbot.lead_information_list_ids:
            for list_id in chatbot.lead_information_list_ids:
                try:
                    lead_list = LeadInformationList.get(hash_key=list_id)
                    lead_lists.append({
                        "id": lead_list.id,
                        "name": lead_list.name,
                        "total_leads": len(lead_list.lead_information_ids),
                        "created_at": lead_list.created_at.isoformat(),
                        "updated_at": lead_list.updated_at.isoformat()
                    })
                except LeadInformationList.DoesNotExist:
                    continue

        return lead_lists

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lead-list/{list_id}/leads", response_model=List[Dict])
async def get_leads_by_list(list_id: str):
    """
    Get all lead information entries in a specific list
    """
    try:
        # Get the lead list
        try:
            lead_list = LeadInformationList.get(hash_key=list_id)
        except LeadInformationList.DoesNotExist:
            raise HTTPException(status_code=404, detail="Lead list not found")

        # Get all leads in the list
        leads = []
        if lead_list.lead_information_ids:
            for lead_id in lead_list.lead_information_ids:
                try:
                    lead = LeadInformation.get(hash_key=lead_id)
                    leads.append({
                        "phone_number": lead.phone_number,
                        "name": lead.name,
                        "email": lead.email,
                        "detail": lead.detail,
                        "chatbot_id": lead.chatbot_id,
                        "created_at": lead.created_at.isoformat(),
                        "updated_at": lead.updated_at.isoformat()
                    })
                except LeadInformation.DoesNotExist:
                    continue

        return leads

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lead/{lead_id}/chat-history", response_model=List[Dict])
async def get_lead_chat_history(lead_id: str):
    """
    Get chat history for a specific lead
    """
    try:
        # Get the chat sessions for the lead
        try:
            sessions = ChatSession.query(
                lead_id,
                scan_index_forward=True  # Get messages in chronological order
            )
            
            chat_history = []
            for session in sessions:
                # Convert messages to dict format
                messages = []
                for msg in session.messages:
                    messages.append({
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat()
                    })
                
                chat_history.append({
                    "session_id": session.user_id,
                    "title": session.title,
                    "created_at": session.created_at.isoformat(),
                    "messages": messages
                })

            return chat_history

        except Exception as e:
            print(f"Error fetching chat history: {str(e)}")
            raise HTTPException(status_code=404, detail="Chat history not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))