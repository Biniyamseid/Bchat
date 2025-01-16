# from mailbox import Message
import uuid
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from typing import Optional, Dict
import boto3
from botocore.exceptions import ClientError
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  
from datetime import timezone
import asyncio
from fastapi import Request
import requests
import aioboto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
from app.models.dynamodb import ChatSession, LeadInformation, Chatbot, ChatbotScript 
from pynamodb.models import Model
from fastapi import Form, Request, HTTPException
from typing import Optional
from fastapi import Request, Response
import json
import hmac
import hashlib
import base64
from fastapi import Request, Header, HTTPException, Form
from typing import Optional
import requests
from fastapi import HTTPException
import traceback


from pynamodb.attributes import (
    UnicodeAttribute,
    UTCDateTimeAttribute,
    ListAttribute,
    NumberAttribute,
    MapAttribute,
)


import requests
from fastapi import HTTPException
from pydantic import BaseModel

    
from pydantic import BaseModel
import requests
import base64
from fastapi import HTTPException
import os
from dotenv import load_dotenv

load_dotenv()


from datetime import datetime, timedelta
import json
from fastapi import Response, Request

# simple message cache to prevent duplicates
# it Stores message IDs and their timestamps
message_cache = {}
CACHE_EXPIRY_SECONDS = 60  # Messages expire after 60 seconds


TEXTGRID_WEBHOOK_SECRET = "b013cfe83a1e4fcebe2e49bc33eb0e90" 
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

TEXTGRID_ACCOUNT_SID = os.getenv("TEXTGRID_ACCOUNT_SID")
TEXTGRID_AUTH_TOKEN = os.getenv("TEXTGRID_AUTH_TOKEN")
TEXTGRID_PHONE_NUMBER_SID = os.getenv("TEXTGRID_PHONE_NUMBER_SID")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")


router = APIRouter()

# Request and Response Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = "You are a real estate sales agent."
      

class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str

# Utility Functions
def check_or_create_table():
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('SessionTable')
        table.load()
        return table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            table = dynamodb.create_table(
                TableName='SessionTable',
                KeySchema=[
                    {'AttributeName': 'SessionId', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'SessionId', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.meta.client.get_waiter('table_exists').wait(TableName='SessionTable')
            return table
        raise HTTPException(status_code=500, detail=f"DynamoDB error: {str(e)}")

async def check_or_create_table_async():
    session = aioboto3.Session()
    async with session.resource('dynamodb') as dynamodb:
        try:
            table = await dynamodb.Table('SessionTable')
            await table.load()
            return table
        except Exception as e:
            table = await dynamodb.create_table(
                TableName='SessionTable',
                KeySchema=[
                    {'AttributeName': 'SessionId', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'SessionId', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            waiter = dynamodb.get_waiter('table_exists')
            await waiter.wait(TableName='SessionTable')
            return table

def get_chat_chain(system_prompt: str):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = prompt | ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: DynamoDBChatMessageHistory(
                table_name="SessionTable",
                session_id=session_id
            ),
            input_messages_key="question",
            history_messages_key="history"
        )
        
        return chain_with_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up chat chain: {str(e)}")


class Message(MapAttribute):
    id = UnicodeAttribute()
    role = UnicodeAttribute()
    content = UnicodeAttribute()
    timestamp = UTCDateTimeAttribute()
    
    def to_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()  # Convert datetime to string
        }

class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_id: str  # Add this to identify the lead


# Function to send SMS using TextGrid
def send_sms_via_textgrid(to_number: str, message_body: str):
    url = "https://api.textgrid.com/2010-04-01/Accounts/KxyPVgnXwhFE48n6dcYCcA==/Messages.json"
    payload = {
        "body": message_body,
        "from": "+14805001652",  # Your TextGrid number
        # "to": to_number,
        "to": "+1-586-447-7339",
        # "to":"+1 214-356-1277"
        
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "insomnia/10.2.0",
        "Authorization": "Bearer S3h5UFZnblh3aEZFNDhuNmRjWUNjQT09Ojk4RDBEMzMwRkM2MjQyMTJBMURGNTE0QTdBOUJBMkJB"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code not in [201,200]:
        raise HTTPException(status_code=500, detail=f"Failed to send SMS: {response.text}")
    return response.json()

def get_chat_chain(system_prompt: str, chatbot_script_content: str, lead_name: str):
    try:
        # Create a more detailed prompt template that includes the script
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a highly intelligent and adaptive sms chatbot designed to interact with customers  via sms based on the following script:

{chatbot_script_content}

You are a real estate agent sms chatbot designed for getting leads and interacting with potential home sellers.
Please follow this script for responses. If there are placeholders:
- For [YourName]: Use the actual name of the chatbot,
- For [SellersName]: Use '{lead_name}'
- For [STREET NAME]: Only use if provided in the conversation

{system_prompt}
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = prompt | ChatOpenAI(
            model="gpt-4-1106-preview", 
            temperature=0.7
        )

        chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: DynamoDBChatMessageHistory(
                table_name="SessionTable",
                session_id=session_id
            ),
            input_messages_key="question",
            history_messages_key="history"
        )
        
        return chain_with_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up chat chain: {str(e)}")

async def check_or_create_table_async():
    session = aioboto3.Session()
    async with session.resource('dynamodb') as dynamodb:
        try:
            table = await dynamodb.Table('SessionTable')
            await table.load()
            return table
        except Exception as e:
            table = await dynamodb.create_table(
                TableName='SessionTable',
                KeySchema=[
                    {'AttributeName': 'SessionId', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'SessionId', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            waiter = dynamodb.get_waiter('table_exists')
            await waiter.wait(TableName='SessionTable')
            return table


@router.post("/chat-with-history", response_model=ChatResponse)
async def chat_with_histor3(request: ChatRequest):
    try:
        # Step 1: Retrieve the lead information
        try:
            lead_information = LeadInformation.query(
                request.user_id, limit=1, scan_index_forward=False
            ).next()
        except StopIteration:
            raise HTTPException(status_code=404, detail="Lead information not found.")

        if not lead_information.chatbot_id:
            raise HTTPException(status_code=404, detail="Chatbot ID not found.")

        # Step 2: Fetch the chatbot
        try:
            chatbot = Chatbot.get(hash_key=lead_information.chatbot_id)
        except Chatbot.DoesNotExist:
            raise HTTPException(status_code=404, detail="Chatbot not found.")

        # Step 3: Query the chatbot script
        if not chatbot.chatbot_script_id:
            raise HTTPException(status_code=400, detail="Chatbot script ID is missing.")
            
        try:
            script_query = ChatbotScript.query(chatbot.chatbot_script_id, limit=1)
            chatbot_script = next(script_query, None)
            if not chatbot_script:
                raise HTTPException(status_code=404, detail="Chatbot script not found.")
        except StopIteration:
            raise HTTPException(status_code=404, detail="Chatbot script not found.")

        # Ensure DynamoDB table exists
        check_or_create_table()

        # Set up chat history
        history = DynamoDBChatMessageHistory(
            table_name="SessionTable",
            session_id=request.session_id
        )

        # Combine chatbot instructions and script
        system_prompt = chatbot.instructions or ""
        
        # Set up chat chain with script content
        chain_with_history = get_chat_chain(
            system_prompt=system_prompt,
            chatbot_script_content=chatbot_script.content,
            lead_name=lead_information.name
        )

        # Configure session
        config = {"configurable": {"session_id": request.session_id}}

        # Get response from the chain
        response = chain_with_history.invoke(
            {"question": request.message},
            config=config
        )

        # Step 4: Fetch or initialize the user chat session
        try:
            session = ChatSession.query(request.user_id, limit=1, scan_index_forward=False).next()
            print(f"Found existing session for user {request.user_id}")
        except StopIteration:
            print(f"Creating new session for user {request.user_id}")
            session = ChatSession(
                user_id=request.user_id,
                created_at=datetime.utcnow(),
                title="New Chat",
                messages=[]
            )
            # Save the new session immediately
            try:
                session.save()
                print("New session saved successfully")
            except Exception as save_error:
                print(f"Error saving new session: {str(save_error)}")
                raise HTTPException(status_code=500, detail="Failed to create chat session")

        # Add user message
        try:
            user_message = {
                "role": "user",
                "content": request.message
            }
            session.add_message(user_message)
            
            # Get response from the chain
            response = chain_with_history.invoke(
                {"question": request.message},
                config=config
            )

            # Add AI message
            ai_message = {
                "role": "assistant",
                "content": str(response.content) if hasattr(response, 'content') else str(response)
            }
            session.add_message(ai_message)

            # Save session
            try:
                session.save()
                print("Session saved successfully")
            except Exception as save_error:
                print(f"Error saving session: {str(save_error)}")
                print(f"Session data: {session.to_dict()}")  # Debug log
                raise HTTPException(status_code=500, detail="Failed to save chat session")
            
            # Send SMS with the AI response
            response = send_sms_via_textgrid(to_number=request.user_id, message_body=ai_message["content"])
            print(f" !!!!!!!! SMS response: {response}")

            return ChatResponse(
                response=ai_message["content"],
                session_id=request.session_id,
                status="success"
            )
        except Exception as e:
            print(f"Error handling messages: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error handling messages: {str(e)}")
    except Exception as e:
        print(f"Error in chat_with_history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    






def verify_textgrid_signature(signature: str, webhook_url: str, body: str) -> bool:
    """Verify the TextGrid webhook signature."""
    try:
        string_to_sign = webhook_url + body
        secret_key = TEXTGRID_WEBHOOK_SECRET.encode('utf-8')
        
        # Calculate HMAC-SHA1
        hmac_obj = hmac.new(secret_key, string_to_sign.encode('utf-8'), hashlib.sha1)
        calculated_signature = base64.b64encode(hmac_obj.digest()).decode('utf-8')
        
        return hmac.compare_digest(calculated_signature, signature)
    except Exception as e:
        print(f"Signature verification error: {str(e)}")
        return False



def format_phone_number(phone_number: str) -> str:
    """
    Format phone number from +15864477339 to +1-586-447-7339
    """
    try:
        # Remove any existing formatting
        digits = ''.join(filter(str.isdigit, phone_number))
        
        if len(digits) == 11 and digits.startswith('1'):
            # Format: +1-XXX-XXX-XXXX
            return f"+{digits[0]}-{digits[1:4]}-{digits[4:7]}-{digits[7:]}"
        elif len(digits) == 10:
            # Add +1 if it's missing and format
            return f"+1-{digits[0:3]}-{digits[3:6]}-{digits[6:]}"
        else:
            # Return original if format is unknown
            return phone_number
    except Exception as e:
        print(f"Error formatting phone number: {str(e)}")
        return phone_number



def is_duplicate_message(message_sid: str) -> bool:
    """
    Check if a message has been recently processed
    """
    now = datetime.utcnow()
    
    # Clean expired entries
    expired = [sid for sid, timestamp in message_cache.items() 
              if (now - timestamp) > timedelta(seconds=CACHE_EXPIRY_SECONDS)]
    for sid in expired:
        message_cache.pop(sid)
    
    # Check if message is in cache
    if message_sid in message_cache:
        return True
    
    # Add new message to cache
    message_cache[message_sid] = now
    return False

@router.post("/receive-sms")
async def receive_sms(request: Request):
    try:
        # Get the raw form data
        form_data = await request.form()
        webhook_data = dict(form_data)
        
        # Log all received data
        print("Received webhook data:")
        print(json.dumps(webhook_data, indent=2))
        
        # Check for duplicate message
        message_sid = webhook_data.get('MessageSid')
        if message_sid and is_duplicate_message(message_sid):
            print(f"Duplicate message detected: {message_sid}")
            return Response(
                content="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response/>",
                media_type="application/xml"
            )
        
        # Extract the important fields if they exist
        from_number = webhook_data.get('From', webhook_data.get('from', ''))
        message_body = webhook_data.get('Body', webhook_data.get('body', ''))
        formatted_number = format_phone_number(from_number)
        
        print(f"Processing new message from: {from_number}")
        print(f"Message: {message_body}")

        if message_body and from_number:
            chat_request = ChatRequest(
                session_id=formatted_number,
                message=message_body,
                user_id=formatted_number
            )

            response = await chat_with_histor3(chat_request)

            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
                <Response>
                    <Message>{response.response}</Message>
                </Response>"""

            return Response(
                content=xml_response,
                media_type="application/xml"
            )
        
        return Response(
            content="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response/>",
            media_type="application/xml"
        )

    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        print(f"Error type: {type(e)}")
        
        print(f"Traceback: {traceback.format_exc()}")
        
        return Response(
            content="<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response/>",
            media_type="application/xml"
        )


def verify_textgrid_signature(signature: str, url: str, params: dict) -> bool:
    return True



class WebhookConfig(BaseModel):
    account_sid: str
    phone_number_sid: str
    webhook_url: str = os.getenv("WEBHOOK_URL")
    
    


def test_credentials(account_sid: str, auth_token: str):
    url = f"https://api.textgrid.com/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers.json"
    auth_string = f"{account_sid}:{auth_token}"
    auth_encoded = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"Basic {auth_encoded}"
    }
    
    response = requests.get(url, headers=headers)
    print(f"Test Response: {response.status_code} - {response.text}")
    return response

def update_phone_number_webhook(
    account_sid: str,
    phone_number_sid: str,
    webhook_url: str,
    auth_token: str
) -> dict:
    """
    Update a TextGrid phone number's webhook settings
    """
    url = f"https://api.textgrid.com/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers/{phone_number_sid}.json"
    
    # Form-encoded parameters as specified in the API docs
    payload = {
        "FriendlyName": "My Webhook Configuration",  # Optional
        "SmsUrl": webhook_url,
        "SmsMethod": "POST",
        "StatusCallback": "",
        "StatusCallbackMethod": "POST"
    }
    
    # Create the authorization string and encode it
    auth_string = f"{account_sid}:{auth_token}"
    auth_encoded = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {auth_encoded}"  # Changed to Basic auth
    }

    try:
        # Use data parameter for form-encoded data
        response = requests.post(url, data=payload, headers=headers)
        
        # Log the request details for debugging
        print(f"Request URL: {url}")
        print(f"Request Headers: {headers}")
        print(f"Request Payload: {payload}")
        print(f"Response Status: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail=f"Unauthorized: Invalid credentials - {response.text}")
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to update phone number: {response.text}"
            )
        
        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating phone number: {str(e)}")

@router.post("/configure-webhook")
async def configure_webhook(config: WebhookConfig):
    """
    Configure the webhook URL for a TextGrid phone number
    """
    try:
        # Use the correct credentials
        result = update_phone_number_webhook(
            account_sid="KxyPVgnXwhFE48n6dcYCcA==",  
            phone_number_sid="f~jDHOvUT71qyyXQlvvZpA==", 
            webhook_url=os.getenv("WEBHOOK_URL"),
            auth_token="98D0D330FC624212A1DF514A7A9BA2BA" 
        )
        
        return {
            "status": "success",
            "message": "Webhook configured successfully",
            "details": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def list_phone_numbers(account_sid: str, auth_token: str) -> dict:
    """
    List all phone numbers associated with your TextGrid account
    """
    url = f"https://api.textgrid.com/2010-04-01/Accounts/{account_sid}/IncomingPhoneNumbers.json"
    
    # Create the authorization header
    auth_string = f"{account_sid}:{auth_token}"
    auth_encoded = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"Bearer {auth_encoded}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid credentials")
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to fetch phone numbers: {response.text}"
            )
        
        return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching phone numbers: {str(e)}")

@router.get("/list-phone-numbers")
async def get_phone_numbers():
    """
    Endpoint to list all phone numbers and their SIDs
    """
    # Replace these with your actual TextGrid credentials
    ACCOUNT_SID = "KxyPVgnXwhFE48n6dcYCcA=="
    AUTH_TOKEN = "98D0D330FC624212A1DF514A7A9BA2BA"
    
    try:
        result = list_phone_numbers(ACCOUNT_SID, AUTH_TOKEN)
        
        # Format the response to show phone numbers and their SIDs
        phone_numbers = []
        for number in result.get('incoming_phone_numbers', []):
            phone_numbers.append({
                'phone_number': number.get('phone_number'),
                'sid': number.get('sid'),
                'friendly_name': number.get('friendly_name')
            })
        
        return {
            "status": "success",
            "phone_numbers": phone_numbers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
