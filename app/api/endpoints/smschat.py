# from mailbox import Message
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import boto3
from botocore.exceptions import ClientError
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from datetime import timezone
import asyncio
import aioboto3
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime
from app.models.dynamodb import ChatSession, LeadInformation, Chatbot, ChatbotScript 
from pynamodb.models import Model
from pynamodb.attributes import (
    UnicodeAttribute,
    UTCDateTimeAttribute,
    ListAttribute,
    NumberAttribute,
    MapAttribute,
)



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

def get_chat_chain(system_prompt: str, chatbot_script_content: str, lead_name: str):
    try:
        # Create a more detailed prompt template that includes the script
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a highly intelligent and adaptive chatbot designed to interact with customers based on the following script:

{chatbot_script_content}

You are a real estate agent chatbot designed for getting leads and interacting with potential home sellers.
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


@router.post("/chat-with-history3", response_model=ChatResponse)
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
    



