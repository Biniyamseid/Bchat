from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import boto3
from datetime import datetime
import uuid
from typing import Optional
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  
from botocore.exceptions import ClientError

router = APIRouter()

# Pydantic models
class MessageRequest(BaseModel):
    user_id: str
    message: str

class MessageResponse(BaseModel):
    response: str

# DynamoDB models
class LeadInformation:
    @staticmethod
    def query(user_id: str, limit: int = 1, scan_index_forward: bool = False):
        """Query lead information from DynamoDB"""
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('LeadInformation')
        response = table.query(
            KeyConditionExpression='user_id = :uid',
            ExpressionAttributeValues={':uid': user_id},
            Limit=limit,
            ScanIndexForward=scan_index_forward
        )
        return iter(response['Items'])

class Chatbot:
    @staticmethod
    def query(chatbot_id: str, limit: int = 1):
        """Query chatbot from DynamoDB"""
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('Chatbot')
        response = table.query(
            KeyConditionExpression='id = :cid',
            ExpressionAttributeValues={':cid': chatbot_id},
            Limit=limit
        )
        return iter(response['Items'])

class ChatbotScript:
    @staticmethod
    def query(script_id: str, limit: int = 1):
        """Query chatbot script from DynamoDB"""
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('ChatbotScript')
        response = table.query(
            KeyConditionExpression='id = :sid',
            ExpressionAttributeValues={':sid': script_id},
            Limit=limit
        )
        return iter(response['Items'])

def ensure_chat_history_table():
    """Ensure DynamoDB table for chat history exists"""
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('ChatMessageHistory')
        table.load()
        return table
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            table = dynamodb.create_table(
                TableName='ChatMessageHistory',
                KeySchema=[
                    {'AttributeName': 'SessionId', 'KeyType': 'HASH'}
                ],
                AttributeDefinitions=[
                    {'AttributeName': 'SessionId', 'AttributeType': 'S'}
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            table.meta.client.get_waiter('table_exists').wait(TableName='ChatMessageHistory')
            return table
        raise

def get_chat_history(user_id: str) -> DynamoDBChatMessageHistory:
    """Get DynamoDB-based chat history for a user"""
    return DynamoDBChatMessageHistory(
        table_name="ChatMessageHistory",
        session_id=user_id
    )

def create_chat_chain(instructions: str):
    """Create a chat chain with the given instructions"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", instructions),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    return prompt | model

@router.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    try:
        # Ensure chat history table exists
        ensure_chat_history_table()
        
        # Step 1: Retrieve lead information
        try:
            lead_information = next(LeadInformation.query(
                request.user_id, limit=1, scan_index_forward=False
            ))
        except StopIteration:
            raise HTTPException(status_code=404, detail="Lead information not found.")
        
        if not lead_information.get('chatbot_id'):
            raise HTTPException(status_code=404, detail="Chatbot ID not found.")
        
        # Step 2: Fetch the chatbot
        try:
            chatbot = next(Chatbot.query(lead_information['chatbot_id'], limit=1))
        except StopIteration:
            raise HTTPException(status_code=404, detail="Chatbot not found.")

        # Step 3: Query the chatbot script
        if not chatbot.get('chatbot_script_id'):
            raise HTTPException(status_code=400, detail="Chatbot script ID is missing.")
        
        try:
            chatbot_script = next(ChatbotScript.query(chatbot['chatbot_script_id'], limit=1))
            if not chatbot_script:
                raise HTTPException(status_code=404, detail="Chatbot script not found.")
        except StopIteration:
            raise HTTPException(status_code=404, detail="Chatbot script not found.")
        
        # Build instructions
        script_info = (
            f"You are a highly intelligent and adaptive chatbot designed to interact with customers based on the following script: {chatbot_script['content']}. "
            f"You are a real estate agent chatbot designed for getting leads and interacting with leads to get potential home sellers. "
            f"Please follow this script for responses. If there are placeholders, you should ask for information from the customer "
            f"and use that information in future interactions. If there is a placeholder for your name like [YourName] or similar, "
            f"please use the chatbot's name. If there is something like [SellersName], use the lead's name. Your name is SMSAI. "
            f"If there is a placeholder for [STREET NAME ONLY] or something similar, use the data presented in the chat; otherwise, "
            f"don't mention a specific place name. Please make sure your response is understandable and grammatically correct."
        )
        
        instructions = chatbot.get('instructions', '') + "\n\n" + script_info
        
        # Get chat history
        chat_history = get_chat_history(request.user_id)
        
        # Add user message to history
        chat_history.add_user_message(request.message)
        
        # Create chat chain and generate response
        chain = create_chat_chain(instructions)
        response = chain.invoke({
            "history": chat_history.messages,
            "input": request.message
        })
        
        # Save AI response to history
        chat_history.add_ai_message(response.content)
        
        return {"response": response.content}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

