from typing import Optional
import uuid
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from ..models.dynamodb import ChatSession, Chatbot, Message, ChatbotScript
# from ..models.db import health_check, add_item, get_item, update_item,get_or_create_item
from ..models.dynamodb import ChatSession, Chatbot, Message, ChatbotScript

from ..core.config import get_settings

settings = get_settings()

class ChatService:
    def __init__(self, chatbot_id: str):
        self.chatbot = Chatbot.get(chatbot_id)
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4-turbo-preview"
        )
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    async def process_message(self, session_id: str, message: str, user_id: Optional[str] = None):
        try:
            session = ChatSession.get(session_id)
        except ChatSession.DoesNotExist:
            session = ChatSession(
                id=session_id or str(uuid.uuid4()),
                user_id=user_id or "anonymous",
                title="New Chat",
                messages=[]
            )
        
        user_message = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=datetime.utcnow()
        )
        session.messages.append(user_message)
        
        self.conversation.memory.chat_memory.add_user_message(
            f"Follow this script for responses: {self.chatbot.script}"
        )
        
        response = self.conversation.predict(input=message)
        
        assistant_message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.utcnow()
        )
        session.messages.append(assistant_message)
        
        session.save()
        
        return {
            "session_id": session.id,
            "response": response
        }


class ChatService2:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4-turbo-preview"
        )
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    async def process_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None
    ):
        # Create or get session
        try:
            session = ChatSession.get(session_id)
        except ChatSession.DoesNotExist:
            session = ChatSession(
                id=session_id or str(uuid.uuid4()),
                user_id=user_id or "anonymous",
                title="New Chat",
                messages=[]
            )
        
        # Add user message
        user_message = Message(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=datetime.utcnow()
        )
        session.messages.append(user_message)
        
        # Get chatbot script
        script = next(ChatbotScript.scan(), None)
        if script:
            self.conversation.memory.chat_memory.add_user_message(
                f"Follow this script for responses: {script.content}"
            )
        
        # Generate response
        response = self.conversation.predict(input=message)
        
        # Add assistant message
        assistant_message = Message(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.utcnow()
        )
        session.messages.append(assistant_message)
        
        # Save session
        session.save()
        
        return {
            "session_id": session.id,
            "response": response
        }
        
    def initialfirstmessagegenerator(customer_data, script_content):
            # Extract necessary details from customer data
            customer_name = customer_data.get("name")
            phone_number = customer_data.get("phone_number")
            
            # Generate the initial message using the script content
            initial_message = f"Hello {customer_name},\n\n{script_content}\n\nLooking forward to assisting you!"
            
            # Return the initial message
            return initial_message
        

    


