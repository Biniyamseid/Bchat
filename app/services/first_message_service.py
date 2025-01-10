from typing import Optional
import uuid
from datetime import datetime
from fastapi import HTTPException
from langchain_openai import ChatOpenAI  
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from ..models.dynamodb import ChatSession, Message, ChatbotScript, MessageAttribute
# from ..models.db import health_check, add_item, get_item, update_item,get_or_create_item

from ..core.config import get_settings

settings = get_settings()



class FirstMessageService:
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
    #TODO: add the first message to the message history chain
    async def process_message(
        self,
        session_id: str,
        message: str,
        user_id: Optional[str] = None,
        chatbot_id: Optional[str] = None,
        chatbot_script_id: Optional[str] = None
    ):
        # Create or get session
        try:
            session = next(ChatSession.scan(ChatSession.user_id == user_id), None)
            if not session:
                session = ChatSession(
                    id=str(uuid.uuid4()),
                    user_id=user_id or "anonymous",
                    title="New Chat",
                    messages=[]
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving or creating chat session: {str(e)}")
        
        # Add user message
        user_message = MessageAttribute(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=datetime.utcnow()
        )
        # session.messages.append(user_message)
        
        # Get chatbot script
        # Get chatbot script
        if chatbot_script_id:
            try:
                script = next(ChatbotScript.query(chatbot_script_id), None)
                if not script:
                    raise HTTPException(status_code=404, detail="Chatbot script not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error retrieving chatbot script: {str(e)}")
        else:
            script = next(ChatbotScript.scan(), None)
        if script:
            self.conversation.memory.chat_memory.add_user_message(
                f"based on the following  script please generate a first message for a user mentioned in this chat: {script.content}"
            )
        
        # Generate response
        response = self.conversation.predict(input=message)
        
        # Add assistant message
        assistant_message = MessageAttribute(
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
        

    


