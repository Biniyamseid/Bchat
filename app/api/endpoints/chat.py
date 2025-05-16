from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime, time

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pynamodb.exceptions import PynamoDBException
from typing import List
from langchain_core.tools import tool
from app.models.db import get_or_create_item
from app.services import first_message_service
# from app.services.chatagent_service import ChatAgentService
# from app.services.agentic_service import ChatAgentService3
from ...core.config import get_settings
from ...models.dynamodb import Campaign, ChatSession, Chatbot, ChatbotScript, LeadInformation, Message, SessionTable
from ...services.chat_service import ChatService
from ...schemas.chat import ChatResponse
from pydantic import BaseModel
from typing import List
from ...models.dynamodb import health_check, add_chat_session, get_chat_session, update_chat_session
# from ...models.db import health_check, add_item, get_item, update_item,get_or_create_item
from fastapi import HTTPException
from datetime import datetime

from dotenv import load_dotenv


from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
import getpass
import os
import boto3

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from ...schemas.input_text import InputText


from ...services.sms_analyzer_service import SMSAnalyzerService
import uuid
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from datetime import datetime
from typing import Literal, List, Dict
# from langgraph.graph import END
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph, START
# from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI         

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
# from openpyxl import load_workbook
from datetime import datetime
import uuid
from app.models.dynamodb import LeadInformation, LeadInformationList
  
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uuid
import os
from app.models.dynamodb import Chatbot, ChatbotScript, LeadInformationList
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

from pydantic import BaseModel
from typing import List
from datetime import datetime

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute, ListAttribute
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
from app.models.dynamodb import ChatbotScript


from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from datetime import datetime
import uuid
from app.models.dynamodb import ChatbotScript
from app.services.pdf_service import PDFService
from typing import Optional
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

from fastapi import UploadFile, File, HTTPException
import pandas as pd
import boto3

import warnings
from langchain.globals import set_debug, set_verbose

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Disable langchain debugging and verbose output
set_debug(False)
set_verbose(False)

load_dotenv()



router = APIRouter()
settings = get_settings()




# Health check and table creation

def initialize_table():
    if not SessionTable.exists():
        SessionTable.create_table(
            read_capacity_units=5,
            write_capacity_units=5,
            wait=True
        )
        print("SessionTable created successfully.")
        


# AWS Configuration
os.environ["AWS_DEFAULT_REGION"] = "us-east-2"
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Initialize FastAPI and Router
app = FastAPI()
router = APIRouter()




# Initialize the DynamoDB Table
def initialize_table():
    if not SessionTable.exists():
        SessionTable.create_table(
            read_capacity_units=5,
            write_capacity_units=5,
            wait=True
        )
        print("SessionTable created successfully.")


# Request schema for chat endpoint
class ChatRequest(BaseModel):
    user_id: str
    message: str


# Configure OpenAI Chat
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
chat_chain = prompt | chat_model

chain_with_history = RunnableWithMessageHistory(
    chat_chain,
    lambda session_id: DynamoDBChatMessageHistory(
        table_name="SessionTable", session_id=session_id
    ),
    input_messages_key="question",
    history_messages_key="history",
)





# Chat history endpoint
@router.get("/history/{user_id}")
async def get_history(user_id: str):
    try:
        # Retrieve chat history for the given user
        history = DynamoDBChatMessageHistory(table_name="SessionTable", session_id=user_id)
        messages = [{"role": "user" if msg.role == "human" else "ai", "content": msg.content} for msg in history.messages]

        return {"history": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






# Secret key for JWT
SECRET_KEY = "Rei"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30





# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Simulated admin credentials
admin_user = {"username": "admin", "hashed_password": pwd_context.hash("password")}



def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != admin_user["username"] or not verify_password(form_data.password, admin_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/admin-dashboard")
async def admin_dashboard(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"message": f"Welcome {payload['sub']} to the Admin Dashboard!"}
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")
    
    




@router.post("/upload-customer-data/")
async def upload_customer_data(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into memory
        content = await file.read()  # Reads the file's content as bytes

        # Load the Excel file using pandas from in-memory bytes
        df = pd.read_excel(BytesIO(content))  # Use BytesIO to handle file content

        # Validation logic: Check for required columns
        required_columns = ["Name", "Phone", "Email"]
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400, detail=f"File missing required columns: {', '.join(required_columns)}"
            )
        items = []
        # Save data to DynamoDB
        for _, row in df.iterrows():
            item = {
                "id": str(uuid.uuid4()),  # Generate a unique ID for each customer
                "data": {
                    "name": row["Name"],
                    "phone_number": row["Phone"],
                    "email": row["Email"],
                    "messages": []
                }
            }
            add_item(item)  # Add the item to the database
            items.append(item)

        return {"message": "Customer data uploaded successfully", "record_count": len(df),"db":items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/customer/{item_id}")
async def get_customer(item_id: str):
    try:
        customer = get_item(item_id)  # Retrieve customer data from DynamoDB
        return customer
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error fetching customer: {str(e)}")

from fastapi import HTTPException
from datetime import datetime
from PyPDF2 import PdfReader
import uuid

@router.post("/upload-chatbot-script/")
async def upload_chatbot_script(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()

        # Check the file type by the extension or content type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension == "json":
            # Process JSON files
            if not content.startswith(b"{") or not content.endswith(b"}"):
                raise HTTPException(status_code=400, detail="Invalid JSON format")
            script_content = content.decode("utf-8")
        elif file_extension == "pdf":
            # Process PDF files
            try:
                pdf_reader = PdfReader(BytesIO(content))
                script_content = " ".join(page.extract_text() for page in pdf_reader.pages)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading PDF file: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only JSON and PDF are allowed.")
        item = {
            "id": str(uuid.uuid4()),
            "data": {
                "content": script_content,
                "uploaded_at": str(datetime.utcnow())
            }
        }
        # Save to DynamoDB
        add_item(item)

        return {"message": "Chatbot script uploaded successfully","item":item, "file_type": file_extension}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")




@router.get("/chatbot-script/{script_id}")
async def get_chatbot_script(script_id: str):
    try:
        # Fetch the script from the database
        script = await get_item(script_id)
        # return script
        if not script:
            raise HTTPException(status_code=404, detail="Chatbot script not found")

        
        return {
            "script_id": script_id,
            "content": script.get("data", {}).get("content"),
            "uploaded_at": script.get("data", {}).get("uploadedAt")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching script: {str(e)}")
    
    
@router.post("/choose-chatbot-script/{script_id}")
async def get_chatbot_script(script_id: str):
    template = """You are a highly intelligent and adaptive chatbot designed to interact with customers based on specific instructions, response guidelines, and contextual understanding. Before engaging in any conversation, ensure that you adhere to the following guidelines and dynamically adjust your responses accordingly:

 response to align with the instructions, the customer's sentiment, and the overall context of the conversation.

### Process:
1. Begin by receiving specific instructions or guidelines for the session.


### Output Requirements:
- Return your responses as structured JSON objects, where applicable, to ensure clarity and ease of processing.
- Example JSON Response:
```json
{
  "response": "Hey {Seller Name} this is {Your Name} I'm calling back about the property on {Street
Name}. I'd like to get caught up to speed so I can have a better understanding of what
you're looking to accomplish and where you are in the process.",
  "context": {
    "key_details": {
      "customer_name": "John Doe",
      "phone_number": "123-456-7890",
      "account_status": "Active"
    }
  }
}
in the following command the basic and fundamental rules and guidelines are set you must follow those,
"""
    try:
        # Fetch the script from the database
        script = get_item(script_id)
        # return script
        if not script:
            raise HTTPException(status_code=404, detail="Chatbot script not found")
        content = script.get("data", {}).get("content")
        template += f"""this is the instruction for chat bot it include all necessary how to operate instructions {content}"""

        
        return {
            "template": template,
            
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching script: {str(e)}")
    



# todo here the initiation message should be saved as the first message for the chatbot with that specific customer id 
@router.post("/send-initiation-messages/")
async def send_initiation_messages():
    customers = table.scan()["Items"]  # Retrieve customer data from DynamoDB
    failed_sends = []

    for customer in customers:
        pass

    return {"message": "Messages sent"}


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")



# Simulated workflow and database (replace with real LangChain and DynamoDB code)
cached_responses = []

# Request schema
class MessageRequest(BaseModel):
    user_id: str
    message: str

# Response schema
class MessageResponse(BaseModel):
    response: str

dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('db')





sessions: Dict[str, dict] = {}


def store_chat_history(user_id: str, chat_history):
    """Store chat history in DynamoDB."""
    timestamp = str(datetime.utcnow().timestamp())
    table.put_item(
        Item={
            'user_id': user_id,
            'timestamp': timestamp,
            'chat_history': [msg.dict() for msg in chat_history]
        }
    )
    

def retrieve_chat_history(user_id: str):
    """Retrieve chat history from DynamoDB."""
    response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id)
    )
    return response['Items']


@router.post("/analyze_sms", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """
    Handle user messages and maintain context.
    """
    try:
        messages = get_or_create_item(request.user_id)
        print("!!!!!!!!!!!!!!!!!!!!!!", messages)
        if messages:
            sms_messages = messages['data']['messages']
            formatted_messages = [{"role": "user", "content": msg} for msg in sms_messages]
            
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!sms-messages", sms_messages)
        print("!!!!!!!!!!!!!!!!!!!!!!", messages)
        
        # Ensure session exists for the user
        if request.user_id not in sessions:
            sessions[request.user_id] = {"config": {"configurable": {"thread_id": str(request.user_id)}}, "messages": formatted_messages}

        # Add the new message to the session
        user_message = HumanMessage(content=request.message)
        sessions[request.user_id]["messages"].append(user_message)
        
        item = {
            "messages":[request.message]
        }
        
        # save sms message to the database
        _ = await update_item(request.user_id,item)

        # Process the message using the LangChain graph
        output = None
        for output in graph.stream(
            {"messages": sessions[request.user_id]["messages"]},
            config=sessions[request.user_id]["config"],
            stream_mode="updates"
        ):
            if output:
                last_message = next(iter(output.values()))["messages"][-1]
                sessions[request.user_id]["messages"].append(last_message)
                return {"response": last_message.content}
        raise Exception("No output generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
                                                                                               
                                                                                                                 
                                                                                                                 

def get_chat_sessions_for_user(user_id: str, start_date: datetime, end_date: datetime):
    try:
        sessions = ChatSession.query(
            user_id,
            ChatSession.created_at.between(start_date, end_date)
        )
        return [session for session in sessions]
    except PynamoDBException as e:
        raise Exception(str(e))
    


def generate_initial_message(customer_data, script_content):
    # Extract necessary details from customer data
    customer_name = customer_data.get("name")
    phone_number = customer_data.get("phone")
    
    
    
    # Generate the initial message using the script content
    return f"Hello {customer_name}, welcome to our service! {script_content}"
@router.post("/start-conversation", response_model=List[MessageResponse])
async def start_conversation(chat_service: ChatService = Depends()):
    """
    Start a conversation with each customer.
    """
    try:
        customer_data = LeadInformation.scan()
        responses = []
        first_message_service1 = first_message_service.FirstMessageService()

        
        for customer in customer_data:
            response = await first_message_service1.process_message(
                session_id=customer.phone_number,  
                message=f"please generate a first message for a user mentioned in this chat {customer},if there is a place holder for your name like [YourName] or similar please use SMSAI, your name is SMSAI and  if there is a place holder like  [STREET NAME ONLY] or something similar, if the data is presented on the chat use that data else don't mention specific place name, please don't reply the place holder as it is.",
                user_id=customer.phone_number
            )
            responses.append(response)
        
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class ChatbotCreateRequest(BaseModel):
    name: str
    chatbot_script_id: str
    lead_information_list_ids: List[str]




class ChatbotResponse(BaseModel):
    id: str
    name: str
    chatbot_script_id: str
    lead_information_list_ids: List[str]
    created_at: datetime
    updated_at: datetime
    instructions: Optional[str] = Field(None, description="Instructions dynamically generated for the chatbot")
    first_name: Optional[str] = Field(None, description="The first name of the chatbot")
    last_name: Optional[str] = Field(None, description="The last name of the chatbot")
    job_title: Optional[str] = Field(None, description="The job title associated with the chatbot")
    personality: Optional[str] = Field(None, description="The personality of the chatbot")
    company_name: Optional[str] = Field(None, description="The company name")
    company_address: Optional[str] = Field(None, description="The company address")
    department: Optional[str] = Field(None, description="The department associated with the chatbot")
    industry: Optional[str] = Field(None, description="The industry of the company")
    year_established: Optional[int] = Field(None, description="The year the company was established")
    company_email: Optional[str] = Field(None, description="The company email")
    phone_number: Optional[str] = Field(None, description="The company phone number")
    company_website: Optional[str] = Field(None, description="The company website")
    job_description: Optional[str] = Field(None, description="A description of the chatbot's job role")
    priorities: Optional[List[str]] = Field(None, description="The chatbot's priorities")
    opinions: Optional[List[str]] = Field(None, description="The chatbot's opinions or notable points")
    ai_creativity: Optional[int] = Field(None, description="The AI creativity level (e.g., from 1-10)")
    scorecard: Optional[dict] = Field(None, description="Scorecard for the chatbot performance metrics")

class ChatbotScriptResponse(BaseModel):
    id: str
    content: str
    created_at: datetime
    updated_at: datetime



class ChatbotScriptResponse(BaseModel):
    id: str
    content: str
    created_at: datetime
    updated_at: datetime






class ChatbotScriptResponse(BaseModel):
    id: str
    content: str
    created_at: datetime
    updated_at: datetime

@router.post("/upload_chatbot_script", response_model=ChatbotScriptResponse)
async def upload_chatbot_script(file: UploadFile = File(...)):
    try:
        pdf_service = PDFService()
        parsed_text = await pdf_service.extract_text(file)
        
        script_id = str(uuid.uuid4())

        chatbot_script = ChatbotScript(
            id=script_id,
            content=parsed_text,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        chatbot_script.save()

        return chatbot_script
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/chatbot_script/{id}", response_model=ChatbotScriptResponse)
async def get_chatbot_script(id: str, created_at: Optional[datetime] = None):
    try:
        if created_at:
            chatbot_script = ChatbotScript.get(hash_key=id, range_key=created_at)
        else:
            # Query for the latest entry with the given id
            scripts = ChatbotScript.query(id, limit=1, scan_index_forward=False)
            chatbot_script = next(scripts, None)
            if chatbot_script is None:
                raise ChatbotScript.DoesNotExist

        return ChatbotScriptResponse(
            id=chatbot_script.id,
            content=chatbot_script.content,
            created_at=chatbot_script.created_at,
            updated_at=chatbot_script.updated_at
        )
    except ChatbotScript.DoesNotExist:
        raise HTTPException(status_code=404, detail="ChatbotScript not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chatbot/{chatbot_id}/verify-update")
async def verify_chatbot_update(chatbot_id: str, lead_list_id: str):
    """
    Manually verify/update chatbot lead lists
    """
    try:
        chatbot = Chatbot.get(chatbot_id)
        current_lists = chatbot.lead_information_list_ids or []
        
        if lead_list_id not in current_lists:
            current_lists.append(lead_list_id)
            
            # Try direct update
            chatbot.lead_information_list_ids = current_lists
            chatbot.save()
            
            # Verify
            updated_chatbot = Chatbot.get(chatbot_id)
            
            return {
                "original_state": chatbot.to_dict(),
                "updated_state": updated_chatbot.to_dict(),
                "success": lead_list_id in (updated_chatbot.lead_information_list_ids or [])
            }
        
        return {"message": "Lead list already present", "current_lists": current_lists}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@router.get("/chatbot/{chatbot_id}/details")
async def get_chatbot_details(chatbot_id: str):
    """
    Get chatbot details including lead lists
    """
    try:
        try:
            chatbot = Chatbot.get(hash_key=chatbot_id)
            print(f"Raw chatbot data: {chatbot.to_dict()}")
            
            # Ensure we're getting the list properly
            lead_lists = chatbot.lead_information_list_ids or []
            print(f"Lead lists from chatbot: {lead_lists}")
            
            return {
                "id": chatbot.id,
                "name": chatbot.name,
                "lead_information_list_ids": lead_lists,
                "created_at": chatbot.created_at.isoformat() if hasattr(chatbot, 'created_at') else None,
                "updated_at": chatbot.updated_at.isoformat() if hasattr(chatbot, 'updated_at') else None,
                "chatbot_script_id": chatbot.chatbot_script_id if hasattr(chatbot, 'chatbot_script_id') else None,
                "raw_data": chatbot.to_dict()  # Include raw data for debugging
            }
            
        except Chatbot.DoesNotExist:
            raise HTTPException(status_code=404, detail="Chatbot not found")
            
    except Exception as e:
        print(f"Error getting chatbot details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create_chatbot", response_model=ChatbotResponse)
async def create_chatbot(request: ChatbotCreateRequest):
    try:
        chatbot = Chatbot(
            id=str(uuid.uuid4()),
            name=request.name,
            chatbot_script_id=request.chatbot_script_id,
            lead_information_list_ids=request.lead_information_list_ids,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        chatbot.save()

        return chatbot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chatbot/{id}", response_model=ChatbotResponse)
async def get_chatbot(id: str):
    try:
        chatbot = Chatbot.get(hash_key=id)
        return chatbot
    except Chatbot.DoesNotExist:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
class LeadInformationResponse(BaseModel):
    phone_number: str
    name: str
    email: str
    detail: Optional[str] = None
    chatbot_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class LeadInformationListResponse(BaseModel):
    id: str
    name: str
    lead_information_ids: List[str]
    created_at: datetime
    updated_at: datetime
    leads: List[LeadInformationResponse] = []

@router.post("/upload_leads", response_model=LeadInformationListResponse)
async def upload_leads(name: str, file: UploadFile = File(...)):
    try:
        # Load the Excel file
        workbook = load_workbook(file.file)
        sheet = workbook.active

        lead_information_ids = []

        # Parse the Excel file
        for row in sheet.iter_rows(min_row=2, values_only=True):  # Skip header row
            name, phone_number, email, detail = row
            if name and phone_number and email and detail:
                lead_info = LeadInformation(
                    phone_number=phone_number,
                    name=name,
                    email=email,
                    detail=detail,
                    chatbot_id="",  # Set chatbot_id if needed
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                lead_info.save()
                lead_information_ids.append(lead_info.phone_number)

        # Create LeadInformationList entry
        lead_info_list = LeadInformationList(
            id=str(uuid.uuid4()),
            name=name,
            lead_information_ids=lead_information_ids,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        lead_info_list.save()

        return lead_info_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    


@router.get("/lead_information_list/{id}", response_model=LeadInformationListResponse)
async def get_lead_information_list(id: str):
    try:
        lead_info_list = LeadInformationList.get(hash_key=id)
        return lead_info_list
    except LeadInformationList.DoesNotExist:
        raise HTTPException(status_code=404, detail="LeadInformationList not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class ChatbotCreateRequest(BaseModel):
    name: str
    chatbot_script_id: str
    lead_information_list_ids: List[str]
    first_name: Optional[str] = Field(None, description="The first name of the chatbot")
    last_name: Optional[str] = Field(None, description="The last name of the chatbot")
    job_title: Optional[str] = Field(None, description="The job title associated with the chatbot")
    personality: Optional[str] = Field(None, description="The personality of the chatbot")
    company_name: Optional[str] = Field(None, description="The company name")
    company_address: Optional[str] = Field(None, description="The company address")
    department: Optional[str] = Field(None, description="The department associated with the chatbot")
    industry: Optional[str] = Field(None, description="The industry of the company")
    year_established: Optional[int] = Field(None, description="The year the company was established")
    company_email: Optional[str] = Field(None, description="The company email")
    phone_number: Optional[str] = Field(None, description="The company phone number")
    company_website: Optional[str] = Field(None, description="The company website")
    job_description: Optional[str] = Field(None, description="A description of the chatbot's job role")
    priorities: Optional[List[str]] = Field(None, description="The chatbot's priorities")
    opinions: Optional[List[str]] = Field(None, description="The chatbot's opinions or notable points")
    ai_creativity: Optional[int] = Field(None, description="The AI creativity level (e.g., from 1-10)")
    scorecard: Optional[dict] = Field(None, description="Scorecard for the chatbot performance metrics")
    
# from app.services.chatbot_service import get_chatbot_script
from typing import Optional
from datetime import datetime
from app.models.dynamodb import ChatbotScript
from fastapi import HTTPException

async def get_chatbot_script(id: str, created_at: Optional[datetime] = None) -> ChatbotScript:
    try:
        if created_at:
            chatbot_script = ChatbotScript.get(hash_key=id, range_key=created_at)
        else:
            # Query for the latest entry with the given id
            scripts = ChatbotScript.query(id, limit=1, scan_index_forward=False)
            chatbot_script = next(scripts, None)
            if chatbot_script is None:
                raise ChatbotScript.DoesNotExist
        return chatbot_script
    except ChatbotScript.DoesNotExist:
        raise HTTPException(status_code=404, detail="Script not found")

@router.post("/create-chatbot2", response_model=ChatbotResponse)
async def create_chatbot(request: ChatbotCreateRequest):
    try:
        # Retrieve the chatbot script using the helper function
        script = await get_chatbot_script(request.script_id)

        # Create a new Chatbot using the existing script
        chatbot = Chatbot(
            id=str(uuid.uuid4()),
            name=request.name,
            chatbot_script_id=script.id,
            lead_information_list_ids=request.lead_information_list_ids,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        chatbot.save()

        # Ensure all fields are correctly populated
        return ChatbotResponse(
            id=chatbot.id,
            name=chatbot.name,
            chatbot_script_id=chatbot.chatbot_script_id,
            lead_information_list_ids=chatbot.lead_information_list_ids,
            created_at=chatbot.created_at,
            updated_at=chatbot.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import HTTPException

from fastapi import Body

@router.post("/create-chatbot4", response_model=ChatbotResponse)
async def create_chatbot4(request: ChatbotCreateRequest = Body(...)):
    try:
        # Retrieve the chatbot script using the helper function
        script = await get_chatbot_script(request.chatbot_script_id)
        if not script:
            raise HTTPException(
                status_code=404, detail=f"Chatbot script with ID {request.chatbot_script_id} not found"
            )

        # Generate the `instructions` field dynamically
        instructions = (
            "You are a highly intelligent and adaptive chatbot designed to interact with customers "
            "based on the following script. Please follow this script for responses. If there are placeholders, "
            "ask the customer for information and use it in future interactions. For placeholders like "
            "[YourName], your name is SMSAI. For placeholders like [STREET NAME ONLY], use the data if provided; "
            "otherwise, avoid specific place names. Ensure responses are understandable and grammatically correct."
            f"\n\nScript Content:\n{script.content}"
        )

        # Create a new Chatbot using the existing script and additional details
        chatbot = Chatbot(
            id=str(uuid.uuid4()),
            name=request.name,
            chatbot_script_id=script.id,
            lead_information_list_ids=request.lead_information_list_ids,
            instructions=instructions,  # Save the generated instructions
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            first_name=request.first_name,
            last_name=request.last_name,
            job_title=request.job_title,
            personality=request.personality,
            company_name=request.company_name,
            company_address=request.company_address,
            department=request.department,
            industry=request.industry,
            year_established=request.year_established,
            company_email=request.company_email,
            phone_number=request.phone_number,
            company_website=request.company_website,
            job_description=request.job_description,
            priorities=request.priorities,
            opinions=request.opinions,
            ai_creativity=request.ai_creativity,
            scorecard=request.scorecard
        )
        chatbot.save()

        # Return a response with all fields included
        return ChatbotResponse(
            id=chatbot.id,
            name=chatbot.name,
            chatbot_script_id=chatbot.chatbot_script_id,
            lead_information_list_ids=chatbot.lead_information_list_ids,
            instructions=chatbot.instructions,
            created_at=chatbot.created_at,
            updated_at=chatbot.updated_at,
            first_name=chatbot.first_name,
            last_name=chatbot.last_name,
            job_title=chatbot.job_title,
            personality=chatbot.personality,
            company_name=chatbot.company_name,
            company_address=chatbot.company_address,
            department=chatbot.department,
            industry=chatbot.industry,
            year_established=chatbot.year_established,
            company_email=chatbot.company_email,
            phone_number=chatbot.phone_number,
            company_website=chatbot.company_website,
            job_description=chatbot.job_description,
            priorities=chatbot.priorities,
            opinions=chatbot.opinions,
            ai_creativity=chatbot.ai_creativity,
            scorecard=chatbot.scorecard
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

                                                             
@router.get("/get-all-chatbots", response_model=List[ChatbotResponse])
async def get_all_chatbots():
    try:
        # Scan the entire Chatbot table to retrieve all entries
        chatbots = []
        for chatbot in Chatbot.scan():
            chatbots.append(
                ChatbotResponse(
                    id=chatbot.id,
                    name=chatbot.name,
                    chatbot_script_id=chatbot.chatbot_script_id,
                    lead_information_list_ids=chatbot.lead_information_list_ids,
                    instructions=chatbot.instructions,
                    created_at=chatbot.created_at,
                    updated_at=chatbot.updated_at,
                    first_name=chatbot.first_name,
                    last_name=chatbot.last_name,
                    job_title=chatbot.job_title,
                    personality=chatbot.personality,
                    company_name=chatbot.company_name,
                    company_address=chatbot.company_address,
                    department=chatbot.department,
                    industry=chatbot.industry,
                    year_established=chatbot.year_established,
                    company_email=chatbot.company_email,
                    phone_number=chatbot.phone_number,
                    company_website=chatbot.company_website,
                    job_description=chatbot.job_description,
                    priorities=chatbot.priorities,
                    opinions=chatbot.opinions,
                    ai_creativity=chatbot.ai_creativity,
                    scorecard=chatbot.scorecard,
                )
            )

        # Return all chatbots
        return chatbots

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-chatbot4/{chatbot_id}", response_model=ChatbotResponse)
async def get_chatbot4(chatbot_id: str):
    try:
        # Retrieve the chatbot using its ID
        try:
            chatbot = Chatbot.get(hash_key=chatbot_id)
        except Chatbot.DoesNotExist:
            raise HTTPException(
                status_code=404, detail=f"Chatbot with ID {chatbot_id} not found"
            )

        # Return the chatbot details in the response
        return ChatbotResponse(
            id=chatbot.id,
            name=chatbot.name,
            chatbot_script_id=chatbot.chatbot_script_id,
            lead_information_list_ids=chatbot.lead_information_list_ids,
            instructions=chatbot.instructions,
            created_at=chatbot.created_at,
            updated_at=chatbot.updated_at,
            first_name=chatbot.first_name,
            last_name=chatbot.last_name,
            job_title=chatbot.job_title,
            personality=chatbot.personality,
            company_name=chatbot.company_name,
            company_address=chatbot.company_address,
            department=chatbot.department,
            industry=chatbot.industry,
            year_established=chatbot.year_established,
            company_email=chatbot.company_email,
            phone_number=chatbot.phone_number,
            company_website=chatbot.company_website,
            job_description=chatbot.job_description,
            priorities=chatbot.priorities,
            opinions=chatbot.opinions,
            ai_creativity=chatbot.ai_creativity,
            scorecard=chatbot.scorecard,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/chat_stream/{user_id}")
async def chat_stream(websocket: WebSocket, user_id: str):
    await websocket.accept()
    messages = get_or_create_item(user_id)
    if messages:
            sms_messages = messages['data']['messages']
            formatted_messages = [{"role": "user", "content": msg} for msg in sms_messages]

    if user_id not in sessions:
        sessions[user_id] = {"config": {"configurable": {"thread_id": str(uuid.uuid4())}}, "messages": formatted_messages}


    try:
        while True:
            data = await websocket.receive_text()
            user_message = HumanMessage(content=data)
            sessions[user_id]["messages"].append(user_message)

            for output in graph.stream(
                {"messages": sessions[user_id]["messages"]},
                config=sessions[user_id]["config"],
                stream_mode="updates"
            ):
                last_message = next(iter(output.values()))["messages"][-1]
                await websocket.send_text(last_message.content)

            if output:
                sessions[user_id]["messages"].append(last_message)
                #
    except Exception as e:
        await websocket.send_text(f"Error: {str(e)}")
        await websocket.close()

@router.delete("/end_session/{user_id}")
async def end_session(user_id: str):
    sessions.pop(user_id, None)
    return JSONResponse(content={"message": "Session ended."})

# Dependency injection
def get_sms_analyzer_service():
    return SMSAnalyzerService()

# API endpoints
@router.get("/health-check/")
def health_check():
    try:
        result = health_check()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
def read_root():
    return {"message": "Welcome to sms-analzer app!"}


async def add_item(item: dict):
    try:
        result = add_item(item)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get-item/{item_id}")
async def get_item(item_id: str):
    try:
        result = get_or_create_item(item_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-item/{item_id}")
async def update_item(item_id: str, new_data: dict):
    try:
        result = await update_item(item_id, new_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

class ChatSessionItem(BaseModel):
    id: str
    user_id: str
    title: str
    messages: list
    created_at: datetime = None

class ChatSessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_user_sessions(user_id: str):
    # Convert database models to response models if necessary
    sessions = [
        {"session_id": "123", "user_id": "456", "created_at": "2024-01-01T12:00:00Z"}
    ]
    return sessions

@router.get("/healthcheck")
def healthcheck():
    return health_check()

@router.post("/chat_session")
def create_chat_session(item: ChatSessionItem):
    return add_chat_session(item.dict())

@router.get("/chat_session/{item_id}")
def read_chat_session(item_id: str):
    return get_chat_session(item_id)

@router.put("/chat_session/{item_id}")
def update_chat_session_endpoint(item_id: str, new_data: ChatSessionItem):
    return update_chat_session(item_id, new_data.dict())

class LeadInformationCreateRequest(BaseModel):
    phone_number: str
    name: str
    email: str
    detail: str
    chatbot_id: str


    
    
@router.post("/create-leads-list", response_model=LeadInformationListResponse)
async def create_leads_list(name: str, file: UploadFile = File(...)):
    try:
        # Read the uploaded file into a DataFrame
        file_content = await file.read()  # Read the file content into memory
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(file_content))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(BytesIO(file_content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Validate required columns
        required_columns = {'Phone', 'Name', 'Email'}
        if not required_columns.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"Missing required columns: {required_columns - set(df.columns)}")

        # Create LeadInformation entries
        lead_ids = []
        for _, row in df.iterrows():
            lead_information = LeadInformation(
                phone_number=row['Phone'],
                name=row['Name'],
                email=row['Email'],
                detail=row.get('detail'),  # Optional field
                chatbot_id=row.get('chatbot_id'),  # Optional field
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            lead_information.save()
            lead_ids.append(lead_information.phone_number)

        # Create LeadInformationList entry
        lead_list = LeadInformationList(
            id=str(uuid.uuid4()),
            name=name,
            lead_information_ids=lead_ids,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await lead_list.save()

        return LeadInformationListResponse(
            id=lead_list.id,
            name=lead_list.name,
            lead_information_ids=lead_list.lead_information_ids,
            created_at=lead_list.created_at,
            updated_at=lead_list.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/get-leads-information-list", response_model=LeadInformationListResponse)
async def get_leads_information_list(list_id: str):
    try:
        # Query the LeadInformationList using the list_id
        lead_lists = LeadInformationList.query(list_id)
        lead_list = next(lead_lists, None)
        
        if not lead_list:
            raise HTTPException(status_code=404, detail="Lead information list not found")

        # Retrieve all LeadInformation entries using query or scan
        leads = []
        for lead_id in lead_list.lead_information_ids:
            try:
                # Fetch lead information by phone_number (hash key)
                lead_information = LeadInformation.query(lead_id, limit=1, scan_index_forward=False).next()
                if lead_information:
                    leads.append(lead_information.to_dict())
                print("!!!!!!!!!!!!!!!!!!!!!!",lead_information.to_dict())
            except StopIteration:
                # Handle case where lead is not found
                print(f"Lead with ID {lead_id} not found.")
            except Exception as e:
                # Log the error or handle it as needed
                print(f"Error retrieving lead with ID {lead_id}: {e}")

        return {
            "id": lead_list.id,
            "name": lead_list.name,
            "lead_information_ids": lead_list.lead_information_ids,
            "created_at": lead_list.created_at,
            "updated_at": lead_list.updated_at,
            "leads": leads
        }
    except StopIteration:
        raise HTTPException(status_code=404, detail="Lead information list not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    

@router.get("/get-all-leads-information-lists", response_model=List[LeadInformationListResponse])
async def get_all_leads_information_lists():
    try:
        # Scan all LeadInformationList entries
        all_lead_lists = LeadInformationList.scan()
        response_lists = []

        for lead_list in all_lead_lists:
            # Retrieve all LeadInformation entries for each list
            leads = []
            for lead_id in lead_list.lead_information_ids:
                try:
                    # Fetch lead information by phone_number (hash key)
                    lead_information = LeadInformation.query(
                        lead_id, 
                        limit=1, 
                        scan_index_forward=False
                    ).next()
                    if lead_information:
                        leads.append(lead_information.to_dict())
                except StopIteration:
                    print(f"Lead with ID {lead_id} not found.")
                except Exception as e:
                    print(f"Error retrieving lead with ID {lead_id}: {e}")

            # Create response object for each list
            list_response = {
                "id": lead_list.id,
                "name": lead_list.name,
                "lead_information_ids": lead_list.lead_information_ids,
                "created_at": lead_list.created_at,
                "updated_at": lead_list.updated_at,
                "leads": leads
            }
            response_lists.append(list_response)

        # Sort lists by created_at in descending order (newest first)
        response_lists.sort(
            key=lambda x: x["created_at"],
            reverse=True
        )

        return response_lists

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    


@router.post("/create_lead", response_model=LeadInformationResponse)
async def create_lead(request: LeadInformationCreateRequest):
    try:
        lead_information = LeadInformation(
            phone_number=request.phone_number,
            name=request.name,
            email=request.email,
            detail=request.detail,
            chatbot_id=request.chatbot_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        lead_information.save()

        return LeadInformationResponse(
            phone_number=lead_information.phone_number,
            name=lead_information.name,
            email=lead_information.email,
            detail=lead_information.detail,
            chatbot_id=lead_information.chatbot_id,
            created_at=lead_information.created_at,
            updated_at=lead_information.updated_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/get_lead", response_model=LeadInformationResponse)
async def get_lead(
    phone_number: Optional[str] = None,
    chatbot_id: Optional[str] = None
):
    """
    Retrieve lead information by phone_number or chatbot_id.
    """
    try:
        if phone_number:
            # Fetch lead information by phone_number (hash key)
            lead_information = LeadInformation.query(phone_number, limit=1, scan_index_forward=False).next()
            if not lead_information:
                raise HTTPException(status_code=404, detail="Lead not found.")
        elif chatbot_id:
            # Fetch lead information by chatbot_id
            leads = LeadInformation.scan(LeadInformation.chatbot_id == chatbot_id)
            lead_information = next(leads, None)
            if not lead_information:
                raise HTTPException(status_code=404, detail="No leads found for the specified chatbot.")
        else:
            raise HTTPException(status_code=400, detail="You must provide either phone_number or chatbot_id.")

        return LeadInformationResponse(
            phone_number=lead_information.phone_number,
            name=lead_information.name,
            email=lead_information.email,
            detail=lead_information.detail,
            chatbot_id=lead_information.chatbot_id,
            created_at=lead_information.created_at,
            updated_at=lead_information.updated_at
        )
    except StopIteration:
        raise HTTPException(status_code=404, detail="Lead not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime



# approach
class ChatRequest(BaseModel):
    session_id: Optional[str]
    message: str
    user_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str

class AgentState(BaseModel):
    messages: List[Dict[str, Any]]
    current_status: str
    user_id: str
    session_id: str
    context: Dict[str, Any] = {}

class LeadInformationResponse(BaseModel):
    phone_number: str
    name: str
    email: str
    detail: Optional[str] = None
    chatbot_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


@router.get("/chat-history/{session_id}", response_model=List[dict])
async def get_chat_history(session_id: str):
    try:
        
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Add these model classes after the existing ones
class CampaignSchedule(BaseModel):
    start_date: datetime
    end_date: datetime
    time_zone: str
    daily_start_time: time
    daily_end_time: time

class CampaignMetrics(BaseModel):
    total_leads: int
    processed_leads: int
    successful_conversations: int
    failed_conversations: int
    success_rate: float

from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime


class CampaignCreateRequest(BaseModel):
    name: str
    chatbot_id: str
    leads_list_id: str
    status: str
    schedule: Optional[CampaignSchedule] = None

class CampaignResponse(BaseModel):
    id: str
    name: str
    chatbot_id: str
    leads_list_id: str
    status: str
    # schedule: Optional[Dict] = None
    # metrics: Dict
    created_at: str  # Expect ISO format string
    # updated_at: str  # Expect ISO format string

class CampaignStatusUpdate(BaseModel):
    status: str

class CampaignAnalytics(BaseModel):
    campaign_id: str
    total_messages_sent: int
    total_responses: int
    response_rate: float
    average_response_time: float
    conversation_metrics: dict
    daily_metrics: List[dict]



from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid
from app.models.dynamodb import Chatbot, LeadInformationList, Campaign


from datetime import datetime, time
from typing import Dict, Optional
import uuid

from datetime import datetime, time
from typing import Dict, Optional
import uuid
from datetime import datetime, time
from typing import Dict, Optional
import uuid
from dateutil.parser import parse

from datetime import datetime
from typing import Dict, Optional
import uuid
from dateutil.parser import parse

from datetime import datetime
from typing import Dict, Optional
import uuid
from dateutil.parser import parse
from dateutil.tz import tzutc

from datetime import datetime, time
from typing import Dict, Optional
import uuid
from dateutil.tz import tzutc
from fastapi import HTTPException

from datetime import datetime, time
from fastapi import HTTPException

from datetime import datetime, time
from fastapi import HTTPException

from datetime import datetime, time

from pydantic import BaseModel


from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class LeadListResponse(BaseModel):
    id: str
    name: str
    total_leads: int
    created_at: str
    updated_at: str

class LeadResponse(BaseModel):
    phone_number: str
    name: str
    email: str
    detail: Optional[str]
    chatbot_id: Optional[str]
    created_at: str
    updated_at: str

class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str

class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    messages: List[ChatMessage]

class ScheduleModel(BaseModel):
    start_date: str
    end_date: str
    time_zone: str
    daily_start_time: str
    daily_end_time: str

class CampaignCreateRequest(BaseModel):
    name: str
    chatbot_id: str
    leads_list_id: str
    status: str
    schedule: ScheduleModel
    
    
    
def validate_schedule(schedule):
    try:
        # Use attribute access if schedule is an object
        start_date = schedule.start_date
        end_date = schedule.end_date
        daily_start_time = schedule.daily_start_time
        daily_end_time = schedule.daily_end_time

        # Parse the fields
        start_date = datetime.fromisoformat(start_date).date()
        end_date = datetime.fromisoformat(end_date).date()
        daily_start_time = time.fromisoformat(daily_start_time)
        daily_end_time = time.fromisoformat(daily_end_time)

        # Validation checks
        if start_date >= end_date:
            raise ValueError("End date must be after start date")
        if daily_start_time >= daily_end_time:
            raise ValueError("Daily end time must be after daily start time")

        return {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "time_zone": schedule.time_zone,
            "daily_start_time": str(daily_start_time),
            "daily_end_time": str(daily_end_time),
        }
    except AttributeError as e:
        raise ValueError(f"Invalid schedule format: {e}")
    except ValueError as ve:
        raise ValueError(f"Invalid schedule format: {str(ve)}")



@router.post("/create-campaign", response_model=CampaignResponse)
async def create_campaign(request: CampaignCreateRequest):
    try:
        # Validate chatbot exists
        try:
            chatbot = Chatbot.get(request.chatbot_id)
            print(f"Initial chatbot state: {chatbot.to_dict()}")
            
            if not chatbot:
                raise HTTPException(status_code=404, detail="Chatbot not found")
        except Chatbot.DoesNotExist:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Validate leads list exists
        lead_lists = LeadInformationList.query(request.leads_list_id)
        lead_list = next(lead_lists, None)
        if not lead_list:
            raise HTTPException(status_code=404, detail="Leads list not found")

        # Update chatbot's lead_information_list_ids
        try:
            # Get current list or initialize empty list
            current_list_ids = chatbot.lead_information_list_ids or []
            print(f"Current list IDs: {current_list_ids}")
            
            # Add new ID if not present
            if request.leads_list_id not in current_list_ids:
                current_list_ids.append(request.leads_list_id)
                
                # Update the chatbot
                chatbot.update(
                    actions=[
                        Chatbot.lead_information_list_ids.set(current_list_ids),
                        Chatbot.updated_at.set(datetime.utcnow())
                    ]
                )
                
                # Verify the update
                updated_chatbot = Chatbot.get(request.chatbot_id)
                print(f"Verified update - lead lists: {updated_chatbot.lead_information_list_ids}")
            
            print(f"Final chatbot state: {chatbot.to_dict()}")
            
        except Exception as e:
            print(f"Error updating chatbot: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating chatbot: {str(e)}")
 
        # Update leads' chatbot_id
        try:
            for phone_number in lead_list.lead_information_ids:
                try:
                    # Query by phone_number (hash key)
                    leads = LeadInformation.query(
                        phone_number,
                        limit=1,
                        scan_index_forward=False
                    )
                    lead = next(leads, None)
                    
                    if lead:
                        print(f"Found lead: {lead.to_dict()}")
                        print(f"Current chatbot_id: {lead.chatbot_id}")
                        print(f"New chatbot_id: {request.chatbot_id}")
                        
                        # Create a new instance with updated values
                        updated_lead = LeadInformation(
                            phone_number=lead.phone_number,
                            name=lead.name,
                            email=lead.email,
                            detail=lead.detail,
                            chatbot_id=request.chatbot_id,
                            created_at=lead.created_at,
                            updated_at=datetime.now(tzutc())
                        )
                        
                        try:
                            print("Attempting to save updated lead...")
                            updated_lead.save()
                            print(f"Successfully saved lead with phone number: {phone_number}")
                            print(f"Updated lead data: {updated_lead.to_dict()}")
                        except Exception as save_error:
                            print(f"Error saving lead: {str(save_error)}")
                            print(f"Lead data that failed to save: {updated_lead.to_dict()}")
                            raise save_error
                except Exception as e:
                    print(f"Error processing lead {phone_number}: {str(e)}")
                    continue
        # send first message
            try:
                customer_data = LeadInformation.scan()
                responses = []
                first_message_service1 = first_message_service.FirstMessageService()

                # TODO: enable it after testing 
                for customer in customer_data:
                    response = await first_message_service1.process_message(
                        session_id=customer.phone_number,  
                        message=f"please generate a first message for a user mentioned in this chat {customer},if there is a place holder for your name like [YourName] or similar please use {chatbot.name}, your name is SMSAI and  if there is a place holder like  [STREET NAME ONLY] or something similar, if the data is presented on the chat use that data else don't mention specific place name, please don't reply the place holder as it is.",
                        user_id=customer.phone_number,
                        chatbot_id=request.chatbot_id,
                        chatbot_script_id = chatbot.chatbot_script_id
                    )
                    responses.append(response)
            
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                    
        except Exception as e:
            print(f"Error updating leads: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating leads: {str(e)}")

        # Get current timestamp with UTC timezone
        current_time = datetime.now(tzutc())

        # Process schedule if it exists
        schedule_data = None
        if request.schedule:
            schedule_data = {
                'start_date': request.schedule.start_date,
                'end_date': request.schedule.end_date,
                'time_zone': request.schedule.time_zone,
                'daily_start_time': request.schedule.daily_start_time,
                'daily_end_time': request.schedule.daily_end_time
            }

        try:
            # Validate schedule
            if request.schedule:
                validated_schedule = validate_schedule(request.schedule)
            else:
                validated_schedule = None

            # Save to PynamoDB
            campaign = Campaign(
                id=str(uuid.uuid4()),
                name=request.name,
                chatbot_id=request.chatbot_id,
                leads_list_id=request.leads_list_id,
                status=request.status,
                schedule=validated_schedule,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
            )
            campaign.save()

            # Create response
            response_data = {
                "id": campaign.id,
                "name": campaign.name,
                "chatbot_id": campaign.chatbot_id,
                "leads_list_id": campaign.leads_list_id,
                "status": campaign.status,
                "schedule": schedule_data,
                # "metrics": campaign.metrics,
                "created_at": current_time.isoformat(),
                "updated_at": current_time.isoformat()
            }

            return response_data

        except Exception as e:
            print(f"Error creating campaign: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    except ValueError as ve:
        print(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid format: {str(ve)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/campaigns", response_model=List[Any])
async def get_all_campaigns():
    try:
        campaigns = []
        for campaign in Campaign.scan():
            campaigns.append(CampaignResponse(
                id=campaign.id,
                name=campaign.name,
                chatbot_id=campaign.chatbot_id,
                leads_list_id=campaign.leads_list_id,
                status=campaign.status,
                created_at=campaign.created_at,
                # schedule=campaign.schedule,
                # metrics=campaign.metrics
            ))
        return campaigns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: str):
    try:
        campaign = Campaign.get(hash_key=campaign_id)
        return CampaignResponse(
            id=campaign.id,
            name=campaign.name,
            chatbot_id=campaign.chatbot_id,
            leads_list_id=campaign.leads_list_id,
            status=campaign.status,
            created_at=campaign.created_at,
            schedule=campaign.schedule,
            metrics=campaign.metrics
        )
    except Campaign.DoesNotExist:
        raise HTTPException(status_code=404, detail="Campaign not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/campaigns/{campaign_id}/status")
async def update_campaign_status(campaign_id: str, status_update: CampaignStatusUpdate):
    try:
        campaign = Campaign.get(hash_key=campaign_id)
        campaign.status = status_update.status
        campaign.updated_at = datetime.utcnow()
        campaign.save()
        
        return {
            "id": campaign.id,
            "status": campaign.status,
            "updated_at": campaign.updated_at
        }
    except Campaign.DoesNotExist:
        raise HTTPException(status_code=404, detail="Campaign not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/campaigns/{campaign_id}/analytics", response_model=CampaignAnalytics)
async def get_campaign_analytics(campaign_id: str):
    try:
        campaign = Campaign.get(hash_key=campaign_id)
        
        # Calculate analytics from campaign data
        analytics = CampaignAnalytics(
            campaign_id=campaign.id,
            total_messages_sent=0,  # TODO: Implement actual metrics
            total_responses=0,
            response_rate=0.0,
            average_response_time=0.0,
            conversation_metrics={
                "successful": campaign.metrics["successful_conversations"],
                "failed": campaign.metrics["failed_conversations"],
                "pending": campaign.metrics["total_leads"] - campaign.metrics["processed_leads"]
            },
            daily_metrics=[]  # TODO: Implement daily metrics calculation
        )
        return analytics
    except Campaign.DoesNotExist:
        raise HTTPException(status_code=404, detail="Campaign not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str):
    try:
        campaign = Campaign.get(hash_key=campaign_id)
        campaign.delete()
        return {"success": True, "message": "Campaign deleted successfully"}
    except Campaign.DoesNotExist:
        raise HTTPException(status_code=404, detail="Campaign not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    
@router.get("/chatbot/{chatbot_id}/lead-lists", response_model=List[Dict])
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
            print("!!!!!!!!!!!!!!!leads",chatbot.lead_information_list_ids)
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
            # lead_list = LeadInformationList.get(hash_key=list_id)
            # lead_list = LeadInformationList.query(list_id).next()
            lead_lists = LeadInformationList.query(list_id)
            lead_list = next(lead_lists, None)
            
        except LeadInformationList.DoesNotExist:
            raise HTTPException(status_code=404, detail="Lead list not found")

        # Get all leads in the list
        leads = []
        if lead_list.lead_information_ids:
            for lead_id in lead_list.lead_information_ids:
                try:
                    # lead = LeadInformation.get(hash_key=lead_id)
                    lead = LeadInformation.query(
                lead_id, limit=1, scan_index_forward=False
            ).next()
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
    
    
    

    