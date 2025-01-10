# from typing import Optional
# import uuid
# from datetime import datetime
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from ..models.dynamodb import ChatSession, Message, ChatbotScript
# # from ..models.db import health_check, add_item, get_item, update_item,get_or_create_item

# from ..core.config import get_settings

# from dotenv import load_dotenv
# import getpass
# import os
# import boto3

# from fastapi import FastAPI, HTTPException, WebSocket
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from io import BytesIO
# from ..schemas.input_text import InputText
# from ..services.sms_analyzer_service import SMSAnalyzerService
# from  ..models.db import *
# import uuid
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
# from datetime import datetime
# from typing import Literal, List, Dict
# from langgraph.graph import END
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph, START
# from langgraph.graph.message import add_messages
# from typing import Annotated
# from typing_extensions import TypedDict
# from langchain_community.chat_models import ChatOpenAI

# load_dotenv()


# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from jose import JWTError, jwt
# from passlib.context import CryptContext
# from datetime import datetime, timedelta

# app = FastAPI()



# template = ""
# script_info = ""
# script = next(ChatbotScript.scan(), None)
# if script:
#         script_info =  f"You are a highly intelligent and adaptive chatbot designed to interact with customers based on the following script, please Follow this script for responses, if there are placeholders you should ask information from customer and use that information on future interactions: {script.content} if there is a place holder for your name like [YourName] or similar please use SMSAI, your name is SMSAI if there is a place holder for [STREET NAME ONLY] or something similar if the data is presented on the chat use that data else don't mention specific place name "
# template += script_info


# from fastapi import HTTPException
# from datetime import datetime
# from PyPDF2 import PdfReader
# import uuid

# # Simulated workflow and database (replace with real LangChain and DynamoDB code)
# cached_responses = []

# # Request schema
# class MessageRequest(BaseModel):
#     user_id: str
#     message: str

# # Response schema
# class MessageResponse(BaseModel):
#     response: str

# dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
# table = dynamodb.Table('db')



# def get_messages_info(messages):
    
#     return [SystemMessage(content=template)] + messages

# class PromptInstructions(BaseModel):
#     """Instructions on how to prompt the LLM."""
#     objective: str
#     variables: List[str]
#     constraints: List[str]
#     requirements: List[str]

# llm = ChatOpenAI(temperature=0)
# llm_with_tool = llm.bind_tools([PromptInstructions])

# def info_chain(state):
#     messages = get_messages_info(state["messages"])
#     response = llm_with_tool.invoke(messages)
#     return {"messages": [response]}

# prompt_system = """Based on the following requirements, interact with the customer:

# {reqs}"""

# def get_prompt_messages(messages: list):
#     tool_call = None
#     other_msgs = []
#     for m in messages:
#         if isinstance(m, AIMessage) and m.tool_calls:
#             tool_call = m.tool_calls[0]["args"]
#         elif isinstance(m, ToolMessage):
#             continue
#         elif tool_call is not None:
#             other_msgs.append(m)
#     return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs

# def prompt_gen_chain(state):
#     messages = get_prompt_messages(state["messages"])
#     response = llm.invoke(messages)
#     return {"messages": [response]}

# def get_state(state):
#     messages = state["messages"]
#     if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
#         return "add_tool_message"
#     elif not isinstance(messages[-1], HumanMessage):
#         return END
#     return "info"

# def store_chat_history(user_id: str, chat_history):
#     """Store chat history in DynamoDB."""
#     timestamp = str(datetime.utcnow().timestamp())
#     table.put_item(
#         Item={
#             'user_id': user_id,
#             'timestamp': timestamp,
#             'chat_history': [msg.dict() for msg in chat_history]
#         }
#     )
    

# def retrieve_chat_history(user_id: str):
#     """Retrieve chat history from DynamoDB."""
#     response = table.query(
#         KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id)
#     )
#     return response['Items']

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# memory = MemorySaver()
# workflow = StateGraph(State)
# workflow.add_node("info", info_chain)
# workflow.add_node("prompt", prompt_gen_chain)

# @workflow.add_node
# def add_tool_message(state: State):
#     return {
#         "messages": [
#             ToolMessage(
#                 content="generated!",
#                 tool_call_id=state["messages"][-1].tool_calls[0]["id"],
#             )
#         ]
#     }

# workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
# workflow.add_edge("add_tool_message", "prompt")
# workflow.add_edge("prompt", END)
# workflow.add_edge(START, "info")
# graph = workflow.compile(checkpointer=memory)

# from IPython.display import Image, display

# # display(Image(graph.get_graph().draw_mermaid_png()))

# sessions: Dict[str, dict] = {}
# def get_messages_info(messages):
    
#     return [SystemMessage(content=template)] + messages

# class PromptInstructions(BaseModel):
#     """Instructions on how to prompt the LLM."""
#     objective: str
#     variables: List[str]
#     constraints: List[str]
#     requirements: List[str]

# llm = ChatOpenAI(temperature=0)
# llm_with_tool = llm.bind_tools([PromptInstructions])

# def info_chain(state):
#     messages = get_messages_info(state["messages"])
#     response = llm_with_tool.invoke(messages)
#     return {"messages": [response]}

# prompt_system = """Based on the following requirements, interact with the customer:

# {reqs}"""

# def get_prompt_messages(messages: list):
#     tool_call = None
#     other_msgs = []
#     for m in messages:
#         if isinstance(m, AIMessage) and m.tool_calls:
#             tool_call = m.tool_calls[0]["args"]
#         elif isinstance(m, ToolMessage):
#             continue
#         elif tool_call is not None:
#             other_msgs.append(m)
#     return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


# def prompt_gen_chain(state):
#     messages = get_prompt_messages(state["messages"])
#     response = llm.invoke(messages)
#     return {"messages": [response]}

# def get_state(state):
#     messages = state["messages"]
#     if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
#         return "add_tool_message"
#     elif not isinstance(messages[-1], HumanMessage):
#         return END
#     return "info"

# def store_chat_history(user_id: str, chat_history):
#     """Store chat history in DynamoDB."""
#     timestamp = str(datetime.utcnow().timestamp())
#     table.put_item(
#         Item={
#             'user_id': user_id,
#             'timestamp': timestamp,
#             'chat_history': [msg.dict() for msg in chat_history]
#         }
#     )
    

# def retrieve_chat_history(user_id: str):
#     """Retrieve chat history from DynamoDB."""
#     response = table.query(
#         KeyConditionExpression=boto3.dynamodb.conditions.Key('user_id').eq(user_id)
#     )
#     return response['Items']

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# memory = MemorySaver()
# workflow = StateGraph(State)
# workflow.add_node("info", info_chain)
# workflow.add_node("prompt", prompt_gen_chain)

# @workflow.add_node
# def add_tool_message(state: State):
#     return {
#         "messages": [
#             ToolMessage(
#                 content="generated!",
#                 tool_call_id=state["messages"][-1].tool_calls[0]["id"],
#             )
#         ]
#     }

# workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
# workflow.add_edge("add_tool_message", "prompt")
# workflow.add_edge("prompt", END)
# workflow.add_edge(START, "info")
# graph = workflow.compile(checkpointer=memory)

# from IPython.display import Image, display

# # display(Image(graph.get_graph().draw_mermaid_png()))

# sessions: Dict[str, dict] = {}




# settings = get_settings()



# import logging

# class ChatAgentService:
#     def __init__(self):
#         self.sessions: Dict[str, dict] = {}
#         self.llm = ChatOpenAI(
#             api_key=settings.OPENAI_API_KEY,
#             model="gpt-4-turbo-preview"
#         )
#         self.memory = ConversationBufferMemory()
#         self.conversation = ConversationChain(
#             llm=self.llm,
#             memory=self.memory,
#             verbose=True
#         )
    
#     # the session_id and user_id should be phone_number since we are building this system for sms and while registering the user  must be using phone number
#     async def process_message(
#         self,
#         session_id: str,
#         message: str,
#         user_id: Optional[str] = "1"
#     ):
#         print("!!!!!!!!!!!!!!!!!!!!!! inside chatagent process_message")
#         print("***********first stage")
        
#         # try:
#         #     # get the message by id. If it does not exist, create a new one
#         #     # first set it with in memory session then move it to the database.
#         #     pass
#         # except Exception as e:
#         #     logging.error(f"Error in initial try block: {str(e)}")
        
#         try:
#             if user_id not in self.sessions:
#                 self.sessions[user_id] = {"config": {"configurable": {"thread_id": str(user_id)}}, "messages": "messages"}
#                 logging.info(f"Created new session for user_id: {user_id}")
#         except Exception as e:
#             logging.error(f"Error in session creation: {str(e)}")
            
#         # Create or get session
#         logging.info(f"************************** before ")
#         print("**********second stage")
        
#         try:
#             print("**********b third stage")
#             session = ChatSession.get(session_id)
#             print("**********third stage",session)
#             logging.info(f"************************** Retrieved session: {session_id}")
#         except ChatSession.DoesNotExist:
#             print("**********forth stage")
#             session = ChatSession(
#                 id=session_id or str(uuid.uuid4()),
#                 user_id=user_id or "anonymous",
#                 title="New Chat",
#                 messages=[]
#             )
#             print("**********fifth stage",session,session.id)
#             logging.info(f"********************************** Created new session: {session.id}")
        
#         # Add user message
#         user_message = Message(
#             id=str(uuid.uuid4()),
#             role="user",
#             content=message,
#             timestamp=datetime.utcnow()
#         )
#         session.messages.append(user_message)
#         logging.info(f"Added user message: {user_message.content}")
        
#         # Process the message using the LangChain graph
#         print("!!!!!!!!!!!!!!!!!!!!!!! before graph.stream")
#         output = None
#         assistant_message = ""
#         try:
#             print(f"!!!!!!!!!!!!!!!!!!!!!!! inside try", graph.stream(
#                 {"messages": self.sessions[user_id]["messages"]},
#                 config=self.sessions[user_id]["config"],
#                 stream_mode="updates"
#             ))
        
#             for output in graph.stream(
#                 {"messages": self.sessions[user_id]["messages"]},
#                 config=self.sessions[user_id]["config"],
#                 stream_mode="updates"
#             ):
#                 print("!!!!!!!!!!!!!!!!!!!!!!! output", output)
#                 if output:
#                     last_message = next(iter(output.values()))["messages"][-1]
#                     print("!!!!!!!!!!!!!!!!! lastmessage", last_message)
#                     # self.sessions[user_id]["messages"].append(last_message.content)
#                     # Add assistant message
#                     assistant_message = Message(
#                         id=str(uuid.uuid4()),
#                         role="assistant",
#                         content=last_message.content,
#                         timestamp=datetime.utcnow()
#                     )
#                     # session.messages.append(assistant_message)
#                     logging.info(f"Added assistant message")
            
#                     return {
#                         "session_id": session_id,
#                         "response": last_message.content
#                     }
#                 raise Exception("No output generated.")
#         except Exception as e:
#             logging.error(f"Error in graph stream: {str(e)}")
#             return {
#                 "session_id": session.id,
#                 "response": f"error {e}"
#             }
        
#         # Save session
#         session.save()
#         logging.info(f"Session saved: {session.id}")
        
#         return {
#             "session_id": session.id,
#             "response": assistant_message.content if assistant_message else "No response generated"
#         }





# # class ChatAgentService:
# #     def __init__(self):
# #         self.sessions: Dict[str, dict] = {}
# #         self.llm = ChatOpenAI(
# #             api_key=settings.OPENAI_API_KEY,
# #             model="gpt-4-turbo-preview"
# #         )
# #         self.memory = ConversationBufferMemory()
# #         self.conversation = ConversationChain(
# #             llm=self.llm,
# #             memory=self.memory,
# #             verbose=True
# #         )
    
# #     # the session_id and user_id should be phone_number since we are building this system for sms and while registering the user  must be using phone number
# #     async def process_message(
# #         self,
# #         session_id: str,
# #         message: str,
# #         user_id: Optional[str] = "1"
# #     ):
# #         print("!!!!!!!!!!!!!!!!!!!!!! inside chatagent process_message")
        
# #         try:
# #             # get the message by id. If it does not exist, create a new one
# #             # first set it with in memory session then move it to the database.
# #             pass
# #         except:
# #             pass
# #         try:
# #             if user_id not in sessions:
# #                 self.sessions[user_id] = {"config": {"configurable": {"thread_id": str(user_id)}}, "messages": "messages"}

            
# #         except:
# #             pass
            
# #         # Create or get session
# #         try:
# #             session = ChatSession.get(session_id)
# #         except ChatSession.DoesNotExist:
# #             session = ChatSession(
# #                 id=session_id or str(uuid.uuid4()),
# #                 user_id=user_id or "anonymous",
# #                 title="New Chat",
# #                 messages=[]
# #             )
            
            
        
# #         # Add user message
# #         # TODO id should be phone number and adding and retrieving from the database should be done like this one
# #         user_message = Message(
# #             id=str(uuid.uuid4()),
# #             role="user",
# #             content=message,
# #             timestamp=datetime.utcnow()
# #         )
# #         session.messages.append(user_message)
# #         # send request and receive a response over here.
        
# #         # Process the message using the LangChain graph
# #         print("!!!!!!!!!!!!!!!!!!!!!!! before graph.stream")
# #         output = None
# #         assistant_message = ""
# #         try:
# #             print(f"!!!!!!!!!!!!!!!!!!!!!!! inside try",graph.stream(
# #                 {"messages":"hello"},
# #                 config=self.sessions[user_id]["config"],
# #                 stream_mode="updates"
# #             ))
        
# #             for output in graph.stream(
# #                 {"messages":"hello"},
# #                 config=self.sessions[user_id]["config"],
# #                 stream_mode="updates"
# #             ):
# #                 print("!!!!!!!!!!!!!!!!!!!!!!! output", output)
# #                 if output:
# #                     last_message = next(iter(output.values()))["messages"][-1]
# #                     print("!!!!!!!!!!!!!!!!! lastmessage",last_message)
# #                     # sessions[user_id]["messages"].append(last_message.content)
# #                     # # Add assistant message
# #                     assistant_message = Message(
# #                         id=str(uuid.uuid4()),
# #                         role="assistant",
# #                         content=last_message.content,
# #                         # content="last_message.content",
# #                         timestamp=datetime.utcnow()
# #                     )
# #                     session.messages.append("assistant_message")
            
# #                     return {"session_id":session_id,
# #                         "response": last_message.content}
# #                 raise Exception("No output generated.")
# #         except Exception as e:
# #             return {
# #             "session_id": session.id,
# #             "response": f"error {e}"
# #             }
# #             return {"error":"errororoeeeeeeeee"}
# #             raise HTTPException(status_code=500, detail=str(e))
        
        
        
        

        
        
        
#         # # Get chatbot script
#         # script = next(ChatbotScript.scan(), None)
#         # if script:
#         #     self.conversation.memory.chat_memory.add_user_message(
#         #         f"based on the following  script please generate a first message for a user mentioned in this chat: {script.content}"
#         #     )
        
#         # # Generate response
#         # response = self.conversation.predict(input=message)
        
#         # # Add assistant message
#         # assistant_message = Message(
#         #     id=str(uuid.uuid4()),
#         #     role="assistant",
#         #     content=response,
#         #     timestamp=datetime.utcnow()
#         # )
#         # session.messages.append(assistant_message)
        
#         # Save session
#         session.save()
        
#         return {
#             "session_id": session.id,
#             "response": assistant_message
#         }
        
#     def initialfirstmessagegenerator(customer_data, script_content):
#             # Extract necessary details from customer data
#             customer_name = customer_data.get("name")
#             phone_number = customer_data.get("phone_number")
            
#             # Generate the initial message using the script content
#             initial_message = f"Hello {customer_name},\n\n{script_content}\n\n!"
            
#             # Return the initial message
#             return initial_message
        

    


