import boto3
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
import openai

router = APIRouter()





