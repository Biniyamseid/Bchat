from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi import HTTPException
import os

class SMSAnalyzerService:
    def __init__(self):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.conversation_chain = None
        self._initialize_vectorstore()
        self._initialize_conversation_chain()

    def _initialize_vectorstore(self):
        """Initialize an empty vectorstore."""
        
        texts = ["Default initialization text"]
        self.vectorstore = FAISS.from_texts(texts, embedding=self.embeddings)

    def _initialize_conversation_chain(self):
        """Initialize the conversation chain with memory and vectorstore."""
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory,
        )

    def add_text_to_vectorstore(self, text):
        """Split text into chunks and add it to the vectorstore."""
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        text_chunks = text_splitter.split_text(text)
        self.vectorstore.add_texts(text_chunks)

    def analyze_sms(self, sms: str):
        """Process SMS using the conversation chain."""
        if not self.conversation_chain:
            raise HTTPException(status_code=500, detail="Conversation chain not initialized")
        self.add_text_to_vectorstore(sms)
        
        question = (
            "You are a smart assistant designed to analyze SMS messages and provide structured insights. "
            "Analyze the given SMS message and provide the following: \n\n"
            "- **Intent:** Classify the SMS intent based on categories such as Wrong Number, Curious Response, Negative Response, Positive Response, Under Contract, Completed, Bankruptcy/Loss Modification, or No Foreclosure Here.\n"
            "- **Details:** Extract relevant details, such as user requests, conditions, or notable keywords that help understand the message context.\n\n"
            "Return your response as a JSON object with the following structure: \n\n"
            "{\n  \"intent\": \"<intent_category>\",\n  \"details\": \"<summary_of_extracted_details>\"\n}\n\n"
            f"SMS message: {sms}"
        )

        response = self.conversation_chain({"question": question})
        chat_history = response["chat_history"]

        # Format the chat history for the response
        formatted_history = [
            {"role": "user" if i % 2 == 0 else "bot", "message": message.content}
            for i, message in enumerate(chat_history)
        ]
        return {"chat_history": formatted_history}