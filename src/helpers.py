import os
from pypdf import PdfReader
from langchain.embeddings import GooglePalmEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_pdf_text(pdf_file):
    text = ""
    for pdf in pdf_file:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([text])
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_conversational_chain(vector_store):
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=os.getenv("GEMINI_API_KEY"))
    conversational_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)
    return conversational_chain

