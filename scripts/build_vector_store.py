"""
Script to build and save the FAISS vector store for MEDICAL-RAG-CHATBOT.
Run this script once to initialize your vector store from your data.
"""

from app.components.vector_store import save_vector_store
from app.components.pdf_loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load documents from the data folder
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
documents = load_documents(DATA_PATH)

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = splitter.split_documents(documents)

# Build and save the vector store
save_vector_store(text_chunks)

print("Vector store built and saved successfully.")
