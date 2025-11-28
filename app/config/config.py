import os
from dotenv import load_dotenv
load_dotenv()
DB_FAISS_PATH="vectorstore/db_faiss"
DATA_PATH="data/"
CHUNK_SIZE=1024
CHUNK_OVERLAP=200
OPENAI_MODEL="gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
