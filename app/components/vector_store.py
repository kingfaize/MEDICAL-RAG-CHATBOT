from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException


DB_FAISS_PATH = "vectorstore/db_faiss"

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        db_path = DB_FAISS_PATH
        if os.path.exists(db_path):
            logger.info(f"Loading existing vectorstore at {db_path}...")
            return FAISS.load_local(
                db_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning(f"No vector store found at {db_path}.")
    except Exception as e:
        error_message = CustomException(f"Failed to load vectorstore" , e)
        logger.error(str(error_message))

# Creating new vectorstore function
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")
        logger.info(f"Generating new vectorstore")
        db_path = DB_FAISS_PATH
        vectorstore_dir = os.path.dirname(db_path)
        if not os.path.exists(vectorstore_dir):
            os.makedirs(vectorstore_dir)
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)
        logger.info(f"Saving vectorstore at {db_path}")
        db.save_local(db_path)
        logger.info("Vectorstore saved successfully...")
        return db
    except Exception as e:
        error_message = CustomException(f"Failed to create new vectorstore" , e)
        logger.error(str(error_message))
    

