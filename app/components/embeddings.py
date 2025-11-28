
from langchain_openai import OpenAIEmbeddings
import os
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Initializing our OpenAI embedding model")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise CustomException("OPENAI_API_KEY environment variable is not set. [embeddings.py]")
        model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        logger.info("OpenAI embedding model loaded successfully....")
        return model
    except Exception as e:
        error_message=CustomException("Error occured while loading embedding model" , e)
        logger.error(str(error_message))
        raise error_message