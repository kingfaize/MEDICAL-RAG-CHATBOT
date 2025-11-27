from langchain_huggingface import HuggingFaceEndpoint
from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID
from typing import Optional

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: Optional[str] = None):
    try:
        logger.info("Loading LLM from HuggingFace")
        # Use .env token if not provided
        if hf_token is None:
            from app.config.config import HF_TOKEN as ENV_HF_TOKEN
            hf_token = ENV_HF_TOKEN
        if not hf_token:
            raise CustomException("HF_TOKEN environment variable is not set.")
        llm = HuggingFaceEndpoint(
            model=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            task="conversational",
            temperature=0.3,
            max_new_tokens=256
        )
        logger.info("LLM loaded sucesfully...")
        return llm
    
    except Exception as e:
        error_message = CustomException("Failed to load a llm" , e)
        logger.error(str(error_message))