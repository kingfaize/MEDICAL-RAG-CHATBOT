
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import Optional
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(temperature: float = 1.0, model: str = "gpt-3.5-turbo"):
    logger.info("Loading LLM from OpenAI")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    print("[llm.py] OPENAI_API_KEY at load_llm:", openai_api_key)
    if not openai_api_key:
        raise CustomException("OPENAI_API_KEY environment variable is not set. [llm.py]")
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    logger.info("OpenAI LLM loaded successfully...")
    return llm