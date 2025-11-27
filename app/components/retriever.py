from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
import os

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import HUGGINGFACE_REPO_ID
from app.common.logger import get_logger
from app.common.custom_exception import CustomException


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)


# Agent-based RAG pattern
db = None
llm = None

def setup_rag_components():
    global db, llm
    if db is None:
        db = load_vector_store()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise CustomException("HF_TOKEN environment variable is not set.")
    if llm is None:
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=hf_token)
    if db is None:
        raise CustomException("Vector store not present or empty")
    if llm is None:
        raise CustomException("LLM not loaded")
    return db, llm

@tool("retrieve_context", response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    db, llm = setup_rag_components()
    retriever = db.as_retriever(search_kwargs={'k':2})
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = set_custom_prompt().format(context=context, question=query)
    answer = llm.invoke([{"role": "user", "content": prompt}])
    return answer



