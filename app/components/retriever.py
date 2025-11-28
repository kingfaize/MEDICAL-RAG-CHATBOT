from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
import os

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.config.config import CHUNK_SIZE, CHUNK_OVERLAP
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
    from app.components.embeddings import get_embedding_model
    if db is None:
        db = load_vector_store()
    if llm is None:
        llm = load_llm()
    if db is None:
        raise CustomException("Vector store not present or empty")
    if llm is None:
        raise CustomException("LLM not loaded")
    return db, llm

@tool("retrieve_context", response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    # Accept dict with 'query' and 'chat_history'
    if isinstance(query, dict):
        user_query = query.get('query', '')
        chat_history = query.get('chat_history', '')
    else:
        user_query = query
        chat_history = ''
    db, llm = setup_rag_components()
    retriever = db.as_retriever(search_kwargs={'k':2})
    retrieved_docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # Include chat history in the prompt
    full_prompt = f"Chat History:\n{chat_history}\n\n" + set_custom_prompt().format(context=context, question=user_query)
    answer = llm.invoke(full_prompt)
    # Ensure response is a tuple for 'content_and_artifact' format
    if hasattr(answer, 'content'):
        return answer.content, None
    return str(answer), None



