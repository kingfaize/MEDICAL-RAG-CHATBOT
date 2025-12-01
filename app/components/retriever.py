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
    # Always reload LLM with default temperature 1
    temperature = 1.0
    from app.components.llm import load_llm
    db = load_vector_store()
    if db is None:
        return ("Error: Vector store not loaded. Please rebuild the vector store.", None)
    llm = load_llm(temperature=temperature)
    # Use chat history + current question for retrieval
    retrieval_query = f"{chat_history}\nUser: {user_query}" if chat_history else user_query
    retriever = db.as_retriever(search_kwargs={'k':2})
    retrieved_docs = retriever.invoke(retrieval_query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # Stronger chat history in prompt
    if chat_history:
        full_prompt = (
            "You are answering follow-up medical questions in a conversation. Use the chat history for context.\n"
            f"Chat History:\n{chat_history}\n\n"
            f"Current Question: {user_query}\n\n"
            f"Context:\n{context}\n\nAnswer in 2-3 lines, only using the context above."
        )
    else:
        full_prompt = set_custom_prompt().format(context=context, question=user_query)
    answer = llm.invoke(full_prompt)

    # Only return the answer string, no hallucination logic
    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
    # Remove backend metadata
    import re
    meta_match = re.search(r'(additional_kwargs|response_metadata|usage_metadata|id=)', answer_text)
    if meta_match:
        answer_text = answer_text[:meta_match.start()].strip()
    return answer_text, None



