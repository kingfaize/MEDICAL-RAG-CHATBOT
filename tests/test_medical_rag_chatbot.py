import unittest
import os
from app.components.embeddings import get_embedding_model
from app.components.vector_store import save_vector_store, load_vector_store
from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.llm import load_llm
from app.components.retriever import retrieve_context

class TestMedicalRAGChatbot(unittest.TestCase):
    def test_openai_embedding(self):
        model = get_embedding_model()
        self.assertIsNotNone(model)

    # def test_huggingface_embedding(self):
    #     model = get_embedding_model(provider="huggingface")
        self.assertIsNotNone(model)

    def test_pdf_loading(self):
        docs = load_pdf_files()
        self.assertIsInstance(docs, list)

    def test_text_chunking(self):
        docs = load_pdf_files()
        chunks = create_text_chunks(docs)
        self.assertIsInstance(chunks, list)

    def test_vector_store_save_and_load(self):
        docs = load_pdf_files()
        chunks = create_text_chunks(docs)
        db = save_vector_store(chunks)
        self.assertIsNotNone(db)
        db_loaded = load_vector_store()
        self.assertIsNotNone(db_loaded)

    def test_openai_llm(self):
        llm = load_llm()
        self.assertIsNotNone(llm)

    # def test_huggingface_llm(self):
    #     llm = load_llm(provider="huggingface")
        self.assertIsNotNone(llm)

    def test_retrieve_context_openai(self):
        result = retrieve_context.invoke("What is diabetes?")
        self.assertTrue(isinstance(result, str) or hasattr(result, 'content'))

    # def test_retrieve_context_huggingface(self):
    #     result = retrieve_context.invoke("What is diabetes?", provider="huggingface")
        self.assertTrue(isinstance(result, str) or hasattr(result, 'content'))

if __name__ == "__main__":
    unittest.main()
