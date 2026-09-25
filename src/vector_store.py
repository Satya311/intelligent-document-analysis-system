from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import config

class VectorStoreManager:
    def __init__(self):
        self.embeddings = self._init_embeddings()
        self.vector_db = None

    def _init_embeddings(self):
        if config.EMBEDDING_PROVIDER == "google" and config.GOOGLE_API_KEY:
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config.GOOGLE_API_KEY)
        elif config.EMBEDDING_PROVIDER == "openai" and config.OPENAI_API_KEY:
            return OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
        else:
            return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL)

    def build_index(self, enriched_chunks: List[Dict[str, Any]]):
        documents = []
        for c in enriched_chunks:
            doc = Document(
                page_content=c["text"],
                metadata={
                    "chunk_id": c["chunk_id"],
                    "page_number": c["page_number"],
                    "topic_cluster": c.get("topic_cluster", 0),
                    "topic_name": c.get("topic_name", "General"),
                    "content_type": c.get("content_type", "Concept"),
                    "file_name": c["file_name"]
                }
            )
            documents.append(doc)

        self.vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR
        )

    def similarity_search(self, query: str, top_k: int = 4) -> List[Document]:
        if not self.vector_db:
            raise ValueError("Vector DB index has not been built yet.")
        return self.vector_db.similarity_search(query, k=top_k)
