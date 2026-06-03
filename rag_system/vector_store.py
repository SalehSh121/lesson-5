import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .embedding_service import EmbeddingService

class VectorStoreManager:
    """Component to manage vector database instantiation, persistence, and rebuilding."""

    def __init__(self, persist_directory: str, embedding_service: EmbeddingService):
        self.persist_directory = persist_directory
        self.embeddings = embedding_service.get_embeddings()

    def get_vector_store(self, chunks: list[Document] = None, rebuild: bool = False) -> Chroma:
        """
        Retrieves the Chroma vector store. If rebuild is True, wipes the old directory and
        builds a fresh store using the provided chunks.
        """
        if rebuild and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Old Chroma vector store deleted at: {self.persist_directory}")

        if rebuild and chunks:
            print("Rebuilding vector store from chunks...")
            return Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            print(f"Loading vector store from: {self.persist_directory}")
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
