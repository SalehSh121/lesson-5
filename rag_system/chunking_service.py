from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class ChunkingService:
    """Component to split documents into chunks and enrich metadata with chunk indexes."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Splits the input documents and appends sequence chunk indices to their metadata."""
        chunks = self.splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks
