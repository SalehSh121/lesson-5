from langchain_text_splitters import RecursiveCharacterTextSplitter # this is a text splitter from langchain_text_splitters, used to split documents into smaller chunks based on character count and overlap
from langchain_core.documents import Document # this is the core Document class from langchain_core, used to represent documents with content and metadata

class ChunkingService:
    """Component to split documents into chunks and enrich metadata with chunk indexes."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Splits the input documents and appends sequence chunk indices to their metadata."""
        chunks = self.splitter.split_documents(documents) # this calls the split_documents method of RecursiveCharacterTextSplitter, which takes a list of Document objects and splits them into smaller chunks based on the specified chunk size and overlap
        for index, chunk in enumerate(chunks): # this loop iterates over the list of chunks, and for each chunk, it assigns a sequential index to the "chunk_index" key in the chunk's metadata dictionary
            chunk.metadata["chunk_index"] = index
        return chunks
