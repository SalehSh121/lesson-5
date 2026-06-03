from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

class DocumentLoader:
    """Component to handle document loading from disk."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        """Loads document content and returns a list of Documents."""
        loader = TextLoader(self.file_path, encoding="utf-8")
        return loader.load()

    def load_with_metadata(self, document_type: str = "policy", department: str = "support", access_level: str = "public") -> list[Document]:
        """Loads document and tags each Document with configured default metadata."""
        documents = self.load()
        documents_with_metadata = []
        for doc in documents:
            documents_with_metadata.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": self.file_path,
                        "document_type": document_type,
                        "department": department,
                        "access_level": access_level
                    }
                )
            )
        return documents_with_metadata
