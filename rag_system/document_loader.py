from pathlib import Path # this is a standard library module used for filesystem path manipulations
from langchain_community.document_loaders import TextLoader # this is a community loader, not part of the core langchain package ,it's used to load text files from disk
from langchain_core.documents import Document # this is the core Document class from langchain_core, used to represent documents with content and metadata

class DocumentLoader:
    """Component to handle document loading from disk and folder corpora."""
    
    def __init__(self, file_path: str = None):
        self.file_path = file_path

    def load(self) -> list[Document]:
        """Loads document content and returns a list of Documents."""
        if not self.file_path:
            raise ValueError("No file_path specified for single document loading.")
        loader = TextLoader(self.file_path, encoding="utf-8") # here we instantiate the TextLoader with the provided file path and specify UTF-8 encoding
        return loader.load() # this calls the load method of TextLoader, which reads the file and returns a list of Document objects

    def load_with_metadata(self, document_type: str = "policy", department: str = "support", access_level: str = "public") -> list[Document]:
        """Loads document and tags each Document with configured default metadata."""
        documents = self.load()
        documents_with_metadata = []
        for doc in documents:
            documents_with_metadata.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": Path(self.file_path).name,
                        "document_type": document_type,
                        "department": department,
                        "access_level": access_level
                    }
                )
            )
        return documents_with_metadata

    def infer_metadata_from_filename(self, file_path: Path) -> dict:
        """Heuristically infers metadata parameters from the filename structure."""
        filename = file_path.name.lower()

        if "refund" in filename:
            topic = "refund"
            document_type = "policy"
            access_level = "admin_visible"
        elif "learning" in filename:
            topic = "learning"
            document_type = "learning_plan"
            access_level = "student_visible"
        else:
            topic = "general_policy"
            document_type = "policy"
            access_level = "student_visible"

        return {
            "source": file_path.name,
            "topic": topic,
            "document_type": document_type,
            "access_level": access_level,
            "department": "support"
        }

    def load_corpus(self, directory_path: str) -> list[Document]:
        """Loads all txt documents from a directory and assigns metadata based on filename."""
        all_documents = []
        dir_path = Path(directory_path) # here we create a Path object for the directory path
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {directory_path}")

        for file_path in dir_path.glob("*.txt"):
            loader = TextLoader(str(file_path), encoding="utf-8") # here we instantiate the TextLoader for each text file found in the directory, specifying UTF-8 encoding
            loaded_docs = loader.load() # this calls the load method of TextLoader, which reads the file and returns a list of Document objects
            metadata = self.infer_metadata_from_filename(file_path) # this calls the infer_metadata_from_filename method to generate metadata based on the filename structure
            # below we append each loaded document to the all_documents list, wrapping it in a new Document object that includes the inferred metadata
            for doc in loaded_docs:
                all_documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=metadata
                    )
                )
        return all_documents
