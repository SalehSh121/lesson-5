from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingService:
    """Component to manage embedding models."""

    def __init__(self, model: str = "gemini-embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Returns the configured embeddings instance."""
        return self.embeddings
