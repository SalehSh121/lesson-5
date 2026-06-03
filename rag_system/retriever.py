from langchain_chroma import Chroma
from langchain_core.documents import Document

class RetrieverService:
    """Component to manage document retrieval from the vector store using top-k, MMR, and scores."""

    def __init__(self, vector_store: Chroma, top_k: int = 3, fetch_k: int = 8, refuse_threshold: float = 0.3):
        self.vector_store = vector_store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.refuse_threshold = refuse_threshold

    def retrieve_top_k(self, query: str) -> list[Document]:
        """Retrieves documents using standard similarity search."""
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        return retriever.invoke(query)

    def retrieve_mmr(self, query: str) -> list[Document]:
        """Retrieves documents using Maximal Marginal Relevance."""
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.top_k,
                "fetch_k": self.fetch_k
            }
        )
        return retriever.invoke(query)

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Retrieves documents along with their relevance scores."""
        return self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=self.top_k
        )

    def should_refuse(self, scored_results: list[tuple[Document, float]]) -> bool:
        """Checks if the highest retrieved score falls below the refuse threshold."""
        if not scored_results:
            return True
        best_doc, best_score = scored_results[0]
        return best_score < self.refuse_threshold
