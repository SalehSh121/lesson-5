from langchain_chroma import Chroma
from langchain_core.documents import Document

class RetrieverService:
    """Component to manage document retrieval from the vector store using top-k, MMR, hybrid search, and scores."""

    def __init__(self, vector_store: Chroma, top_k: int = 3, fetch_k: int = 8, refuse_threshold: float = 0.3):
        self.vector_store = vector_store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.refuse_threshold = refuse_threshold
        self.has_hybrid = False
        self.all_chunks = []
        self.sparse_weight = 0.4
        self.dense_weight = 0.6

    def enable_hybrid_search(self, chunks: list[Document], sparse_weight: float = 0.4, dense_weight: float = 0.6):
        """Prepares the retriever to support hybrid (sparse + dense) search by storing raw chunks."""
        self.all_chunks = chunks
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.has_hybrid = True

    def retrieve_top_k(self, query: str, metadata_filter: dict = None, min_date: str = None) -> list[Document]:
        """Retrieves documents using standard similarity search, with optional metadata and date filtering."""
        search_kwargs = {"k": self.top_k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        docs = retriever.invoke(query)
        if min_date:
            docs = [d for d in docs if d.metadata.get("last_updated", "0000-00-00") >= min_date]
        return docs

    def retrieve_mmr(self, query: str, metadata_filter: dict = None, min_date: str = None) -> list[Document]:
        """Retrieves documents using Maximal Marginal Relevance, with optional metadata and date filtering."""
        search_kwargs = {
            "k": self.top_k,
            "fetch_k": self.fetch_k
        }
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        docs = retriever.invoke(query)
        if min_date:
            docs = [d for d in docs if d.metadata.get("last_updated", "0000-00-00") >= min_date]
        return docs

    def retrieve_hybrid(self, query: str, metadata_filter: dict = None, min_date: str = None) -> list[Document]:
        """Retrieves documents using hybrid (sparse BM25 + dense Chroma) search with metadata and date filtering."""
        if not self.has_hybrid:
            raise ValueError("Hybrid search is not enabled. Call enable_hybrid_search(chunks) first.")

        from langchain_community.retrievers import BM25Retriever
        try:
            from langchain_classic.retrievers import EnsembleRetriever
        except ImportError:
            try:
                from langchain.retrievers import EnsembleRetriever
            except ImportError:
                from langchain_community.retrievers import EnsembleRetriever

        # Filter chunks for sparse search if metadata/date filter is provided
        filtered_chunks = self.all_chunks
        if metadata_filter or min_date:
            filtered_chunks = [
                c for c in self.all_chunks
                if (not metadata_filter or all(c.metadata.get(k) == v for k, v in metadata_filter.items())) and
                   (not min_date or c.metadata.get("last_updated", "0000-00-00") >= min_date)
            ]

        if not filtered_chunks:
            return []

        # Setup sparse keyword search on filtered documents
        bm25_retriever = BM25Retriever.from_documents(filtered_chunks)
        bm25_retriever.k = self.top_k

        # Setup dense semantic search with filters
        dense_kwargs = {"k": self.top_k}
        if metadata_filter:
            dense_kwargs["filter"] = metadata_filter
        dense_retriever = self.vector_store.as_retriever(search_kwargs=dense_kwargs)

        # Assemble hybrid search
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[self.sparse_weight, self.dense_weight]
        )
        docs = ensemble_retriever.invoke(query)
        if min_date:
            docs = [d for d in docs if d.metadata.get("last_updated", "0000-00-00") >= min_date]
        return docs

    def retrieve_with_role(self, query: str, user_role: str = "student", retrieval_mode: str = "top_k", min_date: str = None) -> list[Document]:
        """Filters retrieved documents based on the user's role-based access level and min_date."""
        if user_role == "student":
            metadata_filter = {"access_level": "student_visible"}
        else:
            metadata_filter = None  # Admins can access everything

        if retrieval_mode == "mmr":
            return self.retrieve_mmr(query, metadata_filter=metadata_filter, min_date=min_date)
        elif retrieval_mode == "hybrid":
            return self.retrieve_hybrid(query, metadata_filter=metadata_filter, min_date=min_date)
        else:
            return self.retrieve_top_k(query, metadata_filter=metadata_filter, min_date=min_date)

    def retrieve_with_scores(self, query: str, metadata_filter: dict = None, min_date: str = None) -> list[tuple[Document, float]]:
        """Retrieves documents along with their relevance scores, supporting metadata and date filtering."""
        search_kwargs = {"k": self.top_k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            **search_kwargs
        )
        if min_date:
            results = [(doc, score) for doc, score in results if doc.metadata.get("last_updated", "0000-00-00") >= min_date]
        return results

    def should_refuse(self, scored_results: list[tuple[Document, float]]) -> bool:
        """Checks if the highest retrieved score falls below the refuse threshold."""
        if not scored_results:
            return True
        best_doc, best_score = scored_results[0]
        return best_score < self.refuse_threshold
