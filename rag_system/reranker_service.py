from langchain_core.documents import Document

class RerankerService:
    """Component to re-evaluate and sort retrieved documents based on term overlap."""
    
    def __init__(self, top_n: int = 3):
        self.top_n = top_n

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Reranks retrieved candidate documents using keyword-overlap similarity scoring.
        This provides a clear educational demonstration of two-stage retrieval (Retrieval -> Reranking)
        without requiring external network APIs or heavy PyTorch models.
        """
        if not docs:
            return []

        # Extract words from query (ignoring case and short words/punctuation)
        query_words = set(word.strip("?,.!") for word in query.lower().split() if len(word) > 2)
        
        scored_docs = []
        for doc in docs:
            # Score document by number of matching query keywords in its content
            content_words = set(word.strip("?,.!") for word in doc.page_content.lower().split())
            
            # Intersection of keywords
            matching_keywords = query_words.intersection(content_words)
            score = len(matching_keywords)
            
            scored_docs.append((doc, score))
            
        # Sort by overlap score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N results
        reranked_docs = [doc for doc, score in scored_docs[:self.top_n]]
        return reranked_docs
