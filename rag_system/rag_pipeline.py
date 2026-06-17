from langchain_core.prompts import ChatPromptTemplate
from .retriever import RetrieverService
from .llm_service import LLMService
from .parser import OutputFormatter
from .reranker_service import RerankerService

DEFAULT_PROMPT_TEMPLATE = """
You are a helpful support advisor.

You must answer using ONLY the retrieved context below.

Rules:
- If the answer exists in the context, answer clearly and concisely.
- If the answer is not found in the context, say exactly:
  "I don't have enough information in the provided documents."
- Do not guess.
- Do not use general knowledge.
- Do not invent prices, policies, or conditions.

Retrieved Context:
{context}

User Question:
{question}
"""

class RAGPipeline:
    """Component to coordinate the retriever, reranker, LLM, and prompt structure into an execution pipeline."""

    def __init__(self, retriever_service: RetrieverService, llm_service: LLMService, prompt_template_str: str = DEFAULT_PROMPT_TEMPLATE, reranker_service: RerankerService = None):
        self.retriever_service = retriever_service
        self.llm = llm_service.get_llm()
        self.prompt = ChatPromptTemplate.from_template(prompt_template_str)
        self.parser = OutputFormatter.get_parser()
        self.chain = self.prompt | self.llm | self.parser
        self.reranker = reranker_service or RerankerService(top_n=2)  # Re-ranks and selects top 2

    def answer(self, question: str, retrieval_mode: str = "top_k", user_role: str = "student", use_reranking: bool = False, min_date: str = None) -> tuple[str, float, list]:
        """
        Runs the RAG pipeline.
        Returns:
            tuple: (final_answer, best_score, retrieved_docs)
        """
        # Determine metadata filters based on user role
        if user_role == "student":
            metadata_filter = {"access_level": "student_visible"}
        else:
            metadata_filter = None  # Admins can access all files

        # 1. Similarity search with relevance scores to check refuse threshold
        scored_results = self.retriever_service.retrieve_with_scores(question, metadata_filter=metadata_filter, min_date=min_date)
        if not scored_results:
            return "I don't have enough information in the provided documents.", 0.0, []

        best_doc, best_score = scored_results[0]

        # 2. Check refuse threshold
        if self.retriever_service.should_refuse(scored_results):
            return "I don't have enough information in the provided documents.", best_score, []

        # 3. Retrieve chunks using selected search strategy
        if retrieval_mode == "mmr":
            retrieved_docs = self.retriever_service.retrieve_mmr(question, metadata_filter=metadata_filter, min_date=min_date)
        elif retrieval_mode == "hybrid":
            retrieved_docs = self.retriever_service.retrieve_hybrid(question, metadata_filter=metadata_filter, min_date=min_date)
        else:
            retrieved_docs = self.retriever_service.retrieve_top_k(question, metadata_filter=metadata_filter, min_date=min_date)

        # 4. Optional: Rerank step
        if use_reranking and retrieved_docs:
            retrieved_docs = self.reranker.rerank(question, retrieved_docs)

        # 5. Format context
        context = OutputFormatter.format_docs(retrieved_docs)

        # 6. Call LLM chain
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })

        return answer, best_score, retrieved_docs
