from langchain_core.prompts import ChatPromptTemplate
from .retriever import RetrieverService
from .llm_service import LLMService
from .parser import OutputFormatter

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
    """Component to coordinate the retriever, LLM, and prompt structure into an execution pipeline."""

    def __init__(self, retriever_service: RetrieverService, llm_service: LLMService, prompt_template_str: str = DEFAULT_PROMPT_TEMPLATE):
        self.retriever_service = retriever_service
        self.llm = llm_service.get_llm()
        self.prompt = ChatPromptTemplate.from_template(prompt_template_str)
        self.parser = OutputFormatter.get_parser()
        self.chain = self.prompt | self.llm | self.parser

    def answer(self, question: str, retrieval_mode: str = "top_k", metadata_filter: dict = None) -> tuple[str, float, list]:
        """
        Runs the RAG pipeline.
        Returns:
            tuple: (final_answer, best_score, retrieved_docs)
        """
        # 1. Similarity search with relevance scores to check refuse threshold
        scored_results = self.retriever_service.retrieve_with_scores(question)
        if not scored_results:
            return "I don't have enough information in the provided documents.", 0.0, []

        best_doc, best_score = scored_results[0]

        # 2. Check refuse threshold
        if self.retriever_service.should_refuse(scored_results):
            return "I don't have enough information in the provided documents.", best_score, []

        # 3. Choose retrieval mode (MMR vs Standard Top-k vs Hybrid)
        if retrieval_mode == "mmr":
            retrieved_docs = self.retriever_service.retrieve_mmr(question, metadata_filter=metadata_filter)
        elif retrieval_mode == "hybrid":
            retrieved_docs = self.retriever_service.retrieve_hybrid(question, metadata_filter=metadata_filter)
        else:
            retrieved_docs = self.retriever_service.retrieve_top_k(question, metadata_filter=metadata_filter)

        # 4. Format context
        context = OutputFormatter.format_docs(retrieved_docs)

        # 5. Call LLM chain
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })

        return answer, best_score, retrieved_docs
