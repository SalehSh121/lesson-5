import os
from dotenv import load_dotenv

# Import components from our modular package
from rag_system import (
    DocumentLoader,
    ChunkingService,
    EmbeddingService,
    VectorStoreManager,
    RetrieverService,
    LLMService,
    RAGPipeline,
    OutputFormatter
)

# ============================================================
# 1. Load API key
# ============================================================
load_dotenv()

# ============================================================
# 2. Configuration
# ============================================================
CORPUS_DIR = "./data/corpus"
if not os.path.exists(CORPUS_DIR) and os.path.exists("data/corpus"):
    CORPUS_DIR = "data/corpus"

CHROMA_DIR = "./chroma_policy_db"

CHUNK_SIZE = 120 # Number of characters per chunk
CHUNK_OVERLAP = 30 # Number of characters to overlap between chunks

TOP_K = 3 # Number of chunks to return
FETCH_K = 8 # Fetches 8 chunks from the vector store, but only returns the top 3

# Refuse threshold:
# If the best retrieved score is too weak, we refuse to answer.
REFUSE_THRESHOLD = 0.60

# If True, rebuild vector store from scratch.
REBUILD_VECTOR_STORE = True


def main():
    # ============================================================
    # 3. Load Corpus & Dynamic Metadata Tagging
    # ============================================================
    print(f"Loading corpus documents from: {CORPUS_DIR}")
    loader = DocumentLoader()
    documents_with_metadata = loader.load_corpus(CORPUS_DIR)

    # ============================================================
    # 4. Split Document into Chunks
    # ============================================================
    print("Splitting documents into chunks...")
    chunker = ChunkingService(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.split(documents_with_metadata)
    print(f"Number of chunks created: {len(chunks)}")

    # ============================================================
    # 5. Create Embedding Model
    # ============================================================
    print("Initializing embedding service...")
    embedder = EmbeddingService(model="gemini-embedding-001")

    # ============================================================
    # 6. Create or Load Vector Store
    # ============================================================
    print("Setting up vector store...")
    vector_store_manager = VectorStoreManager(persist_directory=CHROMA_DIR, embedding_service=embedder)
    vector_store = vector_store_manager.get_vector_store(chunks=chunks, rebuild=REBUILD_VECTOR_STORE)

    # ============================================================
    # 7. Create LLM Service
    # ============================================================
    print("Initializing LLM service...")
    llm_service = LLMService(model="gemini-2.5-flash", temperature=0.2)

    # ============================================================
    # 8. Create Retriever Service
    # ============================================================
    print("Setting up retriever service...")
    retriever_service = RetrieverService(
        vector_store=vector_store,
        top_k=TOP_K,
        fetch_k=FETCH_K,
        refuse_threshold=REFUSE_THRESHOLD
    )
    # Enable Hybrid (Sparse + Dense) Search with our text chunks
    retriever_service.enable_hybrid_search(chunks)

    # ============================================================
    # 9. Create RAG Pipeline
    # ============================================================
    print("Assembling RAG pipeline...")
    pipeline = RAGPipeline(retriever_service=retriever_service, llm_service=llm_service)

    # ============================================================
    # 10. Interactive Loop
    # ============================================================
    print("\n--- Advanced RAG Strategies Demo ---")
    print("Commands:")
    print("- Type your question normally")
    print("- Type 'mode top_k' to use standard similarity search")
    print("- Type 'mode mmr' to use Maximal Marginal Relevance (diversity)")
    print("- Type 'mode hybrid' to use Hybrid (BM25 + Chroma) search")
    print("- Type 'role student' to set user role to student (restricted access)")
    print("- Type 'role admin' to set user role to admin (full access)")
    print("- Type 'rerank on' to enable keyword re-ranking")
    print("- Type 'rerank off' to disable keyword re-ranking")
    print("- Type 'exit' to quit\n")

    retrieval_mode = "top_k"
    user_role = "student"
    use_reranking = False

    while True:
        try:
            user_input = input(f"[{user_role}][rerank={use_reranking}][{retrieval_mode}] Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        if user_input.lower() == "mode top_k":
            retrieval_mode = "top_k"
            print("Retrieval mode changed to: top_k\n")
            continue

        if user_input.lower() == "mode mmr":
            retrieval_mode = "mmr"
            print("Retrieval mode changed to: mmr\n")
            continue

        if user_input.lower() == "mode hybrid":
            retrieval_mode = "hybrid"
            print("Retrieval mode changed to: hybrid\n")
            continue

        if user_input.lower() == "role student":
            user_role = "student"
            print("User role set to: student (restricted access)\n")
            continue

        if user_input.lower() == "role admin":
            user_role = "admin"
            print("User role set to: admin (full access)\n")
            continue

        if user_input.lower() == "rerank on":
            use_reranking = True
            print("Re-ranking enabled.\n")
            continue

        if user_input.lower() == "rerank off":
            use_reranking = False
            print("Re-ranking disabled.\n")
            continue

        print("\n====================================================")
        print(f"Question: {user_input}")
        print(f"Retrieval mode: {retrieval_mode}")
        print(f"User Role: {user_role}")
        print(f"Reranking Active: {use_reranking}")
        print("====================================================")

        # Run pipeline
        final_answer, best_score, retrieved_docs = pipeline.answer(
            question=user_input, 
            retrieval_mode=retrieval_mode,
            user_role=user_role,
            use_reranking=use_reranking
        )

        print(f"\nBest relevance score: {best_score:.4f}")

        # If retrieved_docs is empty, it means we either didn't retrieve anything or refuse threshold triggered
        if not retrieved_docs:
            print("\n=== Refuse Threshold Triggered or No Context Found ===")
        else:
            print("\n=== Retrieved Context & Sources ===")
            for i, doc in enumerate(retrieved_docs, start=1):
                source = doc.metadata.get("source", "unknown")
                topic = doc.metadata.get("topic", "unknown")
                access_level = doc.metadata.get("access_level", "unknown")
                chunk_index = doc.metadata.get("chunk_index", "unknown")
                print(f"[{i}] File: {source} | Topic: {topic} | Access: {access_level} | Chunk: {chunk_index}")
                print(f"    Content: {doc.page_content.strip()}")

        print("\n=== Final Answer ===")
        print(final_answer)
        print()

if __name__ == "__main__":
    main()