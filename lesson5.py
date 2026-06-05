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
# Since lesson5.py runs from the 'lesson 5' folder or parent folder,
# we need to ensure the document path points to the correct location.
# In the original file, it was: DOCUMENT_PATH = "company_policy.txt"
# If company_policy.txt is in the lectures root, let's first check if it is in the same directory, 
# otherwise search one directory up to be robust.
DOCUMENT_PATH = "company_policy.txt"
if not os.path.exists(DOCUMENT_PATH) and os.path.exists("../company_policy.txt"):
    DOCUMENT_PATH = "../company_policy.txt"

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
    # 3. Load Document & Metadata Tagging
    # ============================================================
    print(f"Loading document from: {DOCUMENT_PATH}")
    loader = DocumentLoader(DOCUMENT_PATH)
    documents_with_metadata = loader.load_with_metadata(
        document_type="policy",
        department="support",
        access_level="public"
    )

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
    print("\n--- Optimized RAG Demo (Refactored) ---")
    print("Commands:")
    print("- Type your question normally")
    print("- Type 'mode top_k' to use normal top-k retrieval")
    print("- Type 'mode mmr' to use MMR retrieval")
    print("- Type 'mode hybrid' to use Hybrid (BM25 + Chroma) retrieval")
    print("- Type 'filter public' to filter documents with access_level='public'")
    print("- Type 'filter clear' to remove metadata filters")
    print("- Type 'exit' to quit\n")

    retrieval_mode = "top_k"
    current_filter = None

    while True:
        try:
            user_input = input("Ask a question: ").strip()
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

        if user_input.lower() == "filter public":
            current_filter = {"access_level": "public"}
            print("Metadata filter applied: {'access_level': 'public'}\n")
            continue

        if user_input.lower() == "filter clear":
            current_filter = None
            print("Metadata filter cleared.\n")
            continue

        print("\n====================================================")
        print(f"Question: {user_input}")
        print(f"Retrieval mode: {retrieval_mode}")
        print(f"Metadata filter: {current_filter}")
        print("====================================================")

        # Run pipeline
        final_answer, best_score, retrieved_docs = pipeline.answer(
            user_input, 
            retrieval_mode=retrieval_mode,
            metadata_filter=current_filter
        )

        print(f"\nBest relevance score: {best_score:.4f}")

        # If retrieved_docs is empty, it means we either didn't retrieve anything or refuse threshold triggered
        if not retrieved_docs:
            print("\n=== Refuse Threshold Triggered or No Context Found ===")
        else:
            context = OutputFormatter.format_docs(retrieved_docs)
            print("\n=== Retrieved Context ===")
            print(context)

        print("\n=== Final Answer ===")
        print(final_answer)
        print()

if __name__ == "__main__":
    main()