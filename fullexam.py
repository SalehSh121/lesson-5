import os
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. Load API key
# ============================================================

load_dotenv()


# ============================================================
# 2. Configuration
# ============================================================

DOCUMENT_PATH = "company_policy.txt"
CHROMA_DIR = "./chroma_policy_db"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

TOP_K = 3
FETCH_K = 8

# Refuse threshold:
# If the best retrieved score is too weak, we refuse to answer.
# Important:
# Chroma's relevance scores may behave differently depending on metric/model.
# In a real system, test and calibrate this value.
REFUSE_THRESHOLD = 0.30

# If True, rebuild vector store from scratch.
# Set to True when you change document content, chunk size, overlap, or embedding model.
REBUILD_VECTOR_STORE = True


# ============================================================
# 3. Optional: Clear old Chroma DB
# ============================================================

if REBUILD_VECTOR_STORE and os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print("Old Chroma vector store deleted.")


# ============================================================
# 4. Load document
# ============================================================

loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
documents = loader.load()


# ============================================================
# 5. Add metadata to documents before chunking
# ============================================================
# Metadata helps us track source, document type, and access level.
# Later, metadata can be used for filtering, citations, and debugging.

documents_with_metadata = []

for doc in documents:
    documents_with_metadata.append(
        Document(
            page_content=doc.page_content,
            metadata={
                "source": DOCUMENT_PATH,
                "document_type": "policy",
                "department": "support",
                "access_level": "public"
            }
        )
    )


# ============================================================
# 6. Split document into chunks
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(documents_with_metadata)


# Add chunk index to metadata
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = index


print(f"Number of chunks created: {len(chunks)}")


# ============================================================
# 7. Create embedding model
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# 8. Create or load Chroma vector store
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)


# ============================================================
# 9. Create LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# ============================================================
# 10. Create strict grounded prompt
# ============================================================

prompt = ChatPromptTemplate.from_template("""
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
""")


parser = StrOutputParser()

chain = prompt | llm | parser


# ============================================================
# 11. Helper function: format retrieved documents
# ============================================================

def format_docs(docs):
    formatted_chunks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "unknown")
        department = doc.metadata.get("department", "unknown")
        access_level = doc.metadata.get("access_level", "unknown")

        formatted_chunks.append(
            f"""
[Chunk {i}]
Source: {source}
Chunk Index: {chunk_index}
Department: {department}
Access Level: {access_level}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted_chunks)


# ============================================================
# 12. Normal top-k retrieval
# ============================================================

def retrieve_top_k(question):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    docs = retriever.invoke(question)
    return docs


# ============================================================
# 13. MMR retrieval
# ============================================================

def retrieve_mmr(question):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": FETCH_K
        }
    )

    docs = retriever.invoke(question)
    return docs


# ============================================================
# 14. Retrieval with scores for refuse threshold
# ============================================================
# This is used for deciding whether the system should answer or refuse.
# We retrieve documents with relevance scores.

def retrieve_with_scores(question):
    results = vector_store.similarity_search_with_relevance_scores(
        query=question,
        k=TOP_K
    )

    return results


# ============================================================
# 15. Answer function
# ============================================================

def answer_question(question, retrieval_mode="top_k"):
    print("\n====================================================")
    print(f"Question: {question}")
    print(f"Retrieval mode: {retrieval_mode}")
    print("====================================================")

    # First, get results with scores for refuse threshold
    scored_results = retrieve_with_scores(question)

    if not scored_results:
        return "I don't have enough information in the provided documents."

    best_doc, best_score = scored_results[0]

    print(f"\nBest relevance score: {best_score:.4f}")

    # Refuse if score is too low
    if best_score < REFUSE_THRESHOLD:
        print("\n=== Refuse Threshold Triggered ===")
        return "I don't have enough information in the provided documents."

    # Choose retrieval mode
    if retrieval_mode == "mmr":
        retrieved_docs = retrieve_mmr(question)
    else:
        retrieved_docs = retrieve_top_k(question)

    context = format_docs(retrieved_docs)

    print("\n=== Retrieved Context ===")
    print(context)

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer


# ============================================================
# 16. Interactive loop
# ============================================================

print("\n--- Optimized RAG Demo ---")
print("Commands:")
print("- Type your question normally")
print("- Type 'mode top_k' to use normal top-k retrieval")
print("- Type 'mode mmr' to use MMR retrieval")
print("- Type 'exit' to quit\n")

retrieval_mode = "top_k"

while True:
    user_input = input("Ask a question: ").strip()

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

    final_answer = answer_question(user_input, retrieval_mode=retrieval_mode)

    print("\n=== Final Answer ===")
    print(final_answer)
    print()