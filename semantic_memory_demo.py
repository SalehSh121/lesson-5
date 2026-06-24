import os
import shutil
from datetime import datetime

from dotenv import load_dotenv

from langchain_core.documents import Document
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

CHROMA_DIR = "./chroma_semantic_memory_db"

REBUILD_MEMORY = True
TOP_K = 3


# ============================================================
# 3. Rebuild memory store
# ============================================================

if REBUILD_MEMORY and os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
    print("Old semantic memory store deleted.")


# ============================================================
# 4. Create embedding model
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# 5. Create initial memory notes
# ============================================================
# In a real system, these memories would come from previous conversations.

memory_texts = [
    "The student name is Rami.",
    "Rami is learning Python.",
    "Rami struggles with loops and needs visual tracing exercises.",
    "Rami wants to prepare for programming interviews.",
    "Rami prefers short explanations with examples.",
    "Rami solved 5 loop exercises last week."
]


memory_documents = []

for index, text in enumerate(memory_texts):
    memory_documents.append(
        Document(
            page_content=text,
            metadata={
                "memory_id": index,
                "type": "semantic_memory",
                "created_at": datetime.now().isoformat(timespec="seconds")
            }
        )
    )


# ============================================================
# 6. Store memories in Chroma
# ============================================================

vector_store = Chroma.from_documents(
    documents=memory_documents,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": TOP_K}
)


# ============================================================
# 7. Create LLM and prompt
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


prompt = ChatPromptTemplate.from_template("""
You are an AI study coach.

Use the retrieved memory below to answer the user.

Rules:
- If the answer is in memory, use it.
- If the memory does not contain the answer, say:
  "I do not have enough information in memory."
- Do not invent personal information.
- Keep the answer clear and practical.

Retrieved Memory:
{memory}

User Question:
{question}
""")


parser = StrOutputParser()

chain = prompt | llm | parser


# ============================================================
# 8. Helper function
# ============================================================

def format_memories(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        memory_id = doc.metadata.get("memory_id", "unknown")
        created_at = doc.metadata.get("created_at", "unknown")

        formatted.append(f"""
[Memory {i}]
Memory ID: {memory_id}
Created At: {created_at}

Content:
{doc.page_content}
""")

    return "\n\n".join(formatted)


# ============================================================
# 9. Interactive question loop
# ============================================================

print("\n--- Semantic Memory Demo ---")
print("Ask questions about the student's memory.")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask: ").strip()

    if question.lower() == "exit":
        print("Goodbye.")
        break

    retrieved_memories = retriever.invoke(question)

    memory_context = format_memories(retrieved_memories)

    print("\n=== Retrieved Memories ===")
    print(memory_context)

    answer = chain.invoke({
        "memory": memory_context,
        "question": question
    })

    print("\n=== Answer ===")
    print(answer)
    print()
