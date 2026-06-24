import os
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

CHROMA_DIR = "./chroma_dynamic_memory_db"
TOP_K = 3


# ============================================================
# 3. Create embedding model
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# 4. Load or create Chroma memory store
# ============================================================

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": TOP_K}
)


# ============================================================
# 5. Create LLM and prompt
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


prompt = ChatPromptTemplate.from_template("""
You are an AI study coach.

Use the retrieved memory below to answer the user.

Rules:
- If relevant memory exists, use it.
- If the answer is not in memory, say:
  "I do not have enough information in memory."
- Do not invent personal information.

Retrieved Memory:
{memory}

User Question:
{question}
""")


parser = StrOutputParser()

chain = prompt | llm | parser


# ============================================================
# 6. Helper function
# ============================================================

def format_memories(docs):
    if not docs:
        return "No relevant memories found."

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


def add_memory(memory_text: str):
    memory_id = datetime.now().isoformat(timespec="seconds")

    doc = Document(
        page_content=memory_text,
        metadata={
            "memory_id": memory_id,
            "type": "semantic_memory",
            "created_at": memory_id
        }
    )

    vector_store.add_documents([doc])

    print("Memory saved.")


# ============================================================
# 7. Interactive loop
# ============================================================

print("\n--- Dynamic Semantic Memory Demo ---")
print("Commands:")
print("- remember: your memory text")
print("- ask any question")
print("- exit\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye.")
        break

    if user_input.lower().startswith("remember:"):
        memory_text = user_input.replace("remember:", "", 1).strip()

        if memory_text:
            add_memory(memory_text)
        else:
            print("Please provide memory text after 'remember:'.")

        continue

    retrieved_memories = retriever.invoke(user_input)

    memory_context = format_memories(retrieved_memories)

    print("\n=== Retrieved Memories ===")
    print(memory_context)

    answer = chain.invoke({
        "memory": memory_context,
        "question": user_input
    })

    print("\n=== Answer ===")
    print(answer)
    print()
