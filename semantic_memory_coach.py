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

CHROMA_DIR = "./chroma_coach_memory_db"
TOP_K = 3
REFUSE_THRESHOLD = 0.60  # Relevance score threshold for refusal


# ============================================================
# 3. Create embedding model
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# ============================================================
# 4. Create/Load persistent vector store
# ============================================================

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)


# ============================================================
# 5. Create LLM and Prompt
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
- Keep the answer clear, supportive, and direct.

Retrieved Memory:
{memory}

User Question:
{question}
""")

parser = StrOutputParser()
chain = prompt | llm | parser


# ============================================================
# 6. Helper functions
# ============================================================

def format_memories(docs_with_scores):
    if not docs_with_scores:
        return "No relevant memories found."
    
    formatted = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        memory_id = doc.metadata.get("memory_id", "unknown")
        memory_type = doc.metadata.get("type", "general")
        created_at = doc.metadata.get("created_at", "unknown")
        
        formatted.append(f"""[Memory {i}] (Type: {memory_type}, Score: {score:.4f})
Memory ID: {memory_id}
Created At: {created_at}
Content: {doc.page_content}""")
        
    return "\n\n".join(formatted)


def add_memory(memory_text: str, memory_type: str):
    memory_id = datetime.now().isoformat(timespec="seconds")
    
    doc = Document(
        page_content=memory_text,
        metadata={
            "memory_id": memory_id,
            "type": memory_type,
            "created_at": memory_id
        }
    )
    
    vector_store.add_documents([doc])
    print(f"Memory saved under type: '{memory_type}'.")


# ============================================================
# 7. Interactive loop
# ============================================================

def main():
    print("\n--- AI Study Coach with Advanced Semantic Memory ---")
    print("Commands:")
    print("- 'remember <type>: <text>' (Types: profile, goal, weakness, progress, preference)")
    print("  Example: 'remember goal: I want to learn Python for web development.'")
    print("- 'remember: <text>' (Saves as general memory)")
    print("  Example: 'remember: I prefer night study sessions.'")
    print("- 'filter type <type>' (Filter retrieval to specific type, or 'clear' to search all)")
    print("  Example: 'filter type weakness'")
    print("- 'clear memory' (Completely clear the database)")
    print("- Ask any question to retrieve memories and get an answer")
    print("- 'exit' to quit\n")
    
    active_filter = None
    
    while True:
        try:
            # Display current active filter in prompt
            prompt_indicator = f"[filter={active_filter or 'None'}] Ask: "
            user_input = input(prompt_indicator).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == "exit":
            print("Goodbye.")
            break
            
        if user_input.lower() == "clear memory":
            try:
                # Retrieve all document IDs and delete them to avoid file-lock errors on Windows
                all_docs = vector_store.get()
                if all_docs and "ids" in all_docs and all_docs["ids"]:
                    vector_store.delete(ids=all_docs["ids"])
                print("Memory database cleared.\n")
            except Exception as e:
                print(f"Error clearing memory database: {e}\n")
            continue
            
        if user_input.lower().startswith("filter type "):
            filter_val = user_input[12:].strip().lower()
            if filter_val == "clear" or filter_val == "none":
                active_filter = None
                print("Retrieval filter cleared (searching all memory types).\n")
            elif filter_val in ["profile", "goal", "weakness", "progress", "preference", "general"]:
                active_filter = filter_val
                print(f"Retrieval filter set to type: '{active_filter}'.\n")
            else:
                print(f"Unknown type '{filter_val}'. Valid types: profile, goal, weakness, progress, preference, general, clear.\n")
            continue
            
        # Parse remember commands
        if user_input.lower().startswith("remember:"):
            memory_text = user_input[9:].strip()
            if memory_text:
                add_memory(memory_text, "general")
            else:
                print("Please provide memory text after 'remember:'.")
            print()
            continue
            
        if user_input.lower().startswith("remember "):
            content = user_input[9:].strip()
            if ":" in content:
                type_part, text_part = content.split(":", 1)
                type_part = type_part.strip().lower()
                text_part = text_part.strip()
                
                if type_part in ["profile", "goal", "weakness", "progress", "preference", "general"]:
                    if text_part:
                        add_memory(text_part, type_part)
                    else:
                        print(f"Please provide memory text after '{type_part}:'.")
                else:
                    # Treat entire content as a general memory
                    if content:
                        add_memory(content, "general")
            else:
                if content:
                    add_memory(content, "general")
                else:
                    print("Please provide memory text.")
            print()
            continue
            
        # QA execution
        db_empty = False
        try:
            col_count = vector_store._collection.count()
            if col_count == 0:
                db_empty = True
        except Exception:
            db_empty = True
            
        if db_empty:
            print("\n=== Retrieved Memories ===")
            print("No memories stored yet.")
            print("\n=== Answer ===")
            print("I do not have enough information in memory.\n")
            continue
            
        # Similarity search with relevance scores
        search_kwargs = {}
        if active_filter:
            search_kwargs["filter"] = {"type": active_filter}
            
        try:
            results = vector_store.similarity_search_with_relevance_scores(
                query=user_input,
                k=TOP_K,
                **search_kwargs
            )
        except Exception as e:
            print(f"Error retrieving from database: {e}")
            results = []
            
        valid_results = []
        best_score = 0.0
        if results:
            best_score = results[0][1]
            valid_results = [(doc, score) for doc, score in results if score >= REFUSE_THRESHOLD]
            
        print("\n====================================================")
        print(f"Question: {user_input}")
        print(f"Active Filter: {active_filter}")
        print(f"Best Similarity Score: {best_score:.4f}")
        print("====================================================")
        
        if not valid_results:
            print("\n=== Refuse Threshold Triggered or No Match Found ===")
            print("No relevant memories match the query with high enough confidence.")
            print("\n=== Final Answer ===")
            print("I do not have enough information in memory.\n")
            continue
            
        memory_context = format_memories(valid_results)
        print("\n=== Retrieved Memories ===")
        print(memory_context)
        
        answer = chain.invoke({
            "memory": memory_context,
            "question": user_input
        })
        
        print("\n=== Final Answer ===")
        print(answer)
        print()

if __name__ == "__main__":
    main()
