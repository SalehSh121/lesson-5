from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# ============================================================
# 1. Load API key
# ============================================================

load_dotenv()


# ============================================================
# 2. Create LLMs
# ============================================================

router_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

answer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


# ============================================================
# 3. Simulated external MCP tools
# ============================================================
# In a real system, these would be exposed by external MCP servers.
# Here we simulate the same idea using Python functions.

def mcp_search_files(query: str) -> str:
    return """
Found files:
1. README.md
2. advanced_rag_notes.md
3. memory_lesson.md
"""


def mcp_read_file(file_name: str) -> str:
    files = {
        "README.md": """
This project is an AI learning assistant.
It supports RAG, memory, tool calling, and human approval.
The goal is to help students learn AI engineering through practical examples.
""",
        "advanced_rag_notes.md": """
Advanced RAG includes metadata filtering, MMR retrieval, reranking, grounding, and refusal.
""",
        "memory_lesson.md": """
Advanced memory includes short-term memory, long-term memory, summary memory, entity memory, episodic memory, and semantic memory.
"""
    }

    return files.get(file_name, "File not found.")


def mcp_create_github_issue(title: str, description: str) -> str:
    return f"""
GitHub issue created successfully.

Title:
{title}

Description:
{description}
"""


# ============================================================
# 4. Tool registry
# ============================================================

TOOLS = {
    "search_files": {
        "description": "Search project files by keyword.",
        "type": "read",
        "requires_approval": False
    },
    "read_file": {
        "description": "Read the content of a selected file.",
        "type": "read",
        "requires_approval": False
    },
    "create_github_issue": {
        "description": "Create a GitHub issue.",
        "type": "write",
        "requires_approval": True
    },
    "general_answer": {
        "description": "Answer general questions without external tools.",
        "type": "read",
        "requires_approval": False
    }
}


# ============================================================
# 5. Tool router prompt
# ============================================================

router_prompt = ChatPromptTemplate.from_template("""
You are an AI agent connected to external MCP tools.

Choose the best tool for the user request.

Available tools:

1. search_files
Use when the user wants to find files or search project documents.
The tool_input should contain a query.

2. read_file
Use when the user asks to read or summarize a specific file.
The tool_input should contain the file_name.

3. create_github_issue
Use when the user asks to create a GitHub issue or bug report.
This is a write tool and requires approval.
The tool_input should include a title and description.

4. general_answer
Use when no external tool is needed.

Return only valid JSON in this format:

{{
  "tool": "search_files | read_file | create_github_issue | general_answer",
  "tool_input": {{
    "query": "",
    "file_name": "",
    "title": "",
    "description": "",
    "question": ""
  }},
  "reason": "short reason"
}}

User request:
{user_request}
""")

json_parser = JsonOutputParser()
router_chain = router_prompt | router_llm | json_parser


# ============================================================
# 6. Final answer prompt
# ============================================================

final_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use the external tool result to answer the user.

User Request:
{user_request}

Selected Tool:
{tool}

Tool Result:
{tool_result}

Final Answer:
""")

text_parser = StrOutputParser()
final_chain = final_prompt | answer_llm | text_parser


# ============================================================
# 7. Tool executor
# ============================================================

def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_files":
        return mcp_search_files(tool_input.get("query", ""))

    if tool_name == "read_file":
        return mcp_read_file(tool_input.get("file_name", ""))

    if tool_name == "create_github_issue":
        return mcp_create_github_issue(
            title=tool_input.get("title", ""),
            description=tool_input.get("description", "")
        )

    if tool_name == "general_answer":
        return tool_input.get("question", "")

    return "Unknown tool."


# ============================================================
# 8. Approval check
# ============================================================

def requires_approval(tool_name: str) -> bool:
    tool_config = TOOLS.get(tool_name, {})
    return tool_config.get("requires_approval", False)


# ============================================================
# 9. Interactive loop
# ============================================================

print("\n--- Simulated External MCP Tools Demo ---")
print("Type 'exit' to quit.\n")

while True:
    user_request = input("User: ").strip()

    if user_request.lower() == "exit":
        print("Goodbye.")
        break

    decision = router_chain.invoke({
        "user_request": user_request
    })

    selected_tool = decision["tool"]
    tool_input = decision["tool_input"]
    reason = decision["reason"]

    print("\n=== MCP Tool Decision ===")
    print(f"Selected Tool: {selected_tool}")
    print(f"Tool Input: {tool_input}")
    print(f"Reason: {reason}")

    if requires_approval(selected_tool):
        print("\n=== Human Approval Required ===")
        print(f"The agent wants to run this write tool: {selected_tool}")
        print(f"Tool input: {tool_input}")

        approval = input("Approve tool execution? yes/no: ").strip().lower()

        if approval != "yes":
            print("\nTool execution cancelled.")
            continue

    tool_result = execute_tool(selected_tool, tool_input)

    print("\n=== MCP Tool Result ===")
    print(tool_result)

    final_answer = final_chain.invoke({
        "user_request": user_request,
        "tool": selected_tool,
        "tool_result": tool_result
    })

    print("\n=== Final Answer ===")
    print(final_answer)
    print()