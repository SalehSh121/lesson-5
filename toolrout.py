from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# ============================================================
# 1. Load API key
# ============================================================

load_dotenv()


# ============================================================
# 2. Create LLM
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
# 3. Define tools as normal Python functions
# ============================================================

def calculator(expression: str) -> str:
    """
    Simple calculator tool.

    WARNING:
    This is for teaching only.
    In production, do not use eval directly.
    """
    try:
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return "Invalid expression."

        result = eval(expression)
        return str(result)

    except Exception as error:
        return f"Calculation error: {error}"


def policy_search(question: str) -> str:
    """
    Simulated policy search tool.
    In a real system, this could be a RAG retriever.
    """
    policy_text = """
Students can cancel a session up to 24 hours before the session starts.

If a student misses a session without cancelling in advance, the session is counted as used.

Refund requests are reviewed manually by the support team.

If a teacher misses a session, the student may receive a replacement session or refund after admin review.
"""
    return policy_text


def general_answer(question: str) -> str:
    """
    General answer tool.
    """
    prompt = ChatPromptTemplate.from_template("""
You are a helpful teaching assistant.

Answer the question clearly and briefly.

Question:
{question}
""")

    parser = StrOutputParser()
    chain = prompt | answer_llm | parser

    return chain.invoke({"question": question})


# ============================================================
# 4. Tool selection prompt
# ============================================================

tool_selector_prompt = ChatPromptTemplate.from_template("""
You are an AI tool router.

Your job is to select the best tool for the user request.

Available tools:

1. calculator
Use this tool for math, arithmetic, discounts, percentages, or numeric calculations.

2. policy_search
Use this tool for questions about sessions, cancellation, missed sessions, refunds, teachers, or booking policy.

3. general_answer
Use this tool for general explanations that do not require a tool.

Return only valid JSON in this format:

{{
  "tool": "calculator | policy_search | general_answer",
  "tool_input": "input to send to the selected tool",
  "reason": "short reason"
}}

User request:
{question}
""")

json_parser = JsonOutputParser()

tool_selector_chain = tool_selector_prompt | router_llm | json_parser


# ============================================================
# 5. Final answer prompt
# ============================================================

final_answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use the tool result to answer the user clearly.

User Question:
{question}

Selected Tool:
{tool}

Tool Result:
{tool_result}

Final Answer:
""")

text_parser = StrOutputParser()

final_answer_chain = final_answer_prompt | answer_llm | text_parser


# ============================================================
# 6. Execute selected tool
# ============================================================

def execute_tool(tool_name: str, tool_input: str) -> str:
    if tool_name == "calculator":
        return calculator(tool_input)

    if tool_name == "policy_search":
        return policy_search(tool_input)

    if tool_name == "general_answer":
        return general_answer(tool_input)

    return "Unknown tool."


# ============================================================
# 7. Interactive loop
# ============================================================

print("\n--- Tool Calling Demo ---")
print("Ask a question.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("User: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye.")
        break

    decision = tool_selector_chain.invoke({
        "question": user_input
    })

    selected_tool = decision["tool"]
    tool_input = decision["tool_input"]
    reason = decision["reason"]

    print("\n=== Tool Decision ===")
    print(f"Selected Tool: {selected_tool}")
    print(f"Tool Input: {tool_input}")
    print(f"Reason: {reason}")

    tool_result = execute_tool(selected_tool, tool_input)

    print("\n=== Tool Result ===")
    print(tool_result)

    final_answer = final_answer_chain.invoke({
        "question": user_input,
        "tool": selected_tool,
        "tool_result": tool_result
    })

    print("\n=== Final Answer ===")
    print(final_answer)
    print()
