"""
Real MCP Client Demo: GitHub Integration

This teaches how AI applications connect to GitHub via MCP.

Example: User says "Show me open issues in langchain"
- Client discovers GitHub tools
- LLM routes to: get_issues(repo="langchain-ai/langchain", state="open")
- Client calls MCP server
- Server executes via GitHub API
- Returns issues to user

Key: LLM decides which GitHub tool to use based on user request.
"""

import asyncio
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
# 3. Define write tools that require approval
# ============================================================

WRITE_TOOLS = {
    "create_github_issue",
    "send_message",
    "create_calendar_event",
    "delete_file",
    "update_database"
}


def tool_needs_approval(tool_name: str) -> bool:
    """Check if a tool requires human approval."""
    return tool_name in WRITE_TOOLS


# ============================================================
# 4. Format tools for LLM
# ============================================================

def format_tools_for_llm(tools_list) -> str:
    """Convert MCP tools to readable format for LLM."""
    formatted = ""
    for tool in tools_list:
        formatted += f"\n- {tool.name}"
        formatted += f"\n  Description: {tool.description}"
        
        # Format input schema
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            props = tool.inputSchema.get("properties", {})
            if props:
                formatted += "\n  Parameters:"
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "unknown")
                    param_desc = param_info.get("description", "")
                    formatted += f"\n    - {param_name} ({param_type}): {param_desc}"
        formatted += "\n"
    
    return formatted


# ============================================================
# 5. Tool router prompt (LLM decides which tool to use)
# ============================================================

router_prompt = ChatPromptTemplate.from_template("""
You are a GitHub assistant connected to GitHub MCP tools.

The user request is:
{user_request}

Available GitHub MCP tools:
{available_tools}

Choose the best GitHub tool to handle this request.

Return ONLY valid JSON (no other text):

{{
  "tool_name": "tool_name_here",
  "arguments": {{}},
  "requires_approval": true or false,
  "reason": "brief explanation"
}}

Examples:
- User: "Show my repos" → tool_name: "list_repositories"
- User: "Get issues in facebook/react" → tool_name: "get_issues", arguments: {{"repo_name": "facebook/react"}}
- User: "Create issue in my-repo" → tool_name: "create_issue", requires_approval: true

If user request doesn't match any tool, use tool_name: "no_tool".
""")

json_parser = JsonOutputParser()
router_chain = router_prompt | router_llm | json_parser


# ============================================================
# 6. Final answer prompt
# ============================================================

final_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

User Request:
{user_request}

Selected Tool:
{tool_name}

Tool Result:
{tool_result}

Provide a clear, helpful answer based on the tool result.
""")

text_parser = StrOutputParser()
final_chain = final_prompt | answer_llm | text_parser


# ============================================================
# 7. Main MCP application
# ============================================================

async def main():
    """
    Main application: Connect to MCP server, discover tools, 
    route requests, and execute tools.
    """
    
    print("\n" + "="*60)
    print("GitHub MCP Client Demo")
    print("="*60)
    print("\nThis client connects to GitHub via MCP server.")
    
    # ========================================================
    # Connect to external MCP server (GitHub)
    # ========================================================
    
    # This command starts the GitHub MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["external_mcp_server.py"]
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize MCP session
                await session.initialize()
                
                print("\n✓ Connected to MCP server")
                
                # ========================================================
                # Discover available tools from MCP server
                # ========================================================
                
                tools_result = await session.list_tools()
                available_tools = tools_result.tools # this will return a list of Tool objects
                
                print(f"\n✓ Discovered {len(available_tools)} tools from MCP server")
                print("\n=== Available MCP Tools ===")
                for tool in available_tools:
                    print(f"- {tool.name}: {tool.description}")
                
                # ========================================================
                # Interactive loop: Handle user requests
                # ========================================================
                
                print("\n" + "="*60) # this is a separator line 
                print("Type your requests. Type 'exit' to quit.")
                print("="*60 + "\n")
                
                while True:
                    user_request = input("User: ").strip()
                    
                    if user_request.lower() == "exit":
                        print("Goodbye.")
                        break
                    
                    if not user_request:
                        continue
                    
                    # Format tools for LLM
                    tools_description = format_tools_for_llm(available_tools)
                    
                    # ====================================================
                    # Step 3: LLM chooses a tool
                    # ====================================================
                    
                    print("\n--- Routing Request ---")
                    decision = router_chain.invoke({
                        "user_request": user_request,
                        "available_tools": tools_description
                    })
                    
                    tool_name = decision.get("tool_name")
                    arguments = decision.get("arguments", {})
                    requires_approval = decision.get("requires_approval", False)
                    reason = decision.get("reason", "")
                    
                    print(f"Selected Tool: {tool_name}")
                    print(f"Arguments: {arguments}")
                    print(f"Reason: {reason}")
                    
                    # No suitable tool found
                    if tool_name == "no_tool":
                        print("\nNo suitable tool found for this request.")
                        continue
                    
                    # ====================================================
                    # Step 6: Check approval for write operations
                    # ====================================================
                    
                    if tool_needs_approval(tool_name):
                        print("\n=== Human Approval Required ===")
                        print(f"This is a write operation: {tool_name}")
                        print(f"Arguments: {arguments}")
                        
                        approval = input("Approve this tool call? (yes/no): ").strip().lower()
                        
                        if approval != "yes":
                            print("Tool execution cancelled by human.")
                            print()
                            continue
                    
                    # ====================================================
                    # Step 7: Execute MCP tool
                    # ====================================================
                    
                    try:
                        print("\n--- Executing Tool ---")
                        tool_result = await session.call_tool(
                            tool_name,
                            arguments=arguments
                        )
                        print("✓ Tool executed successfully")
                        
                        # Extract result content
                        result_content = ""
                        if hasattr(tool_result, 'content'):
                            for content in tool_result.content:
                                if hasattr(content, 'text'):
                                    result_content += content.text
                                else:
                                    result_content += str(content)
                        else:
                            result_content = str(tool_result)
                        
                        print(f"\nTool Result:\n{result_content}")
                        
                    except Exception as e:
                        print(f"✗ Tool execution failed: {str(e)}")
                        result_content = f"Error: {str(e)}"
                    
                    # ====================================================
                    # Step 9: LLM generates final answer
                    # ====================================================
                    
                    print("\n--- Generating Answer ---")
                    final_answer = final_chain.invoke({
                        "user_request": user_request,
                        "tool_name": tool_name,
                        "tool_result": result_content
                    })
                    
                    print("\n=== Final Answer ===")
                    print(final_answer)
                    print()
    
    except FileNotFoundError:
        print("\n✗ Error: Could not find external_mcp_server.py")
        print("Make sure an MCP server is running.")
        print("Replace server_params with the correct server command.")


# ============================================================
# 8. Entry point
# ============================================================

if __name__ == "__main__":
    print("\nGitHub MCP Integration Concepts:")
    print("- Connect to GitHub via MCP (not direct API calls)")
    print("- LLM reads available tools and chooses best one")
    print("- Works with real GitHub API through MCP server")
    print("- Safety: Approval for write operations (create_issue)")
    print("- Extensible: Easy to add more GitHub tools")
    
    asyncio.run(main())
