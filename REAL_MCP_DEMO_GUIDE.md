# Real MCP Client Demo - Teaching Guide

## Overview

This demo shows how AI applications **connect to and use external MCP tools** - not building the MCP server, but consuming it.

### Key Insight
```
MCP = Model Context Protocol
- MCP Server exposes tools
- MCP Client connects to server
- AI App uses client to call tools
- AI (LLM) decides which tool to use
```

---

## Files Overview

### 1. `external_mcp_server.py` (Server-side)
This is the **external MCP server** that provides tools.

**What it does:**
- Runs as a separate process
- Listens for client connections
- Exposes tools: `read_file`, `search_files`, `create_github_issue`
- Executes tool calls and returns results

**Key components:**
- `@server.list_tools()` - Advertises available tools
- `@server.call_tool()` - Handles tool execution
- `async with server.stdio()` - Waits for client connections

### 2. `real_mcp_tool_call.py` (Client-side)
This is the **AI application** that connects to the MCP server.

**What it does:**
- Connects to external MCP server
- Discovers available tools
- Gets user request
- Uses LLM to decide which tool to use
- Executes the tool
- Generates final answer

**Key components:**
- `StdioServerParameters` - How to connect to server
- `stdio_client()` - Connect to server
- `session.initialize()` - Start MCP session
- `session.list_tools()` - Discover tools
- `session.call_tool()` - Execute tool

---

## The Complete Flow

```
1. Start MCP Server
   └─> external_mcp_server.py runs
   └─> Waits for client connections

2. Run MCP Client
   └─> real_mcp_tool_call.py connects to server
   └─> Initializes MCP session

3. Client Discovers Tools
   └─> Calls session.list_tools()
   └─> Gets: read_file, search_files, create_github_issue

4. User Request
   └─> User types: "What's in README.md?"

5. LLM Routing
   └─> LLM sees available tools
   └─> LLM decides: "Use read_file with path=README.md"

6. Safety Check
   └─> Is read_file a write operation? No
   └─> Execute immediately (no approval needed)

7. Tool Execution
   └─> Client calls: session.call_tool("read_file", {"path": "README.md"})
   └─> Server receives call
   └─> Server executes tool
   └─> Server returns result

8. Final Answer
   └─> LLM receives tool result
   └─> LLM generates helpful answer
   └─> User sees answer
```

---

## Running the Demo

### Step 1: Install MCP SDK
```bash
python -m pip install -U mcp
```

### Step 2: Terminal 1 - Start the Server
```bash
cd "lesson 5"
python external_mcp_server.py
```

Expected output:
```
============================================================
Simple MCP Server
============================================================

This server provides:
- read_file(path): Read file content
- search_files(query): Search for files
- create_github_issue(title, description): Create GitHub issue

Starting simple MCP server...
Waiting for client connections...
Server ready
```

### Step 3: Terminal 2 - Run the Client
```bash
cd "lesson 5"
python real_mcp_tool_call.py
```

Expected output:
```
============================================================
Real MCP Client Demo
============================================================

✓ Connected to MCP server

✓ Discovered 3 tools from MCP server

=== Available MCP Tools ===
- read_file: Read the content of a file.
- search_files: Search for files by keyword.
- create_github_issue: Create a GitHub issue...

============================================================
Type your requests. Type 'exit' to quit.
============================================================

User: 
```

### Step 4: Try Some Requests

**Example 1: Read a file**
```
User: What's in README.md?

--- Routing Request ---
Selected Tool: read_file
Arguments: {'path': 'README.md'}
Reason: User wants to know the content of README.md

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
This is a simple MCP server demo.
It provides file reading and searching capabilities.

=== Final Answer ===
The README.md file describes this as a simple MCP server demo
that provides file reading and searching capabilities...
```

**Example 2: Search files**
```
User: Find files about memory

--- Routing Request ---
Selected Tool: search_files
Arguments: {'query': 'memory'}
Reason: User wants to find files related to memory

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
Found 1 files:
- memory_lesson.md

=== Final Answer ===
I found one file related to memory: memory_lesson.md...
```

**Example 3: Write operation (requires approval)**
```
User: Create an issue about fixing RAG

--- Routing Request ---
Selected Tool: create_github_issue
Arguments: {'title': 'Fix RAG reranking', 'description': 'The reranker...'}
Reason: User wants to create a GitHub issue

=== Human Approval Required ===
This is a write operation: create_github_issue
Arguments: {'title': 'Fix RAG reranking', ...}
Approve this tool call? (yes/no): yes

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
GitHub issue created successfully!
Title: Fix RAG reranking
Description: ...
Issue ID: GH-5234
```

---

## Key Teaching Points

### 1. ClientSession Concept
```python
async with ClientSession(read, write) as session:
    # This is your connection to the MCP server
    # All tool calls go through this session
```
Students should understand: "ClientSession is like a phone line to the server"

### 2. Tool Discovery
```python
tools_result = await session.list_tools()
for tool in tools_result.tools:
    print(f"- {tool.name}: {tool.description}")
```
Students should understand: "We don't hardcode tools; the server tells us what's available"

### 3. LLM Routing
```python
decision = router_chain.invoke({
    "user_request": user_request,
    "available_tools": tools_description
})
```
Students should understand: "The LLM looks at available tools and chooses the best one"

### 4. Safety First
```python
if tool_needs_approval(tool_name):
    # Ask human before executing write operations
```
Students should understand: "Your app controls what the AI can do - not the other way around"

### 5. Tool Execution
```python
tool_result = await session.call_tool(
    tool_name,
    arguments=arguments
)
```
Students should understand: "The client sends a request to the server, server executes, returns result"

---

## Why This Matters

✅ **Separation of Concerns**
- Server: Manages tools
- Client: Orchestrates tool use
- AI App: Makes decisions

✅ **Scalability**
- Add new tools = just add to server
- No client code changes needed

✅ **Safety**
- Client controls tool access
- Approval for sensitive operations
- Audit trail of tool calls

✅ **Flexibility**
- LLM can dynamically choose tools
- Not hardcoded tool decisions
- Easy to add new use cases

---

## Challenges for Students to Try

1. **Add a new tool to the server**
   - Example: `get_time()` tool
   - Update server with new tool
   - Client automatically discovers it

2. **Handle tool errors gracefully**
   - What if tool doesn't exist?
   - What if tool fails?
   - Add error recovery

3. **Add approval for more tools**
   - Mark `search_files` as requiring approval
   - Test the approval flow

4. **Stream tool results**
   - Some tools might return large data
   - Implement streaming

5. **Tool chaining**
   - Can one tool's output feed into another?
   - Example: Search for file, then read it

---

## Comparison: Simulated vs Real

| Aspect | Simulated (Old) | Real MCP (New) |
|--------|-----------------|----------------|
| Tools | Hardcoded functions | Discovered from server |
| Discovery | Manual TOOLS dict | `list_tools()` from server |
| Execution | Direct function call | `call_tool()` via MCP protocol |
| Async | No | Yes (async/await) |
| Multi-process | No | Yes (separate server process) |
| Scalability | Limited | Infinite (many servers) |
| Real-world | Educational only | Production-ready |

---

## Common Errors & Solutions

**Error: "Connection refused"**
- Ensure server is running in Terminal 1
- Check server port matches client config

**Error: "Unknown tool"**
- Server was restarted and tools changed
- Client needs to re-discover tools
- Call `list_tools()` again

**Error: "Tool execution failed"**
- Tool implementation has a bug
- Check server logs
- Fix tool, restart server

**Error: "Timeout waiting for server"**
- Server might be stuck
- Check if `asyncio.Event().wait()` is blocking
- Restart server

---

## Next Steps

1. ✅ Students understand MCP client-server model
2. ✅ Students see LLM-based tool routing
3. ✅ Students learn safety/approval patterns
4. ✅ Next: Build their own MCP server with custom tools
5. ✅ Advanced: Connect multiple servers to one client
