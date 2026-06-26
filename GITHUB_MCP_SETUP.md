# GitHub MCP Demo - Setup & Usage

## Quick Setup

### 1. Install Dependencies
```bash
pip install mcp PyGithub
```

### 2. Get GitHub Token
1. Go to https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Check scope: `repo` (full control of private repositories)
4. Copy the token

### 3. Set Environment Variable
```bash
# Linux/Mac
export GITHUB_TOKEN=ghp_your_token_here

# Windows PowerShell
$env:GITHUB_TOKEN="ghp_your_token_here"

# Windows Command Prompt
set GITHUB_TOKEN=ghp_your_token_here
```

### 4. Create .env File (Alternative)
Create `.env` in lesson 5 folder:
```
GITHUB_TOKEN=ghp_your_token_here
```

---

## Running the Demo

### Terminal 1: Start GitHub MCP Server
```bash
cd "lesson 5"
python external_mcp_server.py
```

Expected output:
```
============================================================
GitHub MCP Server
============================================================

Available GitHub tools:
- list_repositories(): List your GitHub repos
- get_issues(repo_name, state): Get issues from repo
- create_issue(repo_name, title, body): Create issue
- get_repo_info(repo_name): Get repo details

Setup:
1. pip install PyGithub
2. export GITHUB_TOKEN=your_token_here
3. Run this server

Starting GitHub MCP server...

✓ GitHub authenticated as: yourusername

Waiting for client connections...
✓ GitHub MCP Server ready
```

### Terminal 2: Run Client
```bash
cd "lesson 5"
python real_mcp_tool_call.py
```

Expected output:
```
============================================================
GitHub MCP Client Demo
============================================================

This client connects to GitHub via MCP server.

✓ Connected to MCP server

✓ Discovered 4 tools from MCP server

=== Available MCP Tools ===
- list_repositories: List all repositories for the authenticated GitHub user.
- get_issues: Get issues from a specific GitHub repository.
- create_issue: Create a new GitHub issue...
- get_repo_info: Get detailed information about a GitHub repository.

============================================================
Type your requests. Type 'exit' to quit.
============================================================

User:
```

---

## Example Interactions

### Example 1: List Your Repositories
```
User: Show my repositories

--- Routing Request ---
Selected Tool: list_repositories
Reason: User wants to see their GitHub repositories

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
Found 5 repositories:

- username/awesome-project
  Description: An awesome AI project
  Stars: 42

- username/learning-mcp
  Description: Learning MCP servers
  Stars: 5

... more repos ...

=== Final Answer ===
You have 5 repositories. Here are your recent ones:
1. awesome-project (42 stars) - An awesome AI project
2. learning-mcp (5 stars) - Learning MCP servers
...
```

### Example 2: Get Issues from a Repository
```
User: What are the open issues in facebook/react?

--- Routing Request ---
Selected Tool: get_issues
Arguments: {'repo_name': 'facebook/react', 'state': 'open'}
Reason: User wants to see open issues in a specific repository

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
Found 543 open issues in facebook/react:

#12345: Bug: useState with multiple effects
  Created: 2024-06-20
  Comments: 5

#12346: Enhancement: Improve error messages
  Created: 2024-06-18
  Comments: 12

... more issues ...

=== Final Answer ===
facebook/react has 543 open issues. Here are some recent ones:
1. Bug: useState with multiple effects (#12345) - 5 comments
2. Enhancement: Improve error messages (#12346) - 12 comments
...
```

### Example 3: Create a New Issue (Write Operation)
```
User: Create an issue in my-repo about fixing the bug

--- Routing Request ---
Selected Tool: create_issue
Arguments: {'repo_name': 'username/my-repo', 'title': 'Fix the bug', ...}
Reason: User wants to create a new issue

=== Human Approval Required ===
This is a write operation: create_issue
Arguments: {'repo_name': 'username/my-repo', ...}
Approve this tool call? (yes/no): yes

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
Issue created successfully!

Repository: username/my-repo
Issue #: 42
Title: Fix the bug
URL: https://github.com/username/my-repo/issues/42

=== Final Answer ===
Great! I've created issue #42 in your-repo about fixing the bug.
You can view it at: https://github.com/username/my-repo/issues/42
```

### Example 4: Get Repository Information
```
User: Tell me about the langchain repository

--- Routing Request ---
Selected Tool: get_repo_info
Arguments: {'repo_name': 'langchain-ai/langchain'}
Reason: User wants detailed information about a specific repository

--- Executing Tool ---
✓ Tool executed successfully

Tool Result:
Repository: langchain-ai/langchain

Description: Building applications with LLMs through composability and modularity
URL: https://github.com/langchain-ai/langchain
Language: Python
Stars: 75000
Forks: 12000
Watchers: 2000
Open Issues: 145
Created: 2022-10-17
Updated: 2024-06-26

=== Final Answer ===
LangChain is a popular Python project with 75,000 stars!
It's for building AI applications with LLMs...
```

---

## Teaching Points

### 1. Real-World Integration
Unlike simulated examples, this connects to **real GitHub API** through MCP.

### 2. LLM Routing
The LLM intelligently chooses which GitHub tool to use:
- "Show my repos" → `list_repositories`
- "Check issues in X" → `get_issues`
- "Tell me about X" → `get_repo_info`
- "Create issue" → `create_issue` (with approval)

### 3. Safety Matters
Write operations require human approval:
```python
WRITE_TOOLS = {"create_issue"}

if tool_needs_approval(tool_name):
    # Ask human before executing
```

### 4. Extensibility
Easy to add more GitHub tools:
- `create_pull_request()`
- `add_star_to_repo()`
- `list_collaborators()`
- `get_commits()`

Just add a new function and register it in `@server.call_tool()`.

---

## Troubleshooting

### "Connection refused"
- Ensure server is running in Terminal 1
- Both terminals must be in same directory

### "GITHUB_TOKEN not configured"
- Set environment variable before running
- Or create `.env` file with `GITHUB_TOKEN=...`

### "Repository not found"
- Use correct format: `owner/repo`
- Example: `torvalds/linux` not just `linux`
- Check repository is public or you have access

### "Authentication failed"
- Token might be expired
- Regenerate token on GitHub
- Token must have `repo` scope

### "Rate limit exceeded"
- GitHub API has limits (60 requests/hour for unauthenticated)
- With valid token: 5000 requests/hour
- Wait before making more requests

---

## Next Steps for Students

1. ✅ Run the demo locally
2. ✅ Try different user requests
3. ✅ Add approval for `get_issues` tool
4. ✅ Add a new GitHub tool (e.g., `list_branches`)
5. ✅ Create MCP server for another API (Twitter, Slack, etc.)
6. ✅ Connect multiple MCP servers to one client
