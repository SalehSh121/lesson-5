# GitHub MCP Demo - Quick Start

## 1️⃣ Install Dependencies
```bash
pip install mcp PyGithub python-dotenv langchain-google-genai
```

## 2️⃣ Get GitHub Token
1. Go to: https://github.com/settings/tokens/new
2. Select scope: `repo`
3. Click "Generate token"
4. Copy the token

## 3️⃣ Set Token
Create `.env` file in this folder:
```
GITHUB_TOKEN=ghp_your_token_here
GOOGLE_API_KEY=your_google_api_key
```

## 4️⃣ Run Server (Terminal 1)
```bash
python external_mcp_server.py
```

Should show:
```
✓ GitHub authenticated as: your_username
✓ GitHub MCP Server ready
```

## 5️⃣ Run Client (Terminal 2)
```bash
python real_mcp_tool_call.py
```

## 6️⃣ Try It!

```
User: show my repositories
User: get issues in facebook/react
User: tell me about the nodejs/node repository
```

---

## What Happens

```
You type: "Get issues in facebook/react"
    ↓
LLM sees available tools:
  - list_repositories
  - get_issues
  - create_issue
  - get_repo_info
    ↓
LLM chooses: get_issues with repo_name="facebook/react"
    ↓
Client calls: session.call_tool("get_issues", ...)
    ↓
Server connects to GitHub API
    ↓
Returns real issues from that repo
    ↓
LLM generates helpful answer for you
```

---

## Creating an Issue (Write Operation)

```
User: create an issue in my-repo about the bug

=== Human Approval Required ===
This is a write operation: create_issue
Approve this tool call? (yes/no): yes

[Issue created on GitHub]
```

---

## Key Idea

✅ **NOT** simulated - connects to real GitHub  
✅ **LLM decides** which tool to use  
✅ **Human approves** write operations  
✅ **Easy to extend** - add more GitHub tools
