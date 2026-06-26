"""
GITHUB_TOKEN = ? 
GitHub MCP Server

This MCP server connects to GitHub and exposes real GitHub tools.

Tools provided:
- list_repositories: List user's GitHub repositories
- get_issues: Get issues from a repository
- create_issue: Create a new GitHub issue
- get_repo_info: Get repository information

Setup:
  1. Install: pip install PyGithub
  2. Set GITHUB_TOKEN environment variable
  3. Run this server
"""

import asyncio
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import Tool, TextContent
from github import Github
from github.GithubException import GithubException


load_dotenv()

# ============================================================
# 1. Create MCP Server
# ============================================================

server = Server("github-mcp-server") # this creates an MCP server instance with the name "github-mcp-server". The server will listen for incoming connections from clients and handle tool calls.


# ============================================================
# 2. Initialize GitHub client
# ============================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Warning: GITHUB_TOKEN not set. Set it to use real GitHub API.")
    print("Export: export GITHUB_TOKEN=your_token_here")
    g = None
else:
    g = Github(GITHUB_TOKEN) # this initializes the GitHub client using the provided token. The client will be used to interact with the GitHub API for various operations like listing repositories, getting issues, creating issues, etc.


# ============================================================
# 3. Tool implementations
# ============================================================
# below are the actual implementations of the GitHub tools. by default, they are synchronous functions. In a real server, you might want to make them async if they involve network calls.
def list_repositories_impl() -> str:
    """List all repositories for the authenticated user."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        user = g.get_user()
        repos = user.get_repos()
        
        if repos.totalCount == 0:
            return "No repositories found."
        
        result = f"Found {repos.totalCount} repositories:\n\n"
        for repo in repos[:10]:  # Limit to 10 for demo
            result += f"- {repo.full_name}\n"
            result += f"  Description: {repo.description or 'No description'}\n"
            result += f"  Stars: {repo.stargazers_count}\n\n"
        
        return result
    except Exception as e:
        return f"Error listing repositories: {str(e)}"


def get_issues_impl(repo_name: str, state: str = "open") -> str:
    """Get issues from a specific repository."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        repo = g.get_repo(repo_name)
        issues = repo.get_issues(state=state)
        
        if issues.totalCount == 0:
            return f"No {state} issues found in {repo_name}."
        
        result = f"Found {issues.totalCount} {state} issues in {repo_name}:\n\n"
        for issue in issues[:5]:  # Limit to 5 for demo
            result += f"#{issue.number}: {issue.title}\n"
            result += f"  Created: {issue.created_at}\n"
            result += f"  Comments: {issue.comments}\n\n"
        
        return result
    except GithubException.NotFound:
        return f"Repository '{repo_name}' not found."
    except Exception as e:
        return f"Error getting issues: {str(e)}"


def create_issue_impl(repo_name: str, title: str, body: str) -> str:
    """Create a new GitHub issue."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        repo = g.get_repo(repo_name)
        issue = repo.create_issue(title=title, body=body)
        
        return f"""Issue created successfully!

Repository: {repo_name}
Issue #: {issue.number}
Title: {issue.title}
URL: {issue.html_url}
"""
    except GithubException.NotFound:
        return f"Repository '{repo_name}' not found."
    except Exception as e:
        return f"Error creating issue: {str(e)}"


def get_repo_info_impl(repo_name: str) -> str:
    """Get detailed information about a repository."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        repo = g.get_repo(repo_name)
        
        return f"""Repository: {repo.full_name}

Description: {repo.description or "No description"}
URL: {repo.html_url}
Language: {repo.language or "Unknown"}
Stars: {repo.stargazers_count}
Forks: {repo.forks_count}
Watchers: {repo.watchers_count}
Open Issues: {repo.open_issues_count}
Created: {repo.created_at}
Updated: {repo.updated_at}
"""
    except GithubException.NotFound:
        return f"Repository '{repo_name}' not found."
    except Exception as e:
        return f"Error getting repo info: {str(e)}"


# ============================================================
# 4. Define tools for the server
# ============================================================
# the below tools are registered with the MCP server and can be called by clients. Each tool has a name, description, and input schema. all the tools are taken from the implementations abov from the GitHub API.
@server.list_tools() # we take this decorator to define the tools that the server exposes. The client can call list_tools() to discover these tools and their input schemas.
async def list_tools():
    """List all available GitHub tools."""
    return [
        Tool(
            name="list_repositories",
            description="List all repositories for the authenticated GitHub user.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_issues",
            description="Get issues from a specific GitHub repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string",
                        "description": "Repository name in format owner/repo (e.g., torvalds/linux)"
                    },
                    "state": {
                        "type": "string",
                        "description": "Issue state: 'open', 'closed', or 'all' (default: open)",
                        "enum": ["open", "closed", "all"]
                    }
                },
                "required": ["repo_name"]
            }
        ),
        Tool(
            name="create_issue",
            description="Create a new GitHub issue in a repository (write operation - requires approval).",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string",
                        "description": "Repository name in format owner/repo"
                    },
                    "title": {
                        "type": "string",
                        "description": "Issue title"
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue description/body"
                    }
                },
                "required": ["repo_name", "title", "body"]
            }
        ),
        Tool(
            name="get_repo_info",
            description="Get detailed information about a GitHub repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string",
                        "description": "Repository name in format owner/repo"
                    }
                },
                "required": ["repo_name"]
            }
        )
    ]


# ============================================================
# 5. Handle tool calls
# ============================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute a GitHub tool and return result."""
    
    if name == "list_repositories":
        result = list_repositories_impl()
        return [TextContent(type="text", text=result)]
    
    elif name == "get_issues":
        repo_name = arguments.get("repo_name", "")
        state = arguments.get("state", "open")
        result = get_issues_impl(repo_name, state)
        return [TextContent(type="text", text=result)]
    
    elif name == "create_issue":
        repo_name = arguments.get("repo_name", "")
        title = arguments.get("title", "")
        body = arguments.get("body", "")
        result = create_issue_impl(repo_name, title, body)
        return [TextContent(type="text", text=result)]
    
    elif name == "get_repo_info":
        repo_name = arguments.get("repo_name", "")
        result = get_repo_info_impl(repo_name)
        return [TextContent(type="text", text=result)]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ============================================================
# 6. Run server
# ============================================================

async def main():
    print("Starting GitHub MCP server...")
    
    if not GITHUB_TOKEN:
        print("\n⚠️  Warning: GITHUB_TOKEN not configured!")
        print("Please set: export GITHUB_TOKEN=your_github_token")
    else:
        try:
            user = g.get_user()
            print(f"\n✓ GitHub authenticated as: {user.login}")
        except Exception as e:
            print(f"\n✗ GitHub authentication failed: {str(e)}")
    
    print("\nWaiting for client connections...")
    
    async with server.stdio():
        print("✓ GitHub MCP Server ready")
        await asyncio.Event().wait()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GitHub MCP Server")
    print("="*60)
    print("\nAvailable GitHub tools:")
    print("- list_repositories(): List your GitHub repos")
    print("- get_issues(repo_name, state): Get issues from repo")
    print("- create_issue(repo_name, title, body): Create issue")
    print("- get_repo_info(repo_name): Get repo details")
    print("\nSetup:")
    print("1. pip install PyGithub")
    print("2. export GITHUB_TOKEN=your_token_here")
    print("3. Run this server\n")
    
    asyncio.run(main())
