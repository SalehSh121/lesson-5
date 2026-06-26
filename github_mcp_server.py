"""
GitHub MCP Server - Simplified Version

This MCP server provides real GitHub tools for external MCP clients.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
from mcp.server.models import InitializationOptions
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
from github import Github, Auth
from github.GithubException import GithubException


load_dotenv()

# ============================================================
# Setup logging to stderr for debugging
# ============================================================

def log(msg):
    """Print to stderr to avoid interfering with MCP protocol."""
    print(msg, file=sys.stderr, flush=True)


# ============================================================
# Initialize GitHub client
# ============================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if GITHUB_TOKEN:
    g = Github(auth=Auth.Token(GITHUB_TOKEN))
    log(f"✓ GitHub client initialized")
else:
    g = None
    log("⚠️  GITHUB_TOKEN not set")


# ============================================================
# Create MCP Server
# ============================================================

server = Server("github-mcp")


# ============================================================
# Tool Implementations
# ============================================================

def list_repositories_impl() -> str:
    """List user repositories."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        user = g.get_user()
        repos = user.get_repos()
        
        result = f"Your GitHub repositories ({repos.totalCount} total):\n\n"
        for i, repo in enumerate(repos[:5], 1):
            result += f"{i}. {repo.full_name}\n"
            result += f"   Description: {repo.description or 'No description'}\n"
            result += f"   Stars: {repo.stargazers_count}\n\n"
        
        if repos.totalCount > 5:
            result += f"... and {repos.totalCount - 5} more repositories"
        
        return result
    except Exception as e:
        return f"Error listing repositories: {str(e)}"


def get_issues_impl(repo_name: str, state: str = "open") -> str:
    """Get issues from a repository."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        repo = g.get_repo(repo_name)
        issues = repo.get_issues(state=state)
        
        if issues.totalCount == 0:
            return f"No {state} issues found in {repo_name}."
        
        result = f"Issues in {repo_name} ({state}): {issues.totalCount} total\n\n"
        for i, issue in enumerate(issues[:5], 1):
            result += f"{i}. #{issue.number}: {issue.title}\n"
            result += f"   Created: {issue.created_at}\n"
            result += f"   Comments: {issue.comments}\n\n"
        
        if issues.totalCount > 5:
            result += f"... and {issues.totalCount - 5} more issues"
        
        return result
    except Exception as e:
        return f"Error getting issues: {str(e)}"


def create_issue_impl(repo_name: str, title: str, body: str) -> str:
    """Create a new GitHub issue."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        repo = g.get_repo(repo_name)
        issue = repo.create_issue(title=title, body=body)
        return f"✓ Issue created!\nRepo: {repo_name}\nIssue #: {issue.number}\nTitle: {issue.title}\nURL: {issue.html_url}"
    except Exception as e:
        return f"Error creating issue: {str(e)}"


def get_repo_info_impl(repo_name: str) -> str:
    """Get repository information."""
    if not g:
        return "Error: GitHub token not configured."
    
    try:
        repo = g.get_repo(repo_name)
        return f"""Repository: {repo.full_name}
Description: {repo.description or 'No description'}
Language: {repo.language or 'Unknown'}
Stars: {repo.stargazers_count}
Forks: {repo.forks_count}
Open Issues: {repo.open_issues_count}
Created: {repo.created_at}
Updated: {repo.updated_at}"""
    except Exception as e:
        return f"Error getting repo info: {str(e)}"


# ============================================================
# Register Tools
# ============================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_repositories",
            description="List the authenticated user's GitHub repositories.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            name="get_issues",
            description="Get issues from a GitHub repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository (owner/repo)"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Issue state"}
                },
                "required": ["repo_name"]
            }
        ),
        Tool(
            name="create_issue",
            description="Create a new GitHub issue.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository (owner/repo)"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body/description"}
                },
                "required": ["repo_name", "title", "body"]
            }
        ),
        Tool(
            name="get_repo_info",
            description="Get information about a GitHub repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository (owner/repo)"}
                },
                "required": ["repo_name"]
            }
        )
    ]


# ============================================================
# Handle Tool Calls
# ============================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool."""
    
    if name == "list_repositories":
        result = list_repositories_impl()
    elif name == "get_issues":
        result = get_issues_impl(
            arguments.get("repo_name", ""),
            arguments.get("state", "open")
        )
    elif name == "create_issue":
        result = create_issue_impl(
            arguments.get("repo_name", ""),
            arguments.get("title", ""),
            arguments.get("body", "")
        )
    elif name == "get_repo_info":
        result = get_repo_info_impl(arguments.get("repo_name", ""))
    else:
        result = f"Unknown tool: {name}"
    
    return [TextContent(type="text", text=result)]


# ============================================================
# Main
# ============================================================

async def main():
    """Run the MCP server."""
    log("GitHub MCP Server starting...")
    
    if GITHUB_TOKEN:
        try:
            user = g.get_user()
            log(f"✓ Authenticated as: {user.login}")
        except Exception as e:
            log(f"✗ Auth failed: {e}")
    
    async with mcp.server.stdio.stdio_server() as (read, write):
        log("✓ MCP Server ready on stdio")
        await server.run(
            read,
            write,
            InitializationOptions(
                server_name="github-mcp",
                server_version="1.0",
                capabilities={}
            )
        )


if __name__ == "__main__":
    log("\n" + "="*60)
    log("GitHub MCP Server")
    log("="*60)
    log("Tools: list_repositories, get_issues, create_issue, get_repo_info")
    log("=" * 60 + "\n")
    
    asyncio.run(main())
