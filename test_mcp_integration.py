"""
Test MCP Integration
Demonstrates how MCPs work with CWMAI
"""

import asyncio
import logging
import os
from datetime import datetime

from scripts.mcp_integration import MCPIntegrationHub
from scripts.work_item_types import WorkItem, TaskPriority
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_mcp_filesystem():
    """Test filesystem MCP operations."""
    print("\n🗂️  Testing Filesystem MCP...")
    
    async with MCPIntegrationHub() as mcp:
        if not mcp.filesystem:
            print("❌ Filesystem MCP not available")
            return
        
        # Test file operations
        test_file = "/tmp/mcp_test.txt"
        test_content = f"MCP Test at {datetime.now()}"
        
        # Write file
        success = await mcp.filesystem.write_file(test_file, test_content)
        print(f"✅ Write file: {success}")
        
        # Read file
        content = await mcp.filesystem.read_file(test_file)
        print(f"✅ Read file: {content[:50]}...")
        
        # List directory
        files = await mcp.filesystem.list_directory("/tmp")
        print(f"✅ Found {len(files)} files in /tmp")
        
        # Clean up
        await mcp.filesystem.delete_file(test_file)
        print("✅ Cleaned up test file")


async def test_mcp_memory():
    """Test memory MCP operations."""
    print("\n🧠 Testing Memory MCP...")
    
    async with MCPIntegrationHub() as mcp:
        if not mcp.memory:
            print("❌ Memory MCP not available")
            return
        
        # Store context
        test_data = {
            "project": "cwmai",
            "timestamp": datetime.now().isoformat(),
            "tasks_completed": 42
        }
        
        success = await mcp.memory.store_context("test_context", test_data)
        print(f"✅ Store context: {success}")
        
        # Retrieve context
        retrieved = await mcp.memory.retrieve_context("test_context")
        print(f"✅ Retrieved: {retrieved}")
        
        # List contexts
        contexts = await mcp.memory.list_contexts()
        print(f"✅ Found {len(contexts)} contexts")
        
        # Clean up
        await mcp.memory.delete_context("test_context")
        print("✅ Cleaned up test context")


async def test_mcp_github():
    """Test GitHub MCP operations."""
    print("\n🐙 Testing GitHub MCP...")
    
    # Check if GitHub token is available
    if not os.getenv('GITHUB_TOKEN'):
        print("⚠️  GITHUB_TOKEN not set, skipping GitHub tests")
        return
    
    async with MCPIntegrationHub() as mcp:
        if not mcp.github:
            print("❌ GitHub MCP not available")
            return
        
        repo = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
        
        # List recent issues
        issues = await mcp.github.list_issues(repo, limit=5)
        print(f"✅ Found {len(issues)} open issues")
        
        for issue in issues[:3]:
            print(f"   - #{issue['number']}: {issue['title']}")
        
        # Search repositories
        search_results = await mcp.github.search_repositories("AI automation", limit=3)
        print(f"\n✅ Found {len(search_results)} repositories matching 'AI automation'")
        
        for repo_info in search_results:
            print(f"   - {repo_info['full_name']}: ⭐ {repo_info.get('stargazers_count', 0)}")


async def test_mcp_github_issue_creator():
    """Test MCP-enabled GitHub issue creator."""
    print("\n📝 Testing MCP GitHub Issue Creator...")
    
    # Create a test work item
    test_work_item = WorkItem(
        id="test-mcp-001",
        title="Test MCP Integration - Demo Issue",
        description=(
            "This is a test issue created to demonstrate MCP integration.\n\n"
            "The MCP (Model Context Protocol) provides standardized access to external services.\n\n"
            "This issue should:\n"
            "- Demonstrate successful GitHub integration via MCP\n"
            "- Show proper formatting and labeling\n"
            "- Include all metadata and context"
        ),
        task_type="TESTING",
        priority=TaskPriority.LOW,
        estimated_cycles=1,
        repository=os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
    )
    
    # Create issue creator
    creator = MCPGitHubIssueCreator()
    
    if not creator.can_create_issues():
        print("⚠️  Cannot create issues - GitHub credentials not configured")
        return
    
    # Execute (create issue)
    result = await creator.execute_work_item(test_work_item)
    
    if result['success']:
        print(f"✅ Created issue #{result.get('issue_number')}")
        print(f"   URL: {result.get('issue_url')}")
        print(f"   Value created: {result.get('value_created')}")
    else:
        print(f"❌ Failed to create issue: {result.get('error')}")
    
    await creator.close()


async def test_mcp_tools_listing():
    """List all available MCP tools."""
    print("\n🛠️  Available MCP Tools:")
    
    async with MCPIntegrationHub() as mcp:
        tools = mcp.list_available_tools()
        
        print(f"\nFound {len(tools)} tools across {len(mcp.client.servers)} servers:\n")
        
        # Group by server
        by_server = {}
        for tool in tools:
            server, name = tool.split(':', 1)
            if server not in by_server:
                by_server[server] = []
            by_server[server].append(name)
        
        for server, tool_names in by_server.items():
            print(f"{server}:")
            for name in sorted(tool_names):
                print(f"  - {name}")


async def main():
    """Run all MCP tests."""
    print("🚀 CWMAI MCP Integration Test Suite")
    print("=" * 50)
    
    # Test tool listing first
    await test_mcp_tools_listing()
    
    # Test individual MCPs
    await test_mcp_filesystem()
    await test_mcp_memory()
    await test_mcp_github()
    
    # Test integrated functionality
    # await test_mcp_github_issue_creator()  # Commented out to avoid creating test issues
    
    print("\n✅ MCP Integration tests completed!")
    print("\nTo use MCPs in production:")
    print("1. Set GITHUB_TOKEN environment variable")
    print("2. Configure MySQL credentials for database MCP")
    print("3. Update components to use MCP integration layer")


if __name__ == "__main__":
    asyncio.run(main())