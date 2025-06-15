"""
Basic MCP Integration Test
Tests only the available MCP servers
"""

import asyncio
import logging
import os
from datetime import datetime

from scripts.mcp_integration import MCPIntegrationHub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def test_basic_mcp():
    """Test basic MCP functionality with available servers."""
    print("\n🚀 Testing Basic MCP Integration...")
    
    try:
        # Initialize with only available servers
        mcp = MCPIntegrationHub()
        await mcp.initialize(servers=['github', 'filesystem', 'memory'])
        
        print("\n✅ MCP Hub initialized successfully!")
        
        # Test GitHub MCP
        if mcp.github:
            print("\n🐙 Testing GitHub MCP...")
            try:
                # List issues in the current repo
                repo = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
                issues = await mcp.github.list_issues(repo, limit=3)
                print(f"✅ Found {len(issues)} open issues")
                for issue in issues[:3]:
                    print(f"   - #{issue['number']}: {issue['title']}")
            except Exception as e:
                print(f"❌ GitHub test failed: {e}")
        
        # Test Filesystem MCP
        if mcp.filesystem:
            print("\n🗂️ Testing Filesystem MCP...")
            try:
                # Write a test file
                test_file = "/tmp/mcp_test.txt"
                test_content = f"MCP Test at {datetime.now()}"
                
                success = await mcp.filesystem.write_file(test_file, test_content)
                print(f"✅ Write file: {success}")
                
                # Read it back
                content = await mcp.filesystem.read_file(test_file)
                print(f"✅ Read file: {content[:50]}...")
                
                # Clean up
                await mcp.filesystem.delete_file(test_file)
                print("✅ Cleaned up test file")
            except Exception as e:
                print(f"❌ Filesystem test failed: {e}")
        
        # Test Memory MCP
        if mcp.memory:
            print("\n🧠 Testing Memory MCP...")
            try:
                # Store something
                test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
                success = await mcp.memory.store_context("test_key", test_data)
                print(f"✅ Store context: {success}")
                
                # Retrieve it
                retrieved = await mcp.memory.retrieve_context("test_key")
                print(f"✅ Retrieved: {retrieved}")
                
                # Clean up
                await mcp.memory.delete_context("test_key")
                print("✅ Cleaned up test context")
            except Exception as e:
                print(f"❌ Memory test failed: {e}")
        
        # List available tools
        tools = mcp.list_available_tools()
        print(f"\n🛠️ Available tools: {len(tools)}")
        for tool in sorted(tools)[:10]:
            print(f"   - {tool}")
        
        await mcp.close()
        print("\n✅ All basic tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run basic MCP tests."""
    print("🚀 Basic MCP Integration Test")
    print("=" * 50)
    
    await test_basic_mcp()
    
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())