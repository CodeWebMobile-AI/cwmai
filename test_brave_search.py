#!/usr/bin/env python3
"""Test Brave Search MCP integration"""

import asyncio
import logging
from scripts.mcp_integration import MCPIntegrationHub

logging.basicConfig(level=logging.INFO)

async def test_brave_search():
    """Test Brave Search functionality"""
    try:
        print("🔍 Testing Brave Search MCP integration...")
        
        # Initialize MCP hub with brave_search server
        async with MCPIntegrationHub() as mcp:
            # Check if Brave Search is available
            tools = mcp.list_available_tools()
            brave_tools = [t for t in tools if t.startswith('brave_search:')]
            
            if brave_tools:
                print(f"✅ Brave Search tools available: {brave_tools}")
                
                # Test a search
                result = await mcp.client.call_tool("brave_search:search", {
                    "query": "latest AI developer tools 2024",
                    "count": 5
                })
                
                if result:
                    print("\n📊 Search Results:")
                    for item in result[0].get('results', [])[:3]:
                        print(f"\n- Title: {item.get('title', 'N/A')}")
                        print(f"  URL: {item.get('url', 'N/A')}")
                        print(f"  Description: {item.get('description', 'N/A')[:100]}...")
                    
                    print("\n✅ Brave Search is working correctly!")
                else:
                    print("❌ No results returned from Brave Search")
            else:
                print("❌ Brave Search tools not found. Make sure the server is installed.")
                print("\nTo install, run:")
                print("npm install -g @modelcontextprotocol/server-brave-search")
                
    except Exception as e:
        print(f"❌ Error testing Brave Search: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_brave_search())