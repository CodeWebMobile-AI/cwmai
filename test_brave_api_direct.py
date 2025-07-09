#!/usr/bin/env python3
"""Test Brave Search API directly"""

import asyncio
import os
import aiohttp
import json

async def test_brave_api():
    """Test Brave Search API directly."""
    
    # Get API key
    api_key = os.getenv("BRAVE_API_KEY", "BSAn2ZCq32LqCmwmmVQwo1VHehKL4Gt")
    print(f"API Key: {api_key[:10]}...")
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": "trending software problems 2024",
        "count": 5
    }
    
    print(f"\nURL: {url}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    
    try:
        # Try with SSL verification disabled
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, headers=headers, params=params) as response:
                print(f"\nStatus: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                text = await response.text()
                print(f"\nResponse: {text[:500]}...")
                
                if response.status == 200:
                    data = json.loads(text)
                    web_results = data.get("web", {}).get("results", [])
                    print(f"\nFound {len(web_results)} results")
                    
                    for i, result in enumerate(web_results[:3]):
                        print(f"\nResult {i+1}:")
                        print(f"  Title: {result.get('title', 'N/A')}")
                        print(f"  URL: {result.get('url', 'N/A')}")
                        print(f"  Description: {result.get('description', 'N/A')[:100]}...")
                
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Brave Search API...")
    asyncio.run(test_brave_api())