#!/usr/bin/env python3
"""Test Brave Search API directly"""

import asyncio
import aiohttp
import json

async def test_brave_api():
    """Test Brave Search API directly"""
    api_key = "BSAn2ZCq32LqCmwmmVQwo1VHehKL4Gt"
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    query = "latest AI developer tools 2024"
    url = f"https://api.search.brave.com/res/v1/web/search?q={query}&count=5"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Brave Search API is working!")
                    print(f"\nResults for '{query}':")
                    
                    for i, result in enumerate(data.get('web', {}).get('results', [])[:3], 1):
                        print(f"\n{i}. {result.get('title', 'No title')}")
                        print(f"   URL: {result.get('url', 'No URL')}")
                        print(f"   {result.get('description', 'No description')[:100]}...")
                else:
                    print(f"❌ API Error: {response.status}")
                    print(await response.text())
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_brave_api())