#!/usr/bin/env python3
"""Test Gemini JSON parsing improvements"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch


async def test_json_parsing():
    """Test the improved JSON parsing with a simple example"""
    print("🧪 Testing Gemini JSON Parsing Improvements")
    print("=" * 60)
    
    researcher = GeminiDeepResearch()
    
    # Test case 1: Valid JSON array parsing
    test_response_array = {
        'content': '''```json
[
  {
    "problem_id": "P1",
    "problem_title": "Test problem with quotes: \\"example\\"",
    "description": "A test problem that includes special characters.",
    "affected_users": "1000 users",
    "pain_cost": "$500/month",
    "opportunity_score": 8,
    "revenue_potential": "$10K-50K"
  },
  {
    "problem_id": "P2",
    "problem_title": "Another test problem",
    "description": "Second test problem.",
    "affected_users": "5000 users",
    "pain_cost": "$200/month",
    "opportunity_score": 7,
    "revenue_potential": "$5K-20K"
  }
]
```'''
    }
    
    print("\n📝 Test 1: Parsing valid JSON array")
    result = researcher._parse_json_array(test_response_array)
    print(f"Result: {len(result)} items parsed")
    if result:
        print(f"First item: {result[0]['problem_title']}")
    
    # Test case 2: Truncated JSON (simulating token limit)
    test_response_truncated = {
        'content': '''```json
[
  {
    "problem_id": "P1",
    "problem_title": "Test problem",
    "description": "A test problem.",
    "affected_users": "1000 users",
    "pain_cost": "$500/month",
    "opportunity_score": 8,
    "revenue_potential": "$10K-50K"
  },
  {
    "problem_id": "P2",
    "problem_title": "Incomplete prob'''
    }
    
    print("\n📝 Test 2: Parsing truncated JSON")
    result = researcher._parse_json_array(test_response_truncated)
    print(f"Result: {len(result)} items salvaged from truncated JSON")
    
    # Test case 3: JSON with escape issues
    test_response_escapes = {
        'content': '''[
  {
    "problem_id": "P1",
    "problem_title": "Problem with backslash \\\\ and quotes",
    "description": "Has various escape sequences.",
    "affected_users": "1000 users",
    "pain_cost": "$500/month",
    "opportunity_score": 8,
    "revenue_potential": "$10K-50K"
  }
]'''
    }
    
    print("\n📝 Test 3: Parsing JSON with escape sequences")
    result = researcher._parse_json_array(test_response_escapes)
    print(f"Result: {len(result)} items parsed")
    
    # Test case 4: Run actual short research
    print("\n📝 Test 4: Running actual short market research (2 sectors only)")
    researcher.DEFAULT_FOCUS_AREAS = ["Personal productivity", "Health and wellness"]
    
    try:
        opportunities = await researcher.conduct_deep_market_research()
        print(f"\n✅ Found {len(opportunities)} opportunities")
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp.problem_description[:80]}...")
    except Exception as e:
        print(f"❌ Error during research: {e}")


if __name__ == "__main__":
    print("Testing improved JSON parsing for Gemini responses...")
    asyncio.run(test_json_parsing())