#!/usr/bin/env python3
"""
Test using Brave Search data with hardcoded approach structure.

This combines:
1. Real-time search data from Brave
2. The structured problem-solving approach from hardcoded method
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import sys
sys.path.append('/Users/fortressarigo/PhpstormProjects/cwmai')

from scripts.ai_brain_factory import AIBrainFactory
from scripts.brave_search_integration import BraveSearchClient


class BraveWithHardcodedApproach:
    """Use Brave Search data with hardcoded problem-solving structure."""
    
    def __init__(self):
        self.ai_brain = AIBrainFactory.create_for_production()
        api_key = os.getenv("BRAVE_API_KEY", "BSAn2ZCq32LqCmwmmVQwo1VHehKL4Gt")
        self.brave_client = BraveSearchClient(api_key)
        
    async def research_with_structured_approach(self) -> Dict[str, Any]:
        """Use Brave Search but apply hardcoded approach's structure."""
        print("\n🔍 Using Brave Search with Hardcoded Approach Structure...")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Search for SPECIFIC problems people are discussing
            specific_problem_queries = [
                "my biggest problem with software 2025",
                "I hate when apps",
                "frustrated with technology",
                "wish there was an app for",
                "struggling with daily",
                "annoying software problems",
                "why is it so hard to",
                "tired of manually"
            ]
            
            all_problems = []
            
            print("\n🔍 Searching for specific problems people mention...")
            for query in specific_problem_queries:
                print(f"  → Searching: {query}")
                try:
                    results = await self.brave_client.search(query, count=3)
                    web_results = results.get("web", {}).get("results", [])
                    
                    for result in web_results:
                        all_problems.append({
                            "query": query,
                            "title": result.get("title", ""),
                            "snippet": result.get("description", ""),
                            "url": result.get("url", "")
                        })
                    
                    await asyncio.sleep(0.3)  # Rate limiting
                    
                except Exception as e:
                    print(f"    ❌ Search failed: {e}")
            
            print(f"\n✓ Found {len(all_problems)} problem discussions")
            
            # Step 2: Apply hardcoded approach's structured analysis
            structured_prompt = f"""
            Based on these REAL problems people are discussing online, identify 5 specific opportunities.
            
            Real Problems Found:
            {json.dumps(all_problems, indent=2)}
            
            For each opportunity, think like the hardcoded approach - be SPECIFIC and FOCUSED:
            
            1. Identify a SINGLE, CONCRETE problem (not a category of problems)
            2. Define exactly WHO has this problem (specific audience, not "everyone")
            3. What solutions exist and why they fail (be specific)
            4. Your proposed solution (a specific product, not a platform)
            5. How to make money (clear model)
            6. Market size (realistic numbers)
            7. Competition level
            
            Focus on:
            - SPECIFIC products (like "smart water bottle") not platforms
            - NARROW problems (like "subscription tracking") not broad categories
            - CLEAR audiences (like "landlords screening tenants") not "professionals"
            - Solutions you can visualize building, not vague concepts
            
            Avoid:
            - "Platforms" that do everything
            - "AI-powered" as the main feature
            - Broad categories like "productivity" or "wellness"
            - Competing with giants like Google or Microsoft
            
            Return exactly 5 opportunities in this format:
            [
                {{
                    "problem_description": "Specific problem description",
                    "affected_audience": "Exact target users",
                    "current_solutions": "What exists and specific failures",
                    "opportunity": "Your specific solution",
                    "monetization_model": "How to charge for it",
                    "market_size": "Realistic estimate",
                    "competition_level": "Low/Medium/High"
                }}
            ]
            """
            
            response = await self.ai_brain.generate_enhanced_response(structured_prompt, model='gemini')
            
            # Parse response
            opportunities = {"opportunities": [], "error": "Failed to parse"}
            if response and response.get('content'):
                import re
                json_match = re.search(r'\[.*\]', response['content'], re.DOTALL)
                if json_match:
                    opportunities = {
                        "opportunities": json.loads(json_match.group()),
                        "research_timestamp": datetime.now().isoformat(),
                        "source": "brave_search_with_hardcoded_structure",
                        "problems_analyzed": len(all_problems)
                    }
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✓ Research completed in {duration:.2f}s")
            print(f"✓ Found {len(opportunities.get('opportunities', []))} opportunities")
            
            # Print the opportunities
            print("\n📋 Project Ideas (Brave + Hardcoded Structure):")
            print("=" * 80)
            for i, opp in enumerate(opportunities.get('opportunities', []), 1):
                print(f"\n{i}. {opp.get('opportunity', 'N/A')}")
                print(f"   Problem: {opp.get('problem_description', 'N/A')[:100]}...")
                print(f"   Audience: {opp.get('affected_audience', 'N/A')}")
                print(f"   Monetization: {opp.get('monetization_model', 'N/A')}")
            
            return opportunities
            
        except Exception as e:
            print(f"❌ Research failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


async def main():
    """Run the hybrid approach test."""
    print("🚀 Testing Brave Search with Hardcoded Approach Structure")
    print("This searches for real problems but applies focused analysis\n")
    
    researcher = BraveWithHardcodedApproach()
    results = await researcher.research_with_structured_approach()
    
    # Save results
    filename = f"brave_hardcoded_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Full results saved to: {filename}")


if __name__ == "__main__":
    # Set up environment
    os.system('export $(cat .env.local | grep -v "^#" | xargs)')
    asyncio.run(main())