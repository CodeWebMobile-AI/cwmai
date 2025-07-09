#!/usr/bin/env python3
"""
Test comparison between hardcoded research areas and dynamic Brave Search approach.

This test will:
1. Run the current hardcoded research method
2. Run the new dynamic Brave Search MCP method
3. Compare the quality and relevance of results
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import sys
sys.path.append('/Users/fortressarigo/PhpstormProjects/cwmai')

from scripts.project_creator import ProjectCreator
from scripts.ai_brain_factory import AIBrainFactory


class ResearchComparison:
    """Compare hardcoded vs dynamic research approaches."""
    
    def __init__(self):
        self.ai_brain = AIBrainFactory.create_for_production()
        self.project_creator = ProjectCreator(ai_brain=self.ai_brain)
        
    async def run_hardcoded_research(self) -> Dict[str, Any]:
        """Run the current hardcoded research method."""
        print("\n🔍 Running HARDCODED research method...")
        print("=" * 80)
        
        start_time = datetime.now()
        
        # Use the existing method
        try:
            results = await self.project_creator._research_market_opportunities()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Print results summary
            print(f"\n✓ Hardcoded research completed in {duration:.2f}s")
            print(f"✓ Found {len(results.get('opportunities', []))} opportunities")
            
            return {
                'method': 'hardcoded',
                'duration_seconds': duration,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Hardcoded research failed: {e}")
            return {
                'method': 'hardcoded',
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    async def run_dynamic_research(self) -> Dict[str, Any]:
        """Run the new dynamic Brave Search research method using direct API."""
        print("\n🌐 Running DYNAMIC Brave Search research method...")
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Import Brave Search client
            from scripts.brave_search_integration import BraveSearchClient
            
            # Initialize Brave Search with API key
            api_key = os.getenv("BRAVE_API_KEY", "BSAn2ZCq32LqCmwmmVQwo1VHehKL4Gt")
            brave_client = BraveSearchClient(api_key)
            
            # Get current context
            current_date = datetime.now()
            
            print(f"📅 Researching for date: {current_date.strftime('%Y-%m-%d')}")
            print(f"🔑 Using Brave Search API directly")
            
            # Step 1: Discover what's trending now
            discovery_queries = [
                f"trending software problems {current_date.strftime('%B %Y')}",
                f"what apps are people looking for {current_date.strftime('%Y')}",
                f"current technology market gaps {current_date.strftime('%Y')}",
                f"software opportunities {current_date.strftime('%B %Y')}",
                f"problems that need solving app {current_date.strftime('%Y')}"
            ]
            
            all_search_results = []
            
            print("\n🔍 Searching for current market trends...")
            for query in discovery_queries:
                print(f"  → Searching: {query}")
                try:
                    # Call Brave Search API directly
                    results = await brave_client.search(query, count=5)
                    
                    # Extract web results
                    web_results = results.get("web", {}).get("results", [])
                    print(f"    ✓ Found {len(web_results)} results")
                    
                    for result in web_results:
                        all_search_results.append({
                            "query": query,
                            "title": result.get("title", ""),
                            "snippet": result.get("description", ""),
                            "url": result.get("url", ""),
                            "relevance": f"Found via: {query}"
                        })
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    print(f"    ❌ Search failed: {e}")
            
            print(f"\n✓ Collected {len(all_search_results)} total search results")
            
            # Print sample of actual results
            if all_search_results:
                print("\n📊 Sample of actual Brave Search results:")
                for i, result in enumerate(all_search_results[:3]):
                    print(f"\n  Result {i+1}:")
                    print(f"    Title: {result['title'][:80]}...")
                    print(f"    Snippet: {result['snippet'][:100]}...")
            
            # Step 2: Analyze discoveries with AI
            analysis_prompt = f"""
            Analyze these REAL search results from Brave Search to identify market opportunities.
            
            Actual Search Results from Brave (current date: {current_date.strftime('%Y-%m-%d')}):
            {json.dumps(all_search_results, indent=2)}
            
            Based on what people are ACTUALLY discussing and searching for RIGHT NOW:
            1. Identify the top 5 problems that need software solutions
            2. For each problem, assess market demand and competition
            3. Suggest specific project ideas that address these real needs
            4. Base your analysis on the actual search results provided above
            
            Return in the same format as the hardcoded method for fair comparison:
            {{
                "opportunities": [
                    {{
                        "problem_description": "...",
                        "affected_audience": "...",
                        "current_solutions": "...",
                        "opportunity": "...",
                        "monetization_model": "...",
                        "market_size": "...",
                        "competition_level": "Low/Medium/High"
                    }}
                ],
                "research_timestamp": "{current_date.isoformat()}",
                "source": "brave_search_api_realtime",
                "total_search_results_analyzed": {len(all_search_results)}
            }}
            """
            
            response = await self.ai_brain.generate_enhanced_response(analysis_prompt)
            
            # Parse response
            opportunities = {"opportunities": [], "error": "Failed to parse AI analysis"}
            if response and response.get('content'):
                import re
                json_match = re.search(r'\{.*\}', response['content'], re.DOTALL)
                if json_match:
                    opportunities = json.loads(json_match.group())
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✓ Dynamic research completed in {duration:.2f}s")
            print(f"✓ Found {len(opportunities.get('opportunities', []))} opportunities")
            print(f"✓ Based on {len(all_search_results)} real search results")
            
            return {
                'method': 'dynamic_brave_search_api',
                'duration_seconds': duration,
                'results': opportunities,
                'search_results_count': len(all_search_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Dynamic research failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'method': 'dynamic_brave_search_mcp',
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    async def compare_results(self, hardcoded: Dict[str, Any], dynamic: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the two research approaches."""
        print("\n📊 Comparing research results...")
        print("=" * 80)
        
        comparison_prompt = f"""
        Compare these two market research approaches and their results:
        
        HARDCODED APPROACH:
        {json.dumps(hardcoded, indent=2)}
        
        DYNAMIC BRAVE SEARCH MCP APPROACH:
        {json.dumps(dynamic, indent=2)}
        
        Analyze and compare:
        1. Quality of opportunities discovered
        2. Relevance to current market (how timely/current)
        3. Specificity and actionability
        4. Evidence of real demand
        5. Innovation and uniqueness
        6. Business viability
        
        Provide:
        - Strengths of each approach
        - Weaknesses of each approach
        - Which found better opportunities and why
        - Recommendation for which to use
        
        Format as structured analysis.
        """
        
        response = await self.ai_brain.generate_enhanced_response(comparison_prompt)
        
        return {
            'comparison_analysis': response.get('content', ''),
            'hardcoded_duration': hardcoded.get('duration_seconds', 0),
            'dynamic_duration': dynamic.get('duration_seconds', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, hardcoded: Dict, dynamic: Dict, comparison: Dict):
        """Save results to file for review."""
        results = {
            'test_date': datetime.now().isoformat(),
            'hardcoded_results': hardcoded,
            'dynamic_results': dynamic,
            'comparison': comparison
        }
        
        filename = f"research_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Also create a human-readable summary
        summary_filename = f"research_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_filename, 'w') as f:
            f.write("# Research Method Comparison\n\n")
            f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Hardcoded Method\n")
            f.write(f"- Duration: {hardcoded.get('duration_seconds', 0):.2f} seconds\n")
            if 'error' in hardcoded:
                f.write(f"- Error: {hardcoded['error']}\n")
            else:
                opps = hardcoded.get('results', {}).get('opportunities', [])
                f.write(f"- Opportunities found: {len(opps)}\n")
                f.write("- Research areas: Predefined (8 categories)\n")
                f.write("- Sample opportunities:\n")
                for i, opp in enumerate(opps[:3]):
                    f.write(f"  {i+1}. {opp.get('problem_description', 'N/A')}\n")
                f.write("\n")
            
            f.write("## Dynamic Brave Search MCP Method\n")
            f.write(f"- Duration: {dynamic.get('duration_seconds', 0):.2f} seconds\n")
            if 'error' in dynamic:
                f.write(f"- Error: {dynamic['error']}\n")
            else:
                opps = dynamic.get('results', {}).get('opportunities', [])
                f.write(f"- Opportunities found: {len(opps)}\n")
                f.write("- Research areas: Discovered from real-time search\n")
                f.write("- Sample opportunities:\n")
                for i, opp in enumerate(opps[:3]):
                    f.write(f"  {i+1}. {opp.get('problem_description', 'N/A')}\n")
                f.write("\n")
            
            f.write("## AI Comparison Analysis\n")
            f.write(comparison.get('comparison_analysis', 'No analysis available'))
        
        print(f"📄 Summary saved to: {summary_filename}")


async def main():
    """Run the comparison test."""
    print("🚀 Starting Research Method Comparison Test")
    print("This will compare hardcoded vs dynamic Brave Search MCP approaches\n")
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    comparison = ResearchComparison()
    
    # Run both methods
    hardcoded_results = await comparison.run_hardcoded_research()
    dynamic_results = await comparison.run_dynamic_research()
    
    # Compare results
    comparison_analysis = await comparison.compare_results(hardcoded_results, dynamic_results)
    
    # Save results
    comparison.save_results(hardcoded_results, dynamic_results, comparison_analysis)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    print(f"\nHardcoded Method:")
    print(f"  - Duration: {hardcoded_results.get('duration_seconds', 0):.2f}s")
    if 'error' not in hardcoded_results:
        opps = hardcoded_results.get('results', {}).get('opportunities', [])
        print(f"  - Opportunities: {len(opps)}")
        if opps:
            print("  - Sample:", opps[0].get('problem_description', 'N/A')[:100] + "...")
    else:
        print(f"  - Error: {hardcoded_results['error']}")
    
    print(f"\nDynamic Brave Search MCP Method:")
    print(f"  - Duration: {dynamic_results.get('duration_seconds', 0):.2f}s")
    if 'error' not in dynamic_results:
        opps = dynamic_results.get('results', {}).get('opportunities', [])
        print(f"  - Opportunities: {len(opps)}")
        if opps:
            print("  - Sample:", opps[0].get('problem_description', 'N/A')[:100] + "...")
    else:
        print(f"  - Error: {dynamic_results['error']}")
    
    print("\n✅ Test complete! Check the output files for detailed results.")


if __name__ == "__main__":
    # Set up environment
    print("Setting up environment variables...")
    os.system('export $(cat .env.local | grep -v "^#" | xargs)')
    
    # Run the test
    asyncio.run(main())