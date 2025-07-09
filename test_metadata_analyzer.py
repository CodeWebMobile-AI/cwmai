#!/usr/bin/env python3
"""
Test the updated repository analyzer with metadata.json support
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.repository_analyzer import RepositoryAnalyzer


async def test_metadata_based_analysis():
    """Test repository analysis using metadata.json"""
    print("Testing metadata-based repository analysis...\n")
    
    # Initialize analyzer in local mode
    analyzer = RepositoryAnalyzer(storage_mode='local')
    
    # Test repository
    repo_name = "CodeWebMobile-AI/ai-text-summarizer"
    
    print(f"Analyzing {repo_name}...")
    analysis = await analyzer.analyze_repository(repo_name)
    
    # Check if metadata was used
    if analysis.get('metadata_based'):
        print("✅ Successfully used metadata.json for analysis")
        print(f"   Metadata last updated: {analysis.get('metadata_last_updated')}")
    else:
        print("❌ Did not use metadata.json")
        
    # Display key information
    print(f"\nRepository: {analysis.get('repository')}")
    print(f"Health Score: {analysis.get('health_metrics', {}).get('health_score', 'N/A')}")
    print(f"Total Files: {analysis.get('code_analysis', {}).get('total_files', 0)}")
    print(f"Total Lines: {analysis.get('code_analysis', {}).get('total_lines', 0)}")
    print(f"Has Tests: {analysis.get('code_analysis', {}).get('test_coverage') == 'has_tests'}")
    
    # Show needs
    needs = analysis.get('specific_needs', [])
    if needs:
        print(f"\nIdentified Needs ({len(needs)}):")
        for need in needs[:3]:
            print(f"  - [{need.get('priority')}] {need.get('description')}")
    
    # Show opportunities  
    opportunities = analysis.get('improvement_opportunities', [])
    if opportunities:
        print(f"\nImprovement Opportunities ({len(opportunities)}):")
        for opp in opportunities:
            print(f"  - {opp.get('description')} (impact: {opp.get('impact')})")
            
    print("\n" + "="*60 + "\n")
    
    # Test force refresh
    print("Testing force refresh (direct analysis)...")
    analysis2 = await analyzer.analyze_repository(repo_name, force_refresh=True)
    
    if analysis2.get('metadata_based'):
        print("❌ Still using metadata despite force_refresh=True")
    else:
        print("✅ Performed direct analysis as requested")
        
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_metadata_based_analysis())