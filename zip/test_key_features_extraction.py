#!/usr/bin/env python3
"""
Test script to verify key features are properly extracted from research phases
and passed to project creation.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.intelligent_work_finder import IntelligentWorkFinder
from scripts.market_discovery import MarketDiscovery
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager

async def test_key_features_extraction():
    """Test that key features flow from research to project metadata."""
    
    print("Testing Key Features Extraction Fix")
    print("=" * 60)
    
    # Initialize dependencies
    ai_brain = IntelligentAIBrain()
    state_manager = StateManager()
    
    # Initialize work finder
    work_finder = IntelligentWorkFinder(ai_brain=ai_brain, system_state=state_manager)
    
    # First, let's discover a problem with research
    print("\n1. Running market discovery to get research data...")
    market_discovery = MarketDiscovery()
    problems = await market_discovery.discover_problems(count=1)
    
    if not problems:
        print("❌ No problems discovered")
        return
    
    problem = problems[0]
    print(f"✅ Discovered problem: {problem.description[:100]}...")
    
    # Check if research phases data exists
    research_phases = getattr(problem, '_research_phases', None)
    if research_phases:
        print("\n2. Checking research phases for key features...")
        
        # Check Phase 3
        if 'phase_3_competition_analysis' in research_phases:
            phase3_data = research_phases['phase_3_competition_analysis']
            print(f"   Phase 3 data type: {type(phase3_data)}")
            if isinstance(phase3_data, list) and phase3_data:
                for idx, item in enumerate(phase3_data[:2]):  # Show first 2
                    if isinstance(item, dict) and 'key_features' in item:
                        features = item.get('key_features', [])
                        print(f"   Item {idx} has {len(features)} key features")
                        for f in features[:3]:  # Show first 3
                            print(f"     - {f}")
        
        # Check Phase 5
        if 'phase_5_final_selection' in research_phases:
            phase5_data = research_phases['phase_5_final_selection']
            print(f"\n   Phase 5 data type: {type(phase5_data)}")
            if isinstance(phase5_data, list) and phase5_data:
                for idx, item in enumerate(phase5_data[:1]):  # Show first item
                    if isinstance(item, dict) and 'key_features' in item:
                        features = item.get('key_features', [])
                        print(f"   Selected opportunity has {len(features)} key features")
    else:
        print("❌ No research phases data found")
    
    # Now test the extraction method
    print("\n3. Testing key features extraction...")
    extracted_features = work_finder._extract_key_features_from_research(research_phases)
    print(f"✅ Extracted {len(extracted_features)} features:")
    for feature in extracted_features:
        print(f"   - {feature}")
    
    # Generate project ideas to see if features are included
    print("\n4. Generating project ideas to verify feature inclusion...")
    project_ideas = await work_finder._generate_dynamic_project_ideas()
    
    if project_ideas:
        project = project_ideas[0]
        selected_project = project.get('selected_project', {})
        key_features = selected_project.get('key_features', [])
        
        print(f"\n✅ Project metadata includes {len(key_features)} key features:")
        for feature in key_features:
            print(f"   - {feature}")
        
        if key_features:
            print("\n🎉 SUCCESS: Key features are properly flowing from research to project!")
        else:
            print("\n❌ FAIL: No key features in project metadata")
            print("\nDebugging info:")
            print(f"Selected project keys: {list(selected_project.keys())}")
    else:
        print("❌ No project ideas generated")

if __name__ == "__main__":
    # Ensure we have the Gemini API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Run: export $(cat .env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_key_features_extraction())