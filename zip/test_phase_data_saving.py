#!/usr/bin/env python3
"""Test script to verify phase data saving in gemini_deep_research.py"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import from scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.gemini_deep_research import GeminiDeepResearchOptimized
from scripts.ai_brain import IntelligentAIBrain

async def test_phase_data_saving():
    """Test that phase data is being saved correctly."""
    print("Testing phase data saving...")
    
    # Initialize the research system
    ai_brain = IntelligentAIBrain()
    researcher = GeminiDeepResearchOptimized(ai_brain)
    
    # Run a quick research session
    print("\nRunning market research (this may take a minute)...")
    opportunities = await researcher.research_market_opportunities(count=1)
    
    # Check if phase data was saved
    print("\n" + "="*60)
    print("PHASE DATA SAVED:")
    print("="*60)
    
    for phase_name, phase_data in researcher.research_phases_data.items():
        if phase_data:
            print(f"\n{phase_name}:")
            if isinstance(phase_data, list):
                print(f"  - {len(phase_data)} items saved")
                if phase_data:
                    # Show first item as example
                    print(f"  - Example: {json.dumps(phase_data[0], indent=4)[:200]}...")
            else:
                print(f"  - Data: {json.dumps(phase_data, indent=4)[:200]}...")
        else:
            print(f"\n{phase_name}: No data")
    
    # Check the latest saved research file
    research_dir = Path("research_knowledge")
    if research_dir.exists():
        files = sorted(research_dir.glob("research_*.json"), reverse=True)
        if files and not str(files[0]).endswith('summary.json'):
            latest_file = files[0]
            print(f"\n\nLatest research file: {latest_file}")
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
                
            # Check if research_phases has data
            phases = data.get('research_phases', {})
            has_data = any(phases.values())
            
            print(f"Phase data in file: {'YES' if has_data else 'NO'}")
            
            if has_data:
                for phase, content in phases.items():
                    if content:
                        if isinstance(content, list):
                            print(f"  - {phase}: {len(content)} items")
                        else:
                            print(f"  - {phase}: Has data")

if __name__ == "__main__":
    asyncio.run(test_phase_data_saving())