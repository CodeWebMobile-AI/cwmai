#!/usr/bin/env python3
"""
Simple test for organic problem discovery - Phase 1 only
"""

import asyncio
import json
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.gemini_deep_research import GeminiDeepResearch


async def test_phase1_only():
    """Test just the problem discovery phase."""
    print("Testing Phase 1: Problem Discovery\n")
    
    # Initialize researcher
    researcher = GeminiDeepResearch()
    
    # Test problem discovery
    try:
        problems = await researcher._discover_real_world_problems()
        
        if problems:
            print(f"✅ Discovered {len(problems)} problems!\n")
            
            for i, problem in enumerate(problems[:3], 1):
                print(f"{i}. {problem.get('problem_statement', 'N/A')}")
                print(f"   Severity: {problem.get('severity_score', 0)}/10\n")
        else:
            print("❌ No problems discovered")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up environment variables
    if os.path.exists('.env.local'):
        from dotenv import load_dotenv
        load_dotenv('.env.local')
    
    # Run test
    asyncio.run(test_phase1_only())