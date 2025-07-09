#!/usr/bin/env python3
"""
Monitor for schema errors in the system and provide real-time feedback.
"""

import asyncio
import subprocess
import time
import re
from datetime import datetime
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_schema_directly():
    """Test the schema directly to ensure it's working."""
    print("\n🧪 Testing schema directly...")
    
    try:
        from scripts.gemini_deep_research import GeminiDeepResearch
        from scripts.ai_brain import IntelligentAIBrain
        
        # Initialize components
        ai_brain = IntelligentAIBrain()
        research = GeminiDeepResearch(ai_brain)
        
        # Create test data
        test_opportunities = [{
            'opportunity_score': 85,
            'problem_description': 'Test problem',
            'affected_audience': 'Test audience',
            'solution_name': 'Test solution',
            'key_features': ['Feature 1', 'Feature 2'],
            'revenue_potential': '$5000/month',
            'market_size': '$1B',
            'competition_level': 'medium',
            'technical_score': 8,
            'market_score': 8,
            'economics_score': 8,
            'mvp_score': 8
        }]
        
        # Test the final ranking
        print("   Testing _filter_and_rank method...")
        result = await research._filter_and_rank(
            opportunities=test_opportunities,
            count=1,
            existing_portfolio={}
        )
        
        if result:
            print("   ✅ Schema test passed! Final ranking works correctly.")
            return True
        else:
            print("   ⚠️ No results returned, but no schema error.")
            return True
            
    except Exception as e:
        if "too many states" in str(e):
            print(f"   ❌ SCHEMA ERROR DETECTED: {e}")
            return False
        else:
            print(f"   ❌ Other error: {e}")
            return False

def check_running_logs():
    """Check if there are any recent log files with schema errors."""
    print("\n📋 Checking recent logs for schema errors...")
    
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for log_file in log_files[:3]:  # Check last 3 log files
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                if "too many states" in content:
                    # Find the timestamp of the error
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if "too many states" in line:
                            print(f"   ❌ Schema error found in {log_file.name}")
                            # Print context
                            start = max(0, i-2)
                            end = min(len(lines), i+3)
                            for j in range(start, end):
                                if j == i:
                                    print(f"   >>> {lines[j]}")
                                else:
                                    print(f"       {lines[j]}")
                            return False
            except Exception as e:
                print(f"   Error reading {log_file}: {e}")
    
    print("   ✅ No schema errors found in recent logs")
    return True

async def run_quick_research_test():
    """Run a quick research test to see if the system works."""
    print("\n🚀 Running quick research test...")
    
    cmd = [
        'python3', '-c',
        '''
import asyncio
from scripts.gemini_deep_research import GeminiDeepResearch
from scripts.ai_brain import IntelligentAIBrain

async def test():
    ai_brain = IntelligentAIBrain()
    research = GeminiDeepResearch(ai_brain)
    # Just test the market timing phase
    result = await research._phase_0_market_timing()
    if result:
        print("✅ Market timing phase completed successfully")
        return True
    return False

asyncio.run(test())
'''
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("   ✅ Quick research test passed")
            print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Quick research test failed")
            if "too many states" in result.stderr:
                print("   ❌ SCHEMA ERROR DETECTED!")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ⚠️ Test timed out after 30 seconds")
        return False

async def main():
    """Main monitoring function."""
    print("Schema Error Monitor")
    print("=" * 60)
    print("This tool will check for schema errors in the Gemini deep research system.\n")
    
    # Run tests
    results = []
    
    # Test 1: Direct schema test
    results.append(await test_schema_directly())
    
    # Test 2: Check logs
    results.append(check_running_logs())
    
    # Test 3: Quick research test
    results.append(await run_quick_research_test())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    if all(results):
        print("✅ All tests passed! The schema fix is working correctly.")
        print("\nYou can now run the system without schema errors:")
        print("export $(cat .env.local | grep -v '^#' | xargs) && python3 scripts/continuous_orchestrator_no_redis.py")
    else:
        print("❌ Some tests failed. The schema error may still be present.")
        print("\nTroubleshooting steps:")
        print("1. Ensure gemini_deep_research.py has the simplified schema")
        print("2. Clear Python cache: rm -rf scripts/__pycache__")
        print("3. Check for any running Python processes and kill them")
        print("4. Restart your terminal to ensure clean environment")

if __name__ == "__main__":
    asyncio.run(main())