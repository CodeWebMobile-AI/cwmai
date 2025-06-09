#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced debugging capabilities of DynamicSwarmIntelligence.

This script shows how to:
1. Enable comprehensive debug logging
2. Monitor swarm analysis with detailed logs
3. Identify the specific source of "list index out of range" errors
4. Track AI response quality and parsing issues
"""

import logging
import asyncio
import json
from scripts.dynamic_swarm import DynamicSwarmIntelligence
from scripts.ai_brain_factory import create_ai_brain


async def test_swarm_with_debug_logging():
    """Test the swarm with comprehensive debug logging enabled."""
    
    print("🔍 Testing Dynamic Swarm with Enhanced Debug Logging")
    print("=" * 60)
    
    try:
        # Create AI brain
        print("1. Creating AI brain...")
        ai_brain = await create_ai_brain()
        
        # Create dynamic swarm
        print("2. Creating dynamic swarm...")
        swarm = DynamicSwarmIntelligence(ai_brain)
        
        # Enable debug logging
        print("3. Enabling debug logging...")
        swarm.enable_debug_logging("DEBUG")
        
        print("\n" + "="*60)
        print("🧠 SWARM DEBUG INFORMATION")
        print("="*60)
        
        # Get debug summary
        debug_summary = swarm.get_debug_summary()
        print(f"Swarm Configuration:")
        print(f"  - Total Agents: {debug_summary['swarm_config']['total_agents']}")
        print(f"  - Learning System: {debug_summary['swarm_config']['learning_system_available']}")
        print(f"  - Charter System: {debug_summary['swarm_config']['charter_system_available']}")
        
        print(f"\nAgent Details:")
        for agent_detail in debug_summary['swarm_config']['agent_details']:
            print(f"  - {agent_detail['id']} ({agent_detail['role']}) using {agent_detail['model']}")
        
        print("\n" + "="*60)
        print("🚀 RUNNING TEST TASK ANALYSIS")
        print("="*60)
        
        # Create a test task that might trigger the error
        test_task = {
            'id': 'debug_test_task',
            'type': 'NEW_PROJECT',
            'description': 'Create a simple blog application using Laravel React starter',
            'priority': 7,
            'created_at': '2025-01-09T00:00:00Z'
        }
        
        print(f"Test Task: {test_task['description']}")
        print(f"Task Type: {test_task['type']}")
        print(f"Priority: {test_task['priority']}")
        
        # Create basic context
        context = {
            'active_projects': [],
            'charter': {
                'PRIMARY_PURPOSE': 'Build software portfolio',
                'CORE_OBJECTIVES': ['Learn', 'Build', 'Deploy']
            }
        }
        
        print(f"\n🔍 Starting swarm analysis with detailed logging...")
        print("Watch the logs below for any 'list index out of range' errors:")
        print("-" * 60)
        
        # Run the swarm analysis
        result = await swarm.process_task_swarm(test_task, context)
        
        print("-" * 60)
        print("✅ Swarm analysis completed successfully!")
        
        # Print result summary
        collective_review = result.get('collective_review', {})
        print(f"\nResult Summary:")
        print(f"  - Recommendation: {collective_review.get('recommendation', 'N/A')}")
        print(f"  - Priority: {collective_review.get('consensus_priority', 'N/A')}")
        print(f"  - Success Probability: {collective_review.get('success_probability', 'N/A')}")
        print(f"  - Duration: {result.get('duration_seconds', 'N/A')} seconds")
        
        # Check for any errors in individual analyses
        individual_analyses = result.get('individual_analyses', [])
        refined_analyses = result.get('refined_analyses', [])
        
        print(f"\n📊 Analysis Quality Check:")
        print(f"  - Individual Analyses: {len(individual_analyses)}")
        print(f"  - Refined Analyses: {len(refined_analyses)}")
        
        error_count = 0
        empty_challenges_count = 0
        
        for i, analysis in enumerate(individual_analyses + refined_analyses):
            if 'error' in analysis:
                error_count += 1
                print(f"  ❌ Analysis {i} has error: {analysis.get('error', 'Unknown')}")
            
            challenges = analysis.get('challenges', [])
            if not challenges:
                empty_challenges_count += 1
                agent_id = analysis.get('agent_id', 'unknown')
                print(f"  ⚠️  Analysis {i} from {agent_id} has empty challenges list")
        
        print(f"\n📈 Error Summary:")
        print(f"  - Total Errors: {error_count}")
        print(f"  - Empty Challenge Lists: {empty_challenges_count}")
        
        if error_count == 0 and empty_challenges_count == 0:
            print("  ✅ No errors or empty challenge lists detected!")
        else:
            print("  ⚠️  Some issues detected. Check the detailed logs above.")
        
        # Get updated debug summary
        final_debug_summary = swarm.get_debug_summary()
        print(f"\n📊 Final Performance Metrics:")
        perf_metrics = final_debug_summary.get('performance_metrics', {})
        print(f"  - Total Tasks: {perf_metrics.get('total_tasks', 0)}")
        print(f"  - Average Duration: {perf_metrics.get('average_duration', 0):.2f}s")
        print(f"  - Average Confidence: {perf_metrics.get('average_confidence', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False


async def main():
    """Main test function."""
    print("🧪 Enhanced Dynamic Swarm Debug Test")
    print("This test will help identify the source of 'list index out of range' errors")
    print("and provide comprehensive logging for debugging.\n")
    
    success = await test_swarm_with_debug_logging()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\nTo debug future issues:")
        print("1. Look for [SWARM_DEBUG] log messages")
        print("2. Check for 'EMPTY challenges list' warnings")
        print("3. Monitor AI response parsing errors")
        print("4. Track agent performance inconsistencies")
    else:
        print("\n💥 Test failed - check the logs above for details")
        print("\nDebugging tips:")
        print("1. The error logs will show exactly where the failure occurred")
        print("2. Look for parse errors or AI response issues")
        print("3. Check if specific agents are consistently failing")
        print("4. Verify that all required AI models are available")


if __name__ == "__main__":
    asyncio.run(main())