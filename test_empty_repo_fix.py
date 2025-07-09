#!/usr/bin/env python3
"""
Test the fix for empty repository name issue in hybrid_orchestrator_enhanced.py
"""

import asyncio
import logging
from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator

async def test_gather_contexts():
    """Test that _gather_project_contexts handles missing full_name fields correctly."""
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create orchestrator
    orchestrator = EnhancedHybridOrchestrator(max_workers=1)
    
    try:
        # Test gathering project contexts
        print("Testing _gather_project_contexts...")
        contexts = await orchestrator._gather_project_contexts()
        
        print(f"\nSuccessfully gathered contexts for {len(contexts)} projects:")
        for repo_name in sorted(contexts.keys()):
            print(f"  - {repo_name}")
        
        # Check that no empty repository names were processed
        if '' in contexts:
            print("\n❌ ERROR: Empty repository name found in contexts!")
            return False
        
        print("\n✅ SUCCESS: No empty repository names in contexts")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        await orchestrator.shutdown()

if __name__ == '__main__':
    success = asyncio.run(test_gather_contexts())
    exit(0 if success else 1)