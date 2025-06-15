#!/usr/bin/env python3
"""
Test minimal orchestrator initialization to isolate the error
"""

import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_minimal_orchestrator():
    """Test just creating the orchestrator without running it."""
    print("=== Testing Minimal Orchestrator Creation ===")
    
    # Set up logging to capture detailed errors
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("1. Importing ContinuousOrchestrator...")
        from continuous_orchestrator import ContinuousOrchestrator
        print("✓ Import successful")
        
        print("2. Creating orchestrator instance...")
        orchestrator = ContinuousOrchestrator(max_workers=1, enable_parallel=False)
        print("✓ Orchestrator created")
        
        print("3. Testing status retrieval...")
        status = orchestrator.get_status()
        print(f"✓ Status retrieved: running={status['running']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_orchestrator()
    print(f"\nTest {'passed' if success else 'failed'}")
    sys.exit(0 if success else 1)