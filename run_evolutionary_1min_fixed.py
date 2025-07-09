#!/usr/bin/env python3
"""
Run evolutionary improvements for 1 minute with better error handling.
"""

import asyncio
import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up timeout handler
def timeout_handler(signum, frame):
    print("\n⏰ 90-second safety timeout reached!")
    sys.exit(1)

async def main():
    # Import here to avoid any module-level issues
    from run_evolutionary_improvements import EvolutionaryImprovementRunner
    
    print("Starting 1-minute evolutionary run...")
    
    runner = EvolutionaryImprovementRunner(runtime_minutes=1)
    
    try:
        # Set a hard timeout at 90 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(90)
        
        await runner.run()
        
        # Cancel the alarm
        signal.alarm(0)
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        await runner._final_summary()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to get summary even on error
        try:
            await runner._final_summary()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())