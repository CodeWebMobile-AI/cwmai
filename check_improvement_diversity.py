#!/usr/bin/env python3
"""Check improvement diversity in the real improvement system"""

import asyncio
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_real_improvements import DirectImprovementRunner

async def check_diversity(cycles: int = 10):
    """Run improvement cycles and check diversity"""
    runner = DirectImprovementRunner()
    
    improvement_types = defaultdict(int)
    improvement_names = []
    
    print(f"Running {cycles} improvement cycles to check diversity...\n")
    
    for i in range(cycles):
        print(f"Cycle {i+1}/{cycles}:")
        try:
            # Call the runner's create_and_apply_improvement method
            improvement = await runner.create_and_apply_improvement()
            
            if improvement.get('success'):
                imp_type = improvement.get('type', 'unknown')
                imp_name = improvement.get('name', 'N/A')
                
                improvement_types[imp_type] += 1
                improvement_names.append(f"{imp_type}: {imp_name}")
                
                print(f"  ✅ {imp_type}: {imp_name}")
            else:
                print(f"  ❌ Failed: {improvement.get('reason', 'Unknown')}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        # Small delay
        await asyncio.sleep(0.1)
    
    # Summary
    print("\n" + "="*50)
    print("IMPROVEMENT DIVERSITY SUMMARY")
    print("="*50)
    
    print("\nTypes created:")
    for imp_type, count in sorted(improvement_types.items()):
        print(f"  {imp_type}: {count}")
    
    print(f"\nTotal unique types: {len(improvement_types)}")
    print(f"Diversity score: {len(improvement_types) / cycles * 100:.1f}%")
    
    # Show improvement rotation pattern
    print("\nImprovement sequence:")
    for i, name in enumerate(improvement_names):
        print(f"  {i+1}. {name}")

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(check_diversity(cycles))