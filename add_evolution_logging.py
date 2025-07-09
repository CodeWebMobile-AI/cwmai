#!/usr/bin/env python3
"""
Add detailed logging to the evolution system to diagnose where it's getting stuck.
"""

import sys
import re

def add_logging_to_file():
    """Add detailed logging statements to run_knowledge_driven_evolution.py"""
    
    with open('run_knowledge_driven_evolution.py', 'r') as f:
        content = f.read()
    
    # Add logging after research cycle completes
    content = re.sub(
        r'(results = await self\.research_engine\.execute_research_cycle\(\))',
        r'\1\n            print(f"DEBUG: Research cycle returned: {type(results)}, keys: {results.keys() if isinstance(results, dict) else \'Not a dict\'}")\n            print(f"DEBUG: Research insights count: {len(results.get(\'insights\', []))}")',
        content
    )
    
    # Add logging before and after evolve_generation
    content = re.sub(
        r'(# Evolve generation\n\s+)(await self\.evolutionary_engine\.evolve_generation\(\))',
        r'\1print(f"DEBUG: Starting evolve_generation for generation {generation}")\n            \2\n            print(f"DEBUG: Completed evolve_generation for generation {generation}")',
        content
    )
    
    # Add logging at the start of each main loop iteration
    content = re.sub(
        r'(print\(f"\\n🔄 Generation {generation} - {time_display}"\))',
        r'\1\n            print(f"DEBUG: Main loop - Generation {generation} starting")\n            print(f"DEBUG: Population size: {len(self.evolutionary_engine.improvement_population) if hasattr(self.evolutionary_engine, \'improvement_population\') else \'Unknown\'}")',
        content
    )
    
    # Add logging after research insights integration
    content = re.sub(
        r'(await self\._integrate_research_insights\(research_results\[\'insights\'\]\))',
        r'print(f"DEBUG: Integrating {len(research_results[\'insights\'])} research insights")\n                        \1\n                        print("DEBUG: Research insights integrated")',
        content
    )
    
    # Add error handling with logging
    content = re.sub(
        r'(except Exception as e:\n\s+print\(f"❌ Research cycle error: {e}"\))',
        r'\1\n            import traceback\n            print(f"DEBUG: Full research error traceback:")\n            traceback.print_exc()',
        content
    )
    
    # Add logging to the research integration method
    content = re.sub(
        r'(async def _integrate_research_insights\(self, insights: List\[Dict\]\):)',
        r'\1\n        """Integrate research insights into the evolutionary process."""\n        print(f"DEBUG: _integrate_research_insights called with {len(insights)} insights")',
        content
    )
    
    # Add timeout detection
    content = re.sub(
        r'(while end_time is None or datetime\.now\(\)\.timestamp\(\) < end_time:)',
        r'last_generation_time = datetime.now()\n        \1\n            # Check if generation is taking too long\n            current_time = datetime.now()\n            if (current_time - last_generation_time).seconds > 300:  # 5 minutes\n                print(f"WARNING: Generation {generation} has been running for over 5 minutes")\n                print(f"DEBUG: Last checkpoint was at {last_generation_time}")\n            last_generation_time = current_time',
        content
    )
    
    with open('run_knowledge_driven_evolution_debug.py', 'w') as f:
        f.write(content)
    
    print("✅ Created run_knowledge_driven_evolution_debug.py with detailed logging")
    print("\nRun with:")
    print("export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 run_knowledge_driven_evolution_debug.py --breakthrough-focus")

if __name__ == "__main__":
    add_logging_to_file()