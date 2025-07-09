#!/usr/bin/env python3
"""
Fix the final issues in the evolutionary system.
"""

from pathlib import Path

def fix_missing_logger():
    """Fix missing logger in EvolutionaryImprovementRunner."""
    print("1. Fixing missing logger attribute...")
    
    runner_file = Path("run_evolutionary_improvements.py")
    if runner_file.exists():
        content = runner_file.read_text()
        
        # Find the __init__ method and add logger
        init_start = content.find("def __init__(self, runtime_minutes: Optional[int] = None):")
        if init_start > 0:
            # Find the end of the first line after __init__
            init_line_end = content.find("\n", init_start)
            # Insert logger initialization after the first line
            insert_point = content.find("\n", init_line_end + 1)
            
            if "self.logger = " not in content[init_start:init_start + 500]:
                # Add logger after runtime_minutes
                new_content = (
                    content[:insert_point] + 
                    "\n        self.logger = logger  # Use module-level logger" +
                    content[insert_point:]
                )
                runner_file.write_text(new_content)
                print("  ✓ Added logger attribute")
        else:
            print("  ❌ Could not find __init__ method")
    else:
        print("  ❌ File not found")

def fix_name_keyerror():
    """Fix KeyError: 'name' in improvement_ensemble.py."""
    print("\n2. Fixing KeyError: 'name' in ensemble...")
    
    ensemble_file = Path("scripts/improvement_ensemble.py")
    if ensemble_file.exists():
        content = ensemble_file.read_text()
        
        # Fix line 224: Improvement 1: {idea1['name']}
        old_line = "Improvement 1: {idea1['name']}"
        new_line = "Improvement 1: {idea1.get('name', 'unnamed')}"
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            print("  ✓ Fixed idea1['name'] access")
        
        # Also fix idea2['name']
        old_line2 = "Improvement 2: {idea2['name']}"
        new_line2 = "Improvement 2: {idea2.get('name', 'unnamed')}"
        
        if old_line2 in content:
            content = content.replace(old_line2, new_line2)
            print("  ✓ Fixed idea2['name'] access")
        
        # Fix other potential KeyError issues in the prompt
        content = content.replace("idea1['type']", "idea1.get('type', 'unknown')")
        content = content.replace("idea2['type']", "idea2.get('type', 'unknown')")
        content = content.replace('"{idea1[\'name\']}"', '"{idea1.get(\'name\', \'unnamed\')}"')
        content = content.replace('"{idea2[\'name\']}"', '"{idea2.get(\'name\', \'unnamed\')}"')
        
        ensemble_file.write_text(content)
        print("  ✓ Fixed all name/type access issues")
    else:
        print("  ❌ File not found")

def main():
    """Apply all fixes."""
    print("🔧 FIXING FINAL EVOLUTIONARY SYSTEM ISSUES")
    print("=" * 60)
    
    fix_missing_logger()
    fix_name_keyerror()
    
    print("\n✅ All fixes applied!")
    print("\nRun the system again with:")
    print("arch -arm64 /usr/bin/python3 run_evolutionary_improvements.py --minutes 1")

if __name__ == "__main__":
    main()