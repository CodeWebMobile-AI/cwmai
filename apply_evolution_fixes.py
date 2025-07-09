#!/usr/bin/env python3
"""
Apply all research engine fixes to the knowledge-driven evolution system
"""

import os
import sys

def apply_fixes():
    """Apply the fixes to run_knowledge_driven_evolution.py"""
    
    print("Applying research engine fixes to knowledge-driven evolution system...")
    
    # Read the original file
    with open('run_knowledge_driven_evolution.py', 'r') as f:
        content = f.read()
    
    # Add imports for the fixes at the top (after other imports)
    import_addition = """
# Import research engine fixes
from scripts.research_engine_fixes import apply_research_engine_fixes
from scripts.research_engine_optimized import apply_research_optimizations
"""
    
    # Find the last import statement
    import_lines = []
    lines = content.split('\n')
    last_import_idx = 0
    
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            last_import_idx = i
    
    # Insert the new imports after the last import
    lines.insert(last_import_idx + 1, import_addition)
    
    # Now find where research engine is initialized and add fixes
    fix_code = """
        # Apply comprehensive fixes to research engine
        print("🔧 Applying research engine fixes...")
        self.research_engine = apply_research_engine_fixes(self.research_engine)
        self.research_engine = apply_research_optimizations(self.research_engine)
        
        # Update configuration for better performance
        self.research_engine.config.update({
            "max_concurrent_research": 3,
            "max_research_per_cycle": 4,  # Reasonable limit for evolution
            "enable_external_agent_research": False,  # Disable for stability
            "enable_proactive_research": False  # Disable for stability
        })
        print("✅ Research engine fixes applied")
"""
    
    # Find where to insert the fixes (after research engine initialization)
    for i, line in enumerate(lines):
        if 'print("✅ Research components initialized")' in line:
            # Insert after this line
            lines.insert(i + 1, fix_code)
            break
    
    # Write the fixed version
    fixed_content = '\n'.join(lines)
    
    # Backup original
    if not os.path.exists('run_knowledge_driven_evolution.py.backup'):
        with open('run_knowledge_driven_evolution.py.backup', 'w') as f:
            with open('run_knowledge_driven_evolution.py', 'r') as orig:
                f.write(orig.read())
        print("✅ Original backed up to run_knowledge_driven_evolution.py.backup")
    
    # Write fixed version
    with open('run_knowledge_driven_evolution.py', 'w') as f:
        f.write(fixed_content)
    
    print("✅ Fixes applied to run_knowledge_driven_evolution.py")
    print("\nYou can now run:")
    print("export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 run_knowledge_driven_evolution.py --minutes 60 --breakthrough-focus")

if __name__ == "__main__":
    apply_fixes()