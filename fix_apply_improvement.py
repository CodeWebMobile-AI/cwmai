#!/usr/bin/env python3
"""
Fix the apply_best_improvement method to handle missing 'id' field.
"""

import re
from pathlib import Path

# Fix the evolutionary_improvement_engine.py file
engine_file = Path("scripts/evolutionary_improvement_engine.py")
if engine_file.exists():
    content = engine_file.read_text()
    
    # Replace direct field access with .get() methods
    replacements = [
        ("'id': best_improvement['id'],", 
         "'id': best_improvement.get('id', hashlib.md5(f\"{best_improvement['name']}_{datetime.now().isoformat()}\".encode()).hexdigest()[:8]),"),
        ("'generation': best_improvement['generation'],", 
         "'generation': best_improvement.get('generation', self.generation_count),"),
        ("'fitness': best_improvement['fitness'],", 
         "'fitness': best_improvement.get('fitness', 0.0),"),
    ]
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"✓ Fixed: {old[:30]}...")
    
    engine_file.write_text(content)
    print("\n✅ Fixed evolutionary_improvement_engine.py")
else:
    print("❌ File not found: scripts/evolutionary_improvement_engine.py")