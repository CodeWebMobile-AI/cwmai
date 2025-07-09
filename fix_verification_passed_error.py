#!/usr/bin/env python3
"""
Fix the KeyError: 'passed' issue in the evolutionary improvement system.
The verification results use 'is_deployable' field, not 'passed'.
"""

import re
from pathlib import Path

def fix_verification_errors():
    """Fix all instances of v['passed'] to use the correct field."""
    print("🔧 Fixing verification 'passed' errors...")
    
    files_to_fix = [
        ("run_evolutionary_improvements.py", [
            (r"verification\['passed'\]", "verification['is_deployable']"),
            (r"if v\['passed'\]", "if v['is_deployable']"),
        ]),
        ("test_evolutionary_comprehensive.py", [
            (r"verification\['passed'\]", "verification['is_deployable']"),
            (r"result\.get\('passed'\)", "result.get('is_deployable')"),
        ]),
        ("test_evolutionary_4min_fixed.py", [
            (r"verification\['passed'\]", "verification['is_deployable']"),
        ])
    ]
    
    fixed_count = 0
    
    for filename, replacements in files_to_fix:
        filepath = Path(filename)
        if not filepath.exists():
            print(f"  ⚠️  {filename} not found, skipping...")
            continue
            
        print(f"\n  📄 Fixing {filename}...")
        content = filepath.read_text()
        original_content = content
        
        for pattern, replacement in replacements:
            matches = len(re.findall(pattern, content))
            if matches > 0:
                content = re.sub(pattern, replacement, content)
                print(f"    ✓ Fixed {matches} instance(s) of {pattern}")
                fixed_count += matches
        
        if content != original_content:
            filepath.write_text(content)
            print(f"    ✅ Saved changes to {filename}")
        else:
            print(f"    ℹ️  No changes needed in {filename}")
    
    # Also check for any other potential issues in the apply_best_improvement method
    print("\n  🔍 Checking apply_best_improvement implementation...")
    runner_file = Path("run_evolutionary_improvements.py")
    if runner_file.exists():
        content = runner_file.read_text()
        
        # Ensure the method properly handles verification results
        if "'passed'" in content and "'is_deployable'" not in content:
            print("    ⚠️  Found additional 'passed' references that may need fixing")
        else:
            print("    ✅ No additional 'passed' references found")
    
    print(f"\n✅ Fixed {fixed_count} total instances of the 'passed' error")
    return fixed_count

def create_test_script():
    """Create a test script to verify the fix works."""
    print("\n📝 Creating test script...")
    
    test_content = '''#!/usr/bin/env python3
"""
Test that improvements can be applied without KeyError: 'passed'
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.state_manager import StateManager
from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine

async def test_improvement_application():
    """Test that we can apply improvements without errors."""
    print("🧪 Testing improvement application...")
    
    state_manager = StateManager()
    engine = EvolutionaryImprovementEngine(state_manager)
    
    # Create a test improvement
    test_improvement = {
        'name': 'test_improvement',
        'type': 'optimization',
        'description': 'Test improvement for verification',
        'approach': 'Simple test approach',
        'metrics': ['test_metric'],
        'integration_points': ['test_system'],
        'fitness': 0.8,
        'code': """
\"\"\"Test improvement\"\"\"
import logging

class TestImprovement:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def apply(self, context):
        return {'success': True}
"""
    }
    
    # Add to population
    engine.improvement_population = [test_improvement]
    
    # Try to apply it
    try:
        result = await engine.apply_best_improvement()
        if result['success']:
            print("✅ Successfully applied improvement!")
            print(f"   File created: {result['file_path']}")
            
            # Clean up
            if Path(result['file_path']).exists():
                Path(result['file_path']).unlink()
                print("   🧹 Cleaned up test file")
        else:
            print(f"❌ Failed to apply improvement: {result.get('reason', 'Unknown')}")
    except KeyError as e:
        print(f"❌ KeyError encountered: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_improvement_application())
    if success:
        print("\\n✅ Test passed - 'passed' error is fixed!")
    else:
        print("\\n❌ Test failed - issue persists")
'''
    
    with open('test_verification_fix.py', 'w') as f:
        f.write(test_content)
    os.chmod('test_verification_fix.py', 0o755)
    
    print("  ✅ Created test_verification_fix.py")

def main():
    """Apply fixes and create test."""
    print("🔧 FIXING VERIFICATION 'PASSED' ERROR")
    print("=" * 50)
    
    # Apply fixes
    fixed = fix_verification_errors()
    
    # Create test script
    create_test_script()
    
    print("\n" + "=" * 50)
    print("✅ FIXES APPLIED!")
    print("\nTo test the fix, run:")
    print("export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai python3 test_verification_fix.py")

if __name__ == "__main__":
    main()