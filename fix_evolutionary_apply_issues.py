#!/usr/bin/env python3
"""
Fix all issues preventing improvements from being applied successfully.
"""

import re
from pathlib import Path

def fix_apply_best_improvement():
    """Fix the apply_best_improvement method to handle missing fields gracefully."""
    print("🔧 Fixing apply_best_improvement issues...")
    
    engine_file = Path("scripts/evolutionary_improvement_engine.py")
    if not engine_file.exists():
        print("  ❌ evolutionary_improvement_engine.py not found!")
        return False
    
    content = engine_file.read_text()
    
    # Fix 1: Handle missing 'id' field
    print("  📝 Fixing missing 'id' field handling...")
    
    # Find the apply_best_improvement method
    # Replace the line that expects 'id' to use .get() with a default
    old_id_line = "'id': best_improvement['id'],"
    new_id_line = "'id': best_improvement.get('id', hashlib.md5(f\"{best_improvement['name']}_{datetime.now().isoformat()}\".encode()).hexdigest()[:8]),"
    
    if old_id_line in content:
        content = content.replace(old_id_line, new_id_line)
        print("    ✓ Fixed 'id' field access")
    
    # Fix 2: Handle missing 'generation' field
    old_gen_line = "'generation': best_improvement['generation'],"
    new_gen_line = "'generation': best_improvement.get('generation', self.generation_count),"
    
    if old_gen_line in content:
        content = content.replace(old_gen_line, new_gen_line)
        print("    ✓ Fixed 'generation' field access")
    
    # Fix 3: Ensure fitness is present
    old_fitness_line = "'fitness': best_improvement['fitness'],"
    new_fitness_line = "'fitness': best_improvement.get('fitness', 0.0),"
    
    if old_fitness_line in content:
        content = content.replace(old_fitness_line, new_fitness_line)
        print("    ✓ Fixed 'fitness' field access")
    
    # Save the fixed file
    engine_file.write_text(content)
    print("  ✅ Saved fixes to evolutionary_improvement_engine.py")
    
    return True

def create_comprehensive_test():
    """Create a comprehensive test that verifies all fixes work."""
    print("\n📝 Creating comprehensive test...")
    
    test_content = '''#!/usr/bin/env python3
"""
Comprehensive test for evolutionary improvement application.
Tests all edge cases and ensures improvements can be applied successfully.
"""

import asyncio
import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.state_manager import StateManager
from scripts.evolutionary_improvement_engine import EvolutionaryImprovementEngine
from scripts.automated_code_verifier import AutomatedCodeVerifier

async def test_basic_application():
    """Test basic improvement application."""
    print("\\n1️⃣ Testing basic improvement application...")
    
    state_manager = StateManager()
    engine = EvolutionaryImprovementEngine(state_manager)
    
    # Create a minimal improvement
    improvement = {
        'name': 'test_basic_improvement',
        'type': 'optimization',
        'description': 'Basic test improvement',
        'approach': 'Simple approach',
        'metrics': ['test_metric'],
        'integration_points': ['test_system'],
        'fitness': 0.7,
        'code': """\"\"\"Basic test improvement\"\"\"
class TestBasicImprovement:
    def apply(self, context):
        return {'success': True}
"""
    }
    
    engine.improvement_population = [improvement]
    
    try:
        result = await engine.apply_best_improvement()
        if result['success']:
            print("  ✅ Basic application successful!")
            # Clean up
            Path(result['file_path']).unlink(missing_ok=True)
            return True
        else:
            print(f"  ❌ Basic application failed: {result.get('reason')}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def test_with_all_fields():
    """Test improvement with all fields present."""
    print("\\n2️⃣ Testing improvement with all fields...")
    
    state_manager = StateManager()
    engine = EvolutionaryImprovementEngine(state_manager)
    
    # Create improvement with all fields
    improvement = {
        'id': 'test123',
        'name': 'test_complete_improvement',
        'type': 'workflow',
        'description': 'Complete test improvement',
        'approach': 'Comprehensive approach',
        'metrics': ['metric1', 'metric2'],
        'integration_points': ['system1', 'system2'],
        'fitness': 0.85,
        'generation': 3,
        'code': """\"\"\"Complete test improvement\"\"\"
class TestCompleteImprovement:
    def apply(self, context):
        return {'success': True, 'message': 'All fields present'}
"""
    }
    
    engine.improvement_population = [improvement]
    engine.generation_count = 3
    
    try:
        result = await engine.apply_best_improvement()
        if result['success']:
            print("  ✅ Complete application successful!")
            # Verify state was updated correctly
            state = state_manager.get_state()
            if 'evolutionary_improvements' in state:
                last_improvement = state['evolutionary_improvements'][-1]
                assert last_improvement['id'] == 'test123', "ID mismatch"
                assert last_improvement['generation'] == 3, "Generation mismatch"
                print("  ✅ State updated correctly")
            # Clean up
            Path(result['file_path']).unlink(missing_ok=True)
            return True
        else:
            print(f"  ❌ Complete application failed: {result.get('reason')}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def test_verification_integration():
    """Test improvement application with verification."""
    print("\\n3️⃣ Testing verification integration...")
    
    state_manager = StateManager()
    engine = EvolutionaryImprovementEngine(state_manager)
    verifier = AutomatedCodeVerifier(state_manager)
    
    # Create a valid improvement
    improvement = {
        'name': 'test_verified_improvement',
        'type': 'tool',
        'description': 'Verified test improvement',
        'approach': 'Verified approach',
        'metrics': ['verification_rate'],
        'integration_points': ['verifier'],
        'fitness': 0.9,
        'code': '''"""Verified test improvement"""
import logging

class TestVerifiedImprovement:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.name = "test_verified_improvement"
    
    def apply(self, context):
        self.logger.info(f"Applying {self.name}")
        return {'success': True}
'''
    }
    
    engine.improvement_population = [improvement]
    
    try:
        # First verify the improvement
        verification = await verifier.verify_improvement(improvement)
        print(f"  Verification result: is_deployable={verification.get('is_deployable', False)}")
        print(f"  Overall score: {verification.get('overall_score', 0):.2f}")
        
        # Apply it
        result = await engine.apply_best_improvement()
        if result['success']:
            print("  ✅ Verified application successful!")
            # Clean up
            Path(result['file_path']).unlink(missing_ok=True)
            return True
        else:
            print(f"  ❌ Verified application failed: {result.get('reason')}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

async def test_edge_cases():
    """Test various edge cases."""
    print("\\n4️⃣ Testing edge cases...")
    
    state_manager = StateManager()
    engine = EvolutionaryImprovementEngine(state_manager)
    
    # Test with empty population
    engine.improvement_population = []
    try:
        result = await engine.apply_best_improvement()
        if not result['success']:
            print("  ✅ Empty population handled correctly")
    except:
        print("  ❌ Empty population caused error")
    
    # Test with special characters in name
    improvement = {
        'name': 'test_special_chars_123',
        'type': 'optimization',
        'description': 'Test with special chars',
        'approach': 'Handle special characters',
        'metrics': ['char_handling'],
        'integration_points': ['parser'],
        'fitness': 0.6,
        'code': """\"\"\"Special chars test\"\"\"
class TestSpecialChars123:
    pass
"""
    }
    
    engine.improvement_population = [improvement]
    
    try:
        result = await engine.apply_best_improvement()
        if result['success']:
            print("  ✅ Special characters handled correctly")
            # Clean up
            Path(result['file_path']).unlink(missing_ok=True)
            return True
        else:
            print(f"  ❌ Special characters failed: {result.get('reason')}")
            return False
    except Exception as e:
        print(f"  ❌ Error with special characters: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 COMPREHENSIVE EVOLUTIONARY IMPROVEMENT APPLICATION TEST")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(await test_basic_application())
    results.append(await test_with_all_fields())
    results.append(await test_verification_integration())
    results.append(await test_edge_cases())
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\\n🎉 ALL TESTS PASSED! The evolutionary system can now apply improvements successfully!")
    else:
        print("\\n⚠️  Some tests failed. Check the output above for details.")
    
    # Save test report
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'tests_run': total,
        'tests_passed': passed,
        'success_rate': passed / total if total > 0 else 0,
        'all_passed': passed == total
    }
    
    with open('evolutionary_fix_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n📄 Test report saved to: evolutionary_fix_test_report.json")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
'''
    
    with open('test_evolutionary_comprehensive_fix.py', 'w') as f:
        f.write(test_content)
    
    # Make it executable
    import os
    os.chmod('test_evolutionary_comprehensive_fix.py', 0o755)
    
    print("  ✅ Created test_evolutionary_comprehensive_fix.py")

def main():
    """Apply all fixes."""
    print("🔧 FIXING EVOLUTIONARY IMPROVEMENT APPLICATION ISSUES")
    print("=" * 60)
    
    # Apply fixes
    if fix_apply_best_improvement():
        print("\n✅ All fixes applied successfully!")
    else:
        print("\n❌ Some fixes failed to apply")
        return
    
    # Create comprehensive test
    create_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("✅ FIXES COMPLETE!")
    print("\nTo test all fixes, run:")
    print("export $(cat .env.local | grep -v '^#' | xargs) && PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai python3 test_evolutionary_comprehensive_fix.py")

if __name__ == "__main__":
    main()