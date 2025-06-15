#!/usr/bin/env python3
"""
Integration test to verify the staged improvement system works
"""

import os
import sys
import asyncio
import tempfile
import shutil

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from staged_self_improver import StagedSelfImprover
from progressive_confidence import ProgressiveConfidence, RiskLevel
from safe_self_improver import ModificationType


async def test_staging_system():
    """Test the basic staging system functionality."""
    print("🧪 Testing Staged Improvement System\n")
    
    # Create test file with known improvements
    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "test_example.py")
    
    with open(test_file, 'w') as f:
        f.write('''
def process_items(items):
    # This could be optimized to list comprehension
    result = []
    for item in items:
        result.append(item * 2)
    return result

def get_value(data, key):
    # This could use dict.get()
    if key in data:
        value = data[key]
    else:
        value = None
    return value

class Calculator:
    def add(self, x, y):
        # Missing docstring
        return x + y
''')
    
    # Initialize git repo (required by SafeSelfImprover)
    os.system(f'cd {test_dir} && git init > /dev/null 2>&1 && git add . && git commit -m "Initial" > /dev/null 2>&1')
    
    try:
        # 1. Initialize components
        print("1️⃣ Initializing components...")
        improver = StagedSelfImprover(repo_path=test_dir, max_changes_per_day=10)
        confidence = ProgressiveConfidence(test_dir)
        print("✅ Components initialized\n")
        
        # 2. Find improvements
        print("2️⃣ Finding improvement opportunities...")
        opportunities = improver.analyze_improvement_opportunities()
        print(f"✅ Found {len(opportunities)} opportunities:")
        for opp in opportunities:
            print(f"   - {opp['type'].value}: {opp['description']} in {opp['file']}")
        print()
        
        if not opportunities:
            print("❌ No improvements found - pattern matching might need adjustment")
            return False
        
        # 3. Stage improvements
        print("3️⃣ Staging improvements...")
        staged_ids = await improver.stage_batch_improvements(opportunities, max_batch=2)
        print(f"✅ Staged {len(staged_ids)} improvements")
        for sid in staged_ids:
            print(f"   - {sid}")
        print()
        
        if not staged_ids:
            print("❌ Failed to stage improvements")
            return False
        
        # 4. Validate staged improvements
        print("4️⃣ Validating staged improvements...")
        validation_results = await improver.validate_batch(staged_ids)
        
        validated_count = sum(1 for r in validation_results.values() if r.get('ready_to_apply', False))
        print(f"✅ Validated {validated_count}/{len(staged_ids)} improvements")
        
        for sid, result in validation_results.items():
            status = "✅ Ready" if result.get('ready_to_apply', False) else "❌ Not ready"
            print(f"   - {sid[:12]}... : {status}")
            if result.get('errors'):
                for error in result['errors'][:2]:
                    print(f"     Error: {error}")
        print()
        
        # 5. Check confidence system
        print("5️⃣ Checking confidence system...")
        should_apply, reason = confidence.should_auto_apply(
            ModificationType.OPTIMIZATION,
            RiskLevel.LOW
        )
        print(f"✅ Auto-apply decision: {should_apply}")
        print(f"   Reason: {reason}")
        print(f"   Confidence score: {confidence.metrics.confidence_score:.2f}")
        print()
        
        # 6. Generate report
        print("6️⃣ Generating staging report...")
        report = improver.generate_staging_report()
        print(f"✅ Report generated:")
        print(f"   - Total staged: {report['summary']['total_staged']}")
        print(f"   - Total validated: {report['summary']['total_validated']}")
        print(f"   - Total applied: {report['summary']['total_applied']}")
        print()
        
        print("✅ All core components working!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


async def test_simple_staging():
    """Test just the basic staging functionality."""
    print("\n🧪 Testing Simple Staging\n")
    
    try:
        # Use the actual repository
        improver = StagedSelfImprover(repo_path="/workspaces/cwmai", max_changes_per_day=10)
        
        # Check if staging directories exist
        print("Checking staging directories:")
        dirs_exist = all([
            os.path.exists(improver.staging_dir),
            os.path.exists(improver.validated_dir),
            os.path.exists(improver.applied_dir)
        ])
        print(f"✅ Staging directories: {'exist' if dirs_exist else 'missing'}")
        
        # Try to load configuration
        config = improver.config
        print(f"✅ Configuration loaded: {len(config)} settings")
        
        # Check for the simple_optimization_target.py file
        target_file = "/workspaces/cwmai/simple_optimization_target.py"
        if os.path.exists(target_file):
            print(f"✅ Test file exists: {target_file}")
            
            # Analyze it
            with open(target_file, 'r') as f:
                content = f.read()
                print(f"   File has {len(content.splitlines())} lines")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("STAGED IMPROVEMENT SYSTEM TEST")
    print("="*60)
    
    # Test basic functionality first
    basic_ok = await test_simple_staging()
    
    # Then test full integration
    integration_ok = await test_staging_system()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Basic functionality: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Integration test: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    return basic_ok and integration_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)