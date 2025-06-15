#!/usr/bin/env python3
"""
Test the intelligent self-improvement system
"""

import os
import sys
import asyncio

# Set environment variables
os.environ['INTELLIGENT_IMPROVEMENT_ENABLED'] = 'true'
os.environ['SELF_IMPROVEMENT_STAGING_ENABLED'] = 'true'
os.environ['SELF_IMPROVEMENT_AUTO_VALIDATE'] = 'true'
os.environ['SELF_IMPROVEMENT_AUTO_APPLY_VALIDATED'] = 'false'

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from ai_brain import IntelligentAIBrain
from intelligent_self_improver import IntelligentSelfImprover
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


async def test_intelligent_improvements():
    """Test the intelligent improvement system."""
    print("🤖 Testing Intelligent Self-Improvement System\n")
    
    try:
        # Initialize AI brain
        print("1️⃣ Initializing AI brain...")
        ai_brain = IntelligentAIBrain()
        print("✅ AI brain initialized\n")
        
        # Initialize intelligent improver
        print("2️⃣ Initializing intelligent self-improver...")
        improver = IntelligentSelfImprover(
            ai_brain=ai_brain,
            repo_path="/workspaces/cwmai",
            staging_enabled=True
        )
        print("✅ Intelligent improver initialized\n")
        
        # Generate intelligence report
        print("3️⃣ Generating intelligence report...")
        report = improver.generate_intelligence_report()
        print("✅ Intelligence report:")
        print(f"   Configuration:")
        print(f"     - Min confidence: {report['configuration']['min_confidence']}")
        print(f"     - Auto-apply threshold: {report['configuration']['auto_apply_threshold']}")
        print(f"     - Context awareness: {report['configuration']['context_awareness']}")
        print(f"     - Learning enabled: {report['configuration']['learning_enabled']}")
        print(f"   Confidence metrics:")
        print(f"     - Score: {report['confidence_metrics']['score']:.2f}")
        print(f"     - Total improvements: {report['confidence_metrics']['total_improvements']}")
        print(f"     - Can auto-apply: {report['confidence_metrics']['can_auto_apply']}")
        print()
        
        # Find improvements on test file
        print("4️⃣ Finding intelligent improvements...")
        test_files = [
            "/workspaces/cwmai/test_target_for_improvements.py",
            "/workspaces/cwmai/simple_optimization_target.py"
        ]
        
        # Filter to existing files
        existing_files = [f for f in test_files if os.path.exists(f)]
        if not existing_files:
            print("❌ No test files found. Creating one...")
            # Create a test file
            test_file = "/workspaces/cwmai/test_ai_improvements.py"
            with open(test_file, 'w') as f:
                f.write('''"""
Test file for AI improvements
"""

def process_data(items):
    # This could be a list comprehension
    result = []
    for item in items:
        result.append(item.strip().upper())
    return result

def check_value(data, key):
    # This could use dict.get()
    if key in data:
        value = data[key]
    else:
        value = "default"
    return value

def calculate_sum(numbers):
    # Missing docstring
    total = 0
    for num in numbers:
        total = total + num
    return total

class DataProcessor:
    def process(self, data):
        # Complex method that could be simplified
        output = []
        for i in range(len(data)):
            if data[i] is not None:
                if data[i] > 0:
                    output.append(data[i] * 2)
                else:
                    output.append(0)
        return output
''')
            existing_files = [test_file]
        
        improvements = await improver.find_intelligent_improvements(
            target_files=existing_files,
            max_improvements=5
        )
        
        print(f"✅ Found {len(improvements)} intelligent improvements:")
        for i, imp in enumerate(improvements[:3]):  # Show first 3
            print(f"\n   {i+1}. {imp['type'].value}: {imp['description']}")
            print(f"      File: {os.path.basename(imp['file'])}")
            print(f"      Score: {imp['score']:.2f}")
            print(f"      Lines: {imp['line_start']}-{imp['line_end']}")
        print()
        
        if not improvements:
            print("❌ No improvements found. The AI analysis might need adjustment.")
            return False
        
        # Stage and validate improvements
        print("5️⃣ Applying improvements (staging)...")
        results = await improver.apply_intelligent_improvements(
            improvements[:2],  # Apply first 2
            auto_apply=False  # Manual mode for testing
        )
        
        print("✅ Application results:")
        print(f"   Total: {results['total']}")
        print(f"   Staged: {results['staged']}")
        print(f"   Validated: {results['validated']}")
        print(f"   Applied: {results['applied']}")
        print(f"   Failed: {results['failed']}")
        
        # Check staging report
        if improver.staged_improver:
            staging_report = improver.staged_improver.generate_staging_report()
            print(f"\n   Staging summary:")
            print(f"     - Total staged: {staging_report['summary']['total_staged']}")
            print(f"     - Total validated: {staging_report['summary']['total_validated']}")
        
        print("\n✅ Intelligent self-improvement system is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing intelligent improvements: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_analysis():
    """Test the context-aware analysis."""
    print("\n🔍 Testing Context-Aware Analysis\n")
    
    try:
        # Initialize components
        ai_brain = IntelligentAIBrain()
        
        from context_aware_improver import ContextAwareImprover
        
        context_improver = ContextAwareImprover(
            ai_brain=ai_brain,
            repo_path="/workspaces/cwmai"
        )
        
        # Generate context report
        print("Analyzing codebase context...")
        context_report = context_improver.generate_context_report()
        
        print("✅ Context analysis complete:")
        print(f"   Total files: {context_report['summary']['total_files']}")
        print(f"   Total dependencies: {context_report['summary']['total_dependencies']}")
        print(f"   Average complexity: {context_report['summary']['average_complexity']:.1f}")
        print(f"   Critical files: {len(context_report['critical_files'])}")
        
        if context_report['most_connected']:
            print("\n   Most connected files:")
            for file_info in context_report['most_connected'][:3]:
                print(f"     - {file_info['file']}")
                print(f"       Connections: {file_info['total_connections']}")
                print(f"       Complexity: {file_info['complexity']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in context analysis: {e}")
        return False


async def main():
    """Run all tests."""
    print("="*60)
    print("INTELLIGENT SELF-IMPROVEMENT SYSTEM TEST")
    print("="*60)
    
    # Test intelligent improvements
    test1_ok = await test_intelligent_improvements()
    
    # Test context analysis
    test2_ok = await test_context_analysis()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Intelligent improvements: {'✅ PASS' if test1_ok else '❌ FAIL'}")
    print(f"Context analysis: {'✅ PASS' if test2_ok else '❌ FAIL'}")
    print("="*60)
    
    return test1_ok and test2_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)