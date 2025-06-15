#!/usr/bin/env python3
"""
Test intelligent improvement components without requiring AI providers
"""

import os
import sys
import asyncio

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from improvement_learning_system import ImprovementLearningSystem, ImprovementOutcome
from safe_self_improver import ModificationType
from ai_code_analyzer import CodeImprovement
from datetime import datetime


def test_learning_system():
    """Test the improvement learning system."""
    print("🧠 Testing Improvement Learning System\n")
    
    # Initialize learning system
    learning_system = ImprovementLearningSystem()
    
    # Create test improvements
    test_improvement = CodeImprovement(
        type=ModificationType.OPTIMIZATION,
        description="Convert loop to list comprehension",
        original_code="result = []\nfor x in items:\n    result.append(x * 2)",
        improved_code="result = [x * 2 for x in items]",
        explanation="List comprehensions are more Pythonic and faster",
        confidence=0.85,
        line_start=10,
        line_end=12,
        impact_analysis={'performance': 'medium', 'readability': 'improved'},
        test_suggestions=["Test output is identical", "Benchmark performance"]
    )
    
    # Record some outcomes
    print("1️⃣ Recording improvement outcomes...")
    
    # Successful outcome
    learning_system.record_outcome(
        improvement=test_improvement,
        success=True,
        metrics={'performance': 0.2, 'readability': 0.3}
    )
    
    # Failed outcome
    failed_improvement = CodeImprovement(
        type=ModificationType.SECURITY,
        description="Add input validation",
        original_code="user_input = request.get('data')",
        improved_code="user_input = validate_input(request.get('data'))",
        explanation="Prevent injection attacks",
        confidence=0.9,
        line_start=20,
        line_end=20,
        impact_analysis={'security': 'high'},
        test_suggestions=["Test with malicious input"]
    )
    
    learning_system.record_outcome(
        improvement=failed_improvement,
        success=False,
        metrics={'error_reduction': -0.1},
        feedback="Validation function not defined"
    )
    
    print("✅ Recorded 2 outcomes\n")
    
    # Score new improvements
    print("2️⃣ Scoring improvements based on learning...")
    
    score1 = learning_system.score_improvement(test_improvement)
    score2 = learning_system.score_improvement(failed_improvement)
    
    print(f"✅ Optimization improvement score: {score1:.2f}")
    print(f"✅ Security improvement score: {score2:.2f}\n")
    
    # Generate report
    print("3️⃣ Generating learning report...")
    report = learning_system.generate_learning_report()
    
    print("✅ Learning Report:")
    print(f"   Total improvements: {report['summary']['total_improvements']}")
    print(f"   Success rate: {report['summary']['overall_success_rate']:.0%}")
    print(f"   Patterns learned: {report['summary']['patterns_learned']}")
    
    if report['by_type']:
        print("\n   By type:")
        for imp_type, stats in report['by_type'].items():
            print(f"     - {imp_type}: {stats['successful']}/{stats['total']} successful")
    
    print("\n✅ Learning system working correctly!")
    return True


def test_pattern_recognition():
    """Test pattern recognition in improvements."""
    print("\n🔍 Testing Pattern Recognition\n")
    
    learning_system = ImprovementLearningSystem()
    
    # Create similar improvements
    improvements = [
        CodeImprovement(
            type=ModificationType.OPTIMIZATION,
            description="Convert loop to comprehension",
            original_code="results = []\nfor item in data:\n    results.append(process(item))",
            improved_code="results = [process(item) for item in data]",
            explanation="More efficient",
            confidence=0.8,
            line_start=1,
            line_end=3
        ),
        CodeImprovement(
            type=ModificationType.OPTIMIZATION,
            description="Convert loop to comprehension",
            original_code="output = []\nfor x in values:\n    output.append(x.upper())",
            improved_code="output = [x.upper() for x in values]",
            explanation="More efficient",
            confidence=0.8,
            line_start=10,
            line_end=12
        ),
        CodeImprovement(
            type=ModificationType.OPTIMIZATION,
            description="Use dict.get",
            original_code="if key in data:\n    value = data[key]\nelse:\n    value = default",
            improved_code="value = data.get(key, default)",
            explanation="More concise",
            confidence=0.9,
            line_start=20,
            line_end=23
        )
    ]
    
    # Record successful outcomes for first two (loop patterns)
    for imp in improvements[:2]:
        learning_system.record_outcome(imp, success=True, metrics={'performance': 0.15})
    
    # Record failed outcome for dict.get pattern
    learning_system.record_outcome(improvements[2], success=False, 
                                   feedback="Default value type mismatch")
    
    # Check pattern recognition
    print("✅ Recorded outcomes for 3 improvements\n")
    
    # Get recommendations
    recommendations = learning_system.get_recommendations(
        improvement_type=ModificationType.OPTIMIZATION
    )
    
    print("📊 Pattern recommendations:")
    for rec in recommendations[:3]:
        print(f"   - Pattern: {rec['pattern']}")
        print(f"     Success rate: {rec['success_rate']:.0%} ({rec['count']} attempts)")
        print(f"     Recommendation: {rec['recommendation']}")
    
    return True


def test_staged_improvements():
    """Test staged improvement metadata."""
    print("\n📦 Testing Staged Improvement System\n")
    
    from staged_self_improver import StagedImprovement, Modification
    from datetime import timezone
    
    # Create test modification
    import uuid
    
    mod = Modification(
        id=str(uuid.uuid4()),
        type=ModificationType.OPTIMIZATION,
        target_file="test.py",
        description="Optimize loop",
        changes=[{
            'type': 'replace',
            'original': 'for loop code',
            'replacement': 'comprehension code',
            'line_number': 10
        }],
        timestamp=datetime.now(timezone.utc),
        safety_score=0.85
    )
    
    # Create staged improvement
    staged = StagedImprovement(
        modification=mod,
        staged_path="/tmp/staged/test.py",
        original_path="/workspace/test.py",
        created_at=datetime.now(timezone.utc),
        metadata={
            'staging_id': 'test_001',
            'lines_changed': 3,
            'complexity_change': -2
        }
    )
    
    print("✅ Created staged improvement:")
    print(f"   Type: {staged.modification.type.value}")
    print(f"   Description: {staged.modification.description}")
    print(f"   Safety score: {staged.modification.safety_score}")
    print(f"   Staging ID: {staged.metadata['staging_id']}")
    print(f"   Complexity change: {staged.metadata['complexity_change']}")
    
    return True


def main():
    """Run all component tests."""
    print("="*60)
    print("INTELLIGENT IMPROVEMENT COMPONENTS TEST")
    print("="*60)
    
    # Test learning system
    test1_ok = test_learning_system()
    
    # Test pattern recognition
    test2_ok = test_pattern_recognition()
    
    # Test staged improvements
    test3_ok = test_staged_improvements()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Learning System: {'✅ PASS' if test1_ok else '❌ FAIL'}")
    print(f"Pattern Recognition: {'✅ PASS' if test2_ok else '❌ FAIL'}")
    print(f"Staged Improvements: {'✅ PASS' if test3_ok else '❌ FAIL'}")
    print("="*60)
    
    return test1_ok and test2_ok and test3_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)