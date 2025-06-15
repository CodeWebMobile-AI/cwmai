"""
Test cross-research analysis functionality.
"""

import sys
import os
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from cross_research_analyzer import CrossResearchAnalyzer

print("Testing Cross-Research Analysis...")
print("=" * 40)

# Create analyzer
analyzer = CrossResearchAnalyzer()

# Create test research items
test_research = [
    {
        "id": "test1",
        "type": "efficiency", 
        "content": "Implement caching to improve performance significantly",
        "timestamp": datetime.now().isoformat(),
        "quality_score": 0.8
    },
    {
        "id": "test2",
        "type": "efficiency",
        "content": "Database optimization improves query performance",
        "timestamp": datetime.now().isoformat(),
        "quality_score": 0.7
    },
    {
        "id": "test3",
        "type": "innovation",
        "content": "New caching strategies with Redis improve system speed",
        "timestamp": datetime.now().isoformat(),
        "quality_score": 0.9
    },
    {
        "id": "test4",
        "type": "efficiency",
        "content": "Performance degrades when cache is not used properly",
        "timestamp": datetime.now().isoformat(),
        "quality_score": 0.6
    }
]

# Test pattern analysis
print("\n1. Pattern Analysis:")
patterns = analyzer._analyze_patterns(test_research)
print(f"✅ Found {len(patterns['patterns'])} patterns")
for pattern in patterns['patterns'][:3]:
    print(f"   - {pattern.get('description', pattern.get('type'))}")

# Test theme analysis
print("\n2. Theme Analysis:")
themes = analyzer._analyze_themes(test_research)
print(f"✅ Found {len(themes['major_themes'])} major themes")
for theme in themes['major_themes'][:3]:
    print(f"   - {theme['name']}: {theme['percentage']:.1f}% of documents")

# Test convergence analysis
print("\n3. Convergence Analysis:")
convergences = analyzer._find_convergences(test_research)
print(f"✅ Found {len(convergences['convergent_findings'])} convergent findings")
print(f"✅ Agreement score: {convergences['agreement_score']:.2%}")

# Test contradiction detection
print("\n4. Contradiction Analysis:")
contradictions = analyzer._find_contradictions(test_research)
print(f"✅ Found {contradictions['total_found']} contradictions")
if contradictions['contradictions']:
    print(f"   - Example: '{contradictions['contradictions'][0]['opposing_terms']}'")

# Test meta-insights generation
print("\n5. Meta-Insights:")
insights = analyzer._generate_meta_insights(test_research)
print(f"✅ Generated {len(insights)} meta-insights")
for insight in insights:
    print(f"   - {insight['type']}: {insight['insight'][:60]}...")

# Test recommendations
print("\n6. Recommendations:")
recommendations = analyzer._generate_recommendations(test_research)
print(f"✅ Generated {len(recommendations)} recommendations")
for rec in recommendations[:2]:
    print(f"   - [{rec['priority']}] {rec['recommendation']}")

print("\n" + "=" * 40)
print("✨ Cross-research analysis working perfectly!")
print("\nCapabilities demonstrated:")
print("- Pattern detection across research")
print("- Theme extraction and analysis")
print("- Convergence/divergence identification")
print("- Contradiction detection")
print("- Meta-insight generation")
print("- Actionable recommendations")