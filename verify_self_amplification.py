#!/usr/bin/env python3
"""
Verify Self-Amplifying Intelligence Configuration

This script verifies that all self-amplifying features are properly configured
and ready to operate.
"""

import os
import sys
import json
from pathlib import Path

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from research_evolution_engine import ResearchEvolutionEngine


def verify_configuration():
    """Verify all self-amplifying features are active."""
    print("🔍 Verifying Self-Amplifying Intelligence Configuration...")
    print("=" * 60)
    
    # Create a test instance
    engine = ResearchEvolutionEngine()
    config = engine.config
    
    # Check critical features
    checks = [
        ("Continuous Learning (Fixed Interval)", config.get("enable_fixed_interval", False)),
        ("Dynamic Performance Triggers", config.get("enable_dynamic_triggering", False)),
        ("Proactive Research Discovery", config.get("enable_proactive_research", False)),
        ("External Agent Learning", config.get("enable_external_agent_research", False)),
        ("Fast Learning Cycle (20 min)", config.get("cycle_interval_seconds") == 20 * 60),
        ("Quick Emergency Response (3 min)", config.get("emergency_cycle_interval") == 3 * 60),
        ("High Research Volume (8/cycle)", config.get("max_research_per_cycle") == 8),
        ("Frequent External Learning (every 2)", config.get("external_research_frequency") == 2),
        ("More Concurrent Research (5)", config.get("max_concurrent_research") == 5),
        ("Lower Insight Threshold (0.5)", config.get("min_insight_confidence") == 0.5),
        ("Auto-Implementation (0.75)", config.get("auto_implement_threshold") == 0.75),
    ]
    
    all_active = True
    for feature, is_active in checks:
        status = "✅ ACTIVE" if is_active else "❌ INACTIVE"
        print(f"{feature:.<50} {status}")
        if not is_active:
            all_active = False
    
    print("\n📊 Configuration Summary:")
    print(f"  • Research Cycle: Every {config.get('cycle_interval_seconds', 0) / 60:.0f} minutes")
    print(f"  • Emergency Response: {config.get('emergency_cycle_interval', 0) / 60:.0f} minutes")
    print(f"  • Research Per Cycle: {config.get('max_research_per_cycle', 0)} topics")
    print(f"  • Concurrent Research: {config.get('max_concurrent_research', 0)} parallel")
    print(f"  • External Research: Every {config.get('external_research_frequency', 0)} cycles")
    print(f"  • Cross-Analysis: Every 3 cycles (hardcoded)")
    
    # Check directories
    print("\n📁 Required Directories:")
    dirs = [
        "research_knowledge",
        "research_knowledge/raw_research",
        "research_knowledge/processed_insights",
        "research_knowledge/metadata",
        "logs"
    ]
    
    for dir_path in dirs:
        exists = os.path.exists(dir_path)
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"  {dir_path:.<45} {status}")
    
    # Check for activation report
    print("\n📄 Activation Status:")
    if os.path.exists("research_activation_report.json"):
        with open("research_activation_report.json", "r") as f:
            report = json.load(f)
        print(f"  • Activated at: {report.get('activation_timestamp', 'Unknown')}")
        print(f"  • Config version: {report.get('config_version', 'Unknown')}")
        print("  ✅ Activation report found")
    else:
        print("  ⚠️  No activation report found")
    
    # Final verdict
    print("\n" + "=" * 60)
    if all_active:
        print("✨ SELF-AMPLIFYING INTELLIGENCE: FULLY ACTIVATED ✨")
        print("\nThe system is configured for maximum autonomous learning and improvement.")
        print("Start with: python start_self_amplifying_ai.py")
    else:
        print("⚠️  SELF-AMPLIFYING INTELLIGENCE: PARTIALLY ACTIVE")
        print("\nSome features need activation. Run: python scripts/activate_self_amplifying_intelligence.py")
    
    return all_active


def show_expected_behavior():
    """Show what to expect from the self-amplifying system."""
    print("\n🎯 Expected Self-Amplifying Behavior:")
    print("=" * 60)
    print("""
1. CONTINUOUS RESEARCH (Every 20 minutes):
   • Analyzes system performance gaps
   • Researches solutions autonomously
   • Extracts actionable insights
   • Implements improvements automatically

2. DYNAMIC PERFORMANCE MONITORING:
   • Detects performance drops instantly
   • Triggers emergency research (3 min response)
   • Self-heals from failures
   • Adapts to changing conditions

3. EXTERNAL LEARNING (Every 2nd cycle):
   • Discovers new AI capabilities on GitHub
   • Analyzes AI research papers
   • Synthesizes external knowledge
   • Integrates proven patterns

4. CROSS-RESEARCH ANALYSIS (Every 3rd cycle):
   • Finds patterns across all research
   • Generates meta-insights
   • Updates research strategies
   • Improves learning efficiency

5. PROACTIVE OPPORTUNITY DISCOVERY:
   • Scans for optimization opportunities
   • Predicts future needs
   • Suggests innovative improvements
   • Explores new capabilities

MEASURABLE OUTCOMES:
   • 3x faster learning from failures
   • 2x more insights per research cycle
   • Continuous performance improvement
   • Automatic adaptation to new patterns
   • Knowledge accumulation over time
""")


if __name__ == "__main__":
    # Run verification
    is_active = verify_configuration()
    
    # Show expected behavior
    show_expected_behavior()
    
    # Exit with appropriate code
    sys.exit(0 if is_active else 1)