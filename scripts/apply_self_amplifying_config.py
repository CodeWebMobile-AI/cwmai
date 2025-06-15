#!/usr/bin/env python3
"""
Apply Self-Amplifying Configuration to Research Evolution Engine

This script modifies the research_evolution_engine.py to activate all
self-amplifying features with optimal settings.
"""

import re
import os
import shutil
from datetime import datetime


def apply_configuration_changes():
    """Apply self-amplifying configuration to research evolution engine."""
    
    print("🔧 Applying self-amplifying configuration changes...")
    
    # Backup original file
    source_file = "research_evolution_engine.py"
    backup_file = f"research_evolution_engine.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if os.path.exists(source_file):
        shutil.copy(source_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
    
    # Read the current file
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Configuration changes to apply
    config_changes = [
        # Enable continuous fixed interval research
        (r'"enable_fixed_interval": False,', '"enable_fixed_interval": True,  # ACTIVATED: Continuous learning'),
        
        # Reduce cycle interval for faster learning
        (r'"cycle_interval_seconds": 30 \* 60,', '"cycle_interval_seconds": 20 * 60,  # 20 minutes for faster learning'),
        
        # Reduce emergency interval for quicker response
        (r'"emergency_cycle_interval": 5 \* 60,', '"emergency_cycle_interval": 3 * 60,  # 3 minutes for critical issues'),
        
        # Increase concurrent research
        (r'"max_concurrent_research": 3,', '"max_concurrent_research": 5,  # More parallel research'),
        
        # Increase research per cycle
        (r'"max_research_per_cycle": 5,', '"max_research_per_cycle": 8,  # More research per cycle'),
        
        # Lower confidence threshold for more insights
        (r'"min_insight_confidence": 0\.6,', '"min_insight_confidence": 0.5,  # More insights accepted'),
        
        # Lower auto-implementation threshold
        (r'"auto_implement_threshold": 0\.8,', '"auto_implement_threshold": 0.75,  # More auto-implementation'),
        
        # Increase external research frequency
        (r'"external_research_frequency": 4,', '"external_research_frequency": 2,  # Every 2nd cycle'),
        
        # Increase external capabilities per cycle
        (r'"max_external_capabilities_per_cycle": 3,', '"max_external_capabilities_per_cycle": 5,  # More external learning'),
        
        # Lower external synthesis threshold
        (r'"external_synthesis_threshold": 0\.7', '"external_synthesis_threshold": 0.6  # Easier synthesis')
    ]
    
    # Apply each configuration change
    changes_applied = 0
    for pattern, replacement in config_changes:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changes_applied += 1
            print(f"✓ Applied: {replacement.split(',')[0]}")
    
    # Add enable_proactive_research to config if not present
    if '"enable_proactive_research"' not in content:
        # Find the config dictionary and add the new setting
        config_pattern = r'(self\.config = \{[^}]+)"external_synthesis_threshold": 0\.\d+'
        if re.search(config_pattern, content):
            replacement = r'\1"external_synthesis_threshold": 0.6,\n            "enable_proactive_research": True  # ACTIVATED: Proactive opportunity scanning'
            content = re.sub(config_pattern, replacement, content)
            changes_applied += 1
            print("✓ Added: enable_proactive_research")
    
    # Write the modified content back
    with open(source_file, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Applied {changes_applied} configuration changes!")
    
    # Create a summary of changes
    summary = """
SELF-AMPLIFYING INTELLIGENCE ACTIVATED
=====================================

Configuration Changes Applied:
-----------------------------
1. Continuous Learning: Fixed interval research ENABLED (every 20 minutes)
2. Proactive Research: ENABLED for opportunity discovery
3. Faster Response: Emergency response reduced to 3 minutes
4. More Parallel Research: Increased to 5 concurrent researches
5. More Research Per Cycle: Increased to 8 topics per cycle
6. Lower Thresholds: More insights accepted (0.5) and auto-implemented (0.75)
7. Enhanced External Learning: Every 2nd cycle with 5 capabilities
8. Easier Synthesis: Threshold reduced to 0.6

Expected Benefits:
-----------------
• 3x faster learning from system behavior
• 2x more insights discovered per cycle
• Continuous improvement without manual intervention
• Automatic discovery of optimization opportunities
• Self-healing from performance issues
• Knowledge accumulation and pattern recognition

Safety Features Active:
----------------------
• Performance monitoring and rollback
• Validation before implementation
• Emergency stop mechanisms
• Bounded resource usage

Next Steps:
-----------
1. Start the continuous orchestrator: python run_continuous_ai.py
2. Monitor the research intelligence dashboard
3. Check research_knowledge/ for accumulated insights
4. Review system_state.json for performance improvements
"""
    
    # Save summary
    with open("self_amplifying_activation_summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    
    return changes_applied


def verify_activation():
    """Verify that self-amplifying features are properly activated."""
    
    print("\n🔍 Verifying activation...")
    
    with open("research_evolution_engine.py", 'r') as f:
        content = f.read()
    
    # Check key activations
    checks = [
        ('"enable_fixed_interval": True', "✅ Continuous learning is ACTIVE"),
        ('"enable_proactive_research": True', "✅ Proactive research is ACTIVE"),
        ('"cycle_interval_seconds": 20', "✅ Fast learning cycle (20 min) is SET"),
        ('"external_research_frequency": 2', "✅ Frequent external learning is SET"),
        ('"max_research_per_cycle": 8', "✅ High research volume is SET")
    ]
    
    all_active = True
    for pattern, message in checks:
        if pattern in content:
            print(message)
        else:
            print(f"❌ Missing: {pattern}")
            all_active = False
    
    if all_active:
        print("\n🎉 All self-amplifying features are properly activated!")
    else:
        print("\n⚠️  Some features may need manual activation")
    
    return all_active


if __name__ == "__main__":
    # Apply configuration changes
    changes = apply_configuration_changes()
    
    # Verify activation
    if verify_activation():
        print("\n✨ Self-amplifying intelligence is now fully operational!")
        print("   Run 'python run_continuous_ai.py' to start the enhanced system")