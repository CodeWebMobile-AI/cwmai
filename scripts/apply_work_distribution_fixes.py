#!/usr/bin/env python3
"""
Integrate Work Distribution Fixes

This script shows how to apply the work distribution fixes to the orchestrator.
"""

import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.work_distribution_patch import WorkDistributionPatch


def apply_fixes_to_hybrid_orchestrator():
    """Apply fixes to the enhanced hybrid orchestrator."""
    
    # Import the orchestrator
    from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
    
    # Create original init method wrapper
    original_init = EnhancedHybridOrchestrator.__init__
    
    def patched_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Apply work distribution patch
        WorkDistributionPatch.apply_to_orchestrator(self)
        
        print("✅ Work distribution fixes applied to orchestrator")
    
    # Replace init method
    EnhancedHybridOrchestrator.__init__ = patched_init
    
    print("✅ Enhanced hybrid orchestrator patched with work distribution fixes")


# Apply the fixes when this module is imported
apply_fixes_to_hybrid_orchestrator()
