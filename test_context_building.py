#!/usr/bin/env python3
"""Test context building with the fixed architecture"""

import sys
import os
import asyncio
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator

async def test_context_building():
    """Test that context building works without MONOLITH errors"""
    print("=== Testing Context Building ===\n")
    
    # Create orchestrator
    orchestrator = EnhancedHybridOrchestrator(max_workers=1)
    
    # Create a fake project data
    fake_project = {
        'id': 'test-project',
        'full_name': 'CodeWebMobile-AI/test-project',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'open_issues': 5
    }
    
    try:
        print("Building context for test project...")
        context = await orchestrator._build_project_context(
            'CodeWebMobile-AI/test-project',
            fake_project
        )
        
        print(f"✓ Context built successfully!")
        print(f"  - Architecture: {context.technical_context.architecture_type}")
        print(f"  - Project stage: {context.stage}")
        print(f"  - Project ID: {context.project_id}")
        
        # The key test: architecture should be LARAVEL_REACT
        assert context.technical_context.architecture_type.value == "laravel_react", \
            f"Expected laravel_react, got {context.technical_context.architecture_type.value}"
        
        print("\n✅ Context building test passed! No MONOLITH errors.")
        
    except Exception as e:
        print(f"\n❌ Context building failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_context_building())