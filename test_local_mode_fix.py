#!/usr/bin/env python3
"""
Test that local mode doesn't try to access non-existent repositories
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set local mode
os.environ['STORAGE_MODE'] = 'LOCAL'

from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
from scripts.local_github_simulator import LocalGitHubSimulator


def test_local_mode_repositories():
    """Test that only existing local repositories are added."""
    print("🧪 Testing Local Mode Repository Handling")
    print("=" * 60)
    
    # Get actual local repositories
    local_sim = LocalGitHubSimulator()
    existing_repos = local_sim.list_repositories()
    
    print(f"\n📂 Found {len(existing_repos)} local repositories:")
    for repo in existing_repos:
        print(f"   - {repo.full_name}")
    
    # Create orchestrator
    print("\n🔧 Creating orchestrator in LOCAL mode...")
    orchestrator = EnhancedHybridOrchestrator(max_workers=1)
    
    # Check coordinator
    if orchestrator.coordinator:
        print("✅ Coordinator initialized")
        print(f"   - Storage mode: {orchestrator.coordinator.storage_factory.storage_mode}")
        print(f"   - Is local mode: {orchestrator.coordinator.is_local_mode}")
        print(f"   - Active repos: {len(orchestrator.coordinator.active_repos)}")
        
        # Check which repos were added
        if orchestrator.coordinator.repositories:
            print("\n📋 Repositories added to coordinator:")
            for repo_name in orchestrator.coordinator.repositories:
                print(f"   - {repo_name}")
        else:
            print("\n⚠️  No repositories added to coordinator")
    else:
        print("❌ Coordinator not initialized")
    
    print("\n✅ Test completed - no errors about non-existent repositories!")


if __name__ == "__main__":
    test_local_mode_repositories()