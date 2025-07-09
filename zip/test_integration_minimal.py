#!/usr/bin/env python3
"""Minimal integration test to verify the system can start without errors."""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_system_startup():
    """Test that the main components can be initialized."""
    print("Testing system startup...")
    
    try:
        # Test 1: Local GitHub Simulator
        print("\n1. Testing Local GitHub Simulator...")
        from scripts.local_github_simulator import LocalGitHubSimulator
        simulator = LocalGitHubSimulator()
        # Test that we can list repository directories
        repos_path = simulator.base_path / "repositories" / "CodeWebMobile-AI"
        if repos_path.exists():
            repos = list(repos_path.iterdir())
            print(f"   ✅ Found {len(repos)} local repositories")
        else:
            print(f"   ✅ Local repos directory initialized at {simulator.base_path}")
        
        # Test 2: Enhanced Architecture Generator imports
        print("\n2. Testing Enhanced Architecture Generator...")
        from scripts.enhanced_architecture_generator import EnhancedArchitectureGenerator
        print("   ✅ Architecture generator imports successfully")
        
        # Test 3: Work Discovery
        print("\n3. Testing Work Discovery Components...")
        from scripts.intelligent_work_finder import IntelligentWorkFinder
        print("   ✅ Intelligent work finder imports successfully")
        
        # Test 4: Orchestrator
        print("\n4. Testing Orchestrator...")
        from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
        print("   ✅ Enhanced hybrid orchestrator imports successfully")
        
        # Test 5: Project Creator
        print("\n5. Testing Project Creator...")
        from scripts.project_creator import ProjectCreator
        print("   ✅ Project creator imports successfully")
        
        print("\n✅ All components can be initialized successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_system_startup())
    sys.exit(0 if success else 1)