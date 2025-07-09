#!/usr/bin/env python3
"""
Test script to verify documentation checking for README, ARCHITECTURE, and MONETIZATION_STRATEGY files.
"""

import asyncio
import json
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.project_lifecycle_analyzer import ProjectLifecycleAnalyzer
from scripts.state_manager import StateManager
from scripts.ai_brain_factory import AIBrainFactory

async def test_documentation_checking():
    """Test the updated documentation checking system."""
    
    print("=== Testing Documentation Checking System ===\n")
    
    # Initialize components
    state_manager = StateManager()
    ai_brain = AIBrainFactory.create_for_conversational()
    
    # Create analyzers
    repo_analyzer = RepositoryAnalyzer(
        github_token=None,  # Use local mode
        state_manager=state_manager,
        ai_brain=ai_brain,
        use_local=True
    )
    
    lifecycle_analyzer = ProjectLifecycleAnalyzer(ai_brain)
    
    # Test repository (you can change this to any local repository)
    test_repo = "CodeWebMobile-AI/hydra-health-ai"
    
    print(f"Testing repository: {test_repo}")
    print("-" * 50)
    
    # Analyze repository
    print("\n1. Analyzing repository structure...")
    repo_analysis = await repo_analyzer.analyze_repository(test_repo)
    
    # Check documentation findings
    print("\n2. Documentation Analysis Results:")
    
    # Check collected documentation files
    docs_found = repo_analysis.get('code_analysis', {}).get('documentation', [])
    print(f"   - Documentation files found: {docs_found}")
    
    # Check ARCHITECTURE.md
    architecture = repo_analysis.get('architecture', {})
    print(f"   - ARCHITECTURE.md exists: {architecture.get('document_exists', False)}")
    
    # Check MONETIZATION_STRATEGY.md
    monetization = repo_analysis.get('monetization_strategy', {})
    print(f"   - MONETIZATION_STRATEGY.md exists: {monetization.get('document_exists', False)}")
    
    # Check specific needs
    print("\n3. Identified Documentation Needs:")
    specific_needs = repo_analysis.get('specific_needs', [])
    doc_needs = [need for need in specific_needs if 'documentation' in need.get('type', '')]
    
    for need in doc_needs:
        print(f"   - Type: {need['type']}")
        print(f"     Priority: {need['priority']}")
        print(f"     Description: {need['description']}")
        print(f"     Action: {need['suggested_action']}")
        print()
    
    # Test lifecycle analysis
    print("\n4. Lifecycle Documentation Score:")
    lifecycle_result = await lifecycle_analyzer.analyze_project_stage(repo_analysis)
    
    indicators = lifecycle_result.get('stage_indicators', {})
    doc_score = indicators.get('documentation_score', 0)
    print(f"   - Documentation Score: {doc_score:.2f} / 1.0")
    
    # Show which documents contribute to the score
    has_readme = any('readme.md' in doc.lower() for doc in docs_found)
    has_architecture = architecture.get('document_exists', False)
    has_monetization = monetization.get('document_exists', False)
    
    print(f"   - README.md: {'✓' if has_readme else '✗'} (0.33 points)")
    print(f"   - ARCHITECTURE.md: {'✓' if has_architecture else '✗'} (0.33 points)")
    print(f"   - MONETIZATION_STRATEGY.md: {'✓' if has_monetization else '✗'} (0.34 points)")
    
    # Check improvements recommended
    print("\n5. Recommended Improvements:")
    transition_plan = lifecycle_result.get('transition_plan', {})
    improvements = transition_plan.get('required_improvements', [])
    
    doc_improvements = [imp for imp in improvements if 'documentation' in imp.lower()]
    for imp in doc_improvements:
        print(f"   - {imp}")
    
    print("\n=== Test Complete ===")
    
    # Save results for inspection
    with open('documentation_test_results.json', 'w') as f:
        json.dump({
            'repository': test_repo,
            'documentation_files': docs_found,
            'architecture_exists': architecture.get('document_exists', False),
            'monetization_exists': monetization.get('document_exists', False),
            'documentation_score': doc_score,
            'documentation_needs': doc_needs,
            'recommended_improvements': doc_improvements
        }, f, indent=2)
    
    print("\nResults saved to documentation_test_results.json")

if __name__ == "__main__":
    asyncio.run(test_documentation_checking())