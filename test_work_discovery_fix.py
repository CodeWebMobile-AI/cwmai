#!/usr/bin/env python3
"""Test the work discovery fixes."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_work_discovery():
    """Test that work discovery now works correctly."""
    
    print("🧪 Testing work discovery fixes...\n")
    
    # 1. Test repository analysis
    print("1️⃣ Testing repository documentation detection:")
    from scripts.repository_analyzer import RepositoryAnalyzer
    from scripts.local_github_simulator import LocalGitHubSimulator
    from scripts.state_manager import StateManager
    
    simulator = LocalGitHubSimulator()
    state_manager = StateManager()
    system_state = state_manager.load_state_with_repository_discovery()
    
    analyzer = RepositoryAnalyzer(
        ai_brain=None,
        state_manager=state_manager,
        use_cache=False
    )
    
    # Test first repository
    repos = simulator.list_repositories()
    if repos:
        repo = repos[0]
        print(f"  Testing {repo.full_name}...")
        
        # Check documentation files directly
        docs_found = 0
        for doc in ['README.md', 'ARCHITECTURE.md', 'MONETIZATION_STRATEGY.md']:
            try:
                content = repo.get_contents(doc)
                print(f"    ✓ {doc} exists (size: {content.size} bytes)")
                docs_found += 1
            except:
                print(f"    ✗ {doc} missing")
        
        # Analyze repository
        print(f"\n  Running repository analysis...")
        analysis = await analyzer.analyze_repository(repo.full_name)
        
        # Check specific needs
        needs = analysis.get('needs', {})
        doc_needs = [need for need in analysis.get('specific_needs', []) 
                    if 'documentation' in need.get('type', '')]
        
        print(f"\n  Analysis results:")
        print(f"    - Documentation files found: {docs_found}/3")
        print(f"    - Documentation needs identified: {len(doc_needs)}")
        
        if docs_found == 3 and len(doc_needs) > 0:
            print(f"\n  ❌ ERROR: Documentation needs identified despite all files existing!")
            print(f"     Needs: {[n['type'] for n in doc_needs]}")
        elif docs_found == 3 and len(doc_needs) == 0:
            print(f"\n  ✅ SUCCESS: No documentation needs - correctly detected existing files!")
        
    # 2. Test work discovery
    print("\n\n2️⃣ Testing work discovery prioritization:")
    
    from scripts.intelligent_work_finder import IntelligentWorkFinder
    from scripts.ai_brain_factory import AIBrainFactory
    
    ai_brain = AIBrainFactory.create_for_production()
    work_finder = IntelligentWorkFinder(ai_brain, system_state)
    
    # Quick discovery with timeout
    try:
        print("  Discovering work (30 second timeout)...")
        work_items = await asyncio.wait_for(
            work_finder.discover_work(max_items=5),
            timeout=30.0
        )
        
        print(f"\n  Discovered {len(work_items)} work items:")
        
        issue_count = 0
        doc_count = 0
        other_count = 0
        
        for item in work_items:
            print(f"    - [{item.task_type}] {item.repository}: {item.title}")
            
            if item.metadata.get('from_local_issue'):
                issue_count += 1
            elif 'documentation' in item.task_type.lower():
                doc_count += 1
            else:
                other_count += 1
        
        print(f"\n  Summary:")
        print(f"    - Local issues: {issue_count}")
        print(f"    - Documentation tasks: {doc_count}")
        print(f"    - Other tasks: {other_count}")
        
        if issue_count > 0 and doc_count == 0:
            print(f"\n  ✅ SUCCESS: System prioritized existing issues over documentation!")
        elif doc_count > 0:
            print(f"\n  ⚠️  WARNING: Still generating documentation tasks")
            
    except asyncio.TimeoutError:
        print("\n  ⏱️  Work discovery timed out (this suggests it's still making too many AI calls)")
    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # 3. Check for issues
    print("\n\n3️⃣ Checking available issues:")
    issue_count = 0
    for repo in repos[:2]:
        issues = simulator.get_issues(repo.full_name, state="open")
        print(f"  {repo.full_name}: {len(issues)} open issues")
        issue_count += len(issues)
        for issue in issues[:2]:
            print(f"    - #{issue.number}: {issue.title}")
    
    if issue_count == 0:
        print("\n  ⚠️  No issues found - system will fall back to AI generation")
    else:
        print(f"\n  ✓ Total {issue_count} issues available for work")


if __name__ == "__main__":
    asyncio.run(test_work_discovery())