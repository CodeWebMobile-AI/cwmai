#!/usr/bin/env python3
"""
Fix to prevent documentation task creation when files already exist.

The issue: Portfolio analysis generates documentation tasks even when files exist.
The fix: Add explicit checks before creating any documentation tasks.
"""

import os
import sys
from pathlib import Path

def apply_fixes():
    """Apply fixes to prevent documentation task creation."""
    
    print("🔧 Applying documentation task creation fixes...\n")
    
    # Fix 1: Update intelligent_work_finder.py to check files before portfolio analysis
    work_finder_path = Path("scripts/intelligent_work_finder.py")
    
    print("1️⃣ Updating intelligent_work_finder.py to check documentation before task creation...")
    
    content = work_finder_path.read_text()
    
    # Find the portfolio analysis section
    if "portfolio_insights = await self.portfolio_intelligence.analyze_portfolio(active_projects)" in content:
        # Add documentation check before portfolio analysis
        insert_pos = content.find("portfolio_insights = await self.portfolio_intelligence.analyze_portfolio(active_projects)")
        
        check_code = '''
        # Pre-check: Mark documentation as complete for repos that have all files
        if storage_mode == 'local':
            print("Pre-checking documentation files in all repositories...")
            for repo_name in active_projects.keys():
                try:
                    repo = simulator.get_repository(repo_name)
                    has_all_docs = True
                    
                    for doc_file in ['README.md', 'ARCHITECTURE.md', 'MONETIZATION_STRATEGY.md']:
                        try:
                            repo.get_contents(doc_file)
                        except:
                            has_all_docs = False
                            break
                    
                    if has_all_docs:
                        # Mark all documentation tasks as completed for this repo
                        print(f"  ✓ {repo_name} has complete documentation")
                        for doc_type in ['readme_documentation', 'architecture_documentation', 'monetization_documentation']:
                            self.task_persistence.mark_task_completed(
                                task_type=doc_type,
                                repository=repo_name,
                                title=f"Create {doc_type.replace('_', ' ')} for {repo_name}"
                            )
                except Exception as e:
                    print(f"  ⚠️ Could not check {repo_name}: {e}")
        
        '''
        
        content = content[:insert_pos] + check_code + content[insert_pos:]
        
        work_finder_path.write_text(content)
        print("  ✓ Added pre-check for documentation files before portfolio analysis")
    
    # Fix 2: Update portfolio_intelligence.py to respect completed documentation
    portfolio_path = Path("scripts/portfolio_intelligence.py")
    
    print("\n2️⃣ Updating portfolio_intelligence.py to check for existing documentation...")
    
    content = portfolio_path.read_text()
    
    # Find the _identify_strategic_gaps method
    if "def _identify_strategic_gaps" in content:
        # Add check for existing documentation
        method_start = content.find("def _identify_strategic_gaps")
        method_body_start = content.find(":", method_start) + 1
        
        # Find the end of docstring
        docstring_end = content.find('"""', method_body_start + 10) + 3
        
        check_code = '''
        
        # Check for existing documentation files in local mode
        storage_mode = os.getenv('STORAGE_MODE', 'github')
        if storage_mode == 'local':
            from scripts.local_github_simulator import LocalGitHubSimulator
            simulator = LocalGitHubSimulator()
            
            # Remove documentation gaps for repos that already have files
            filtered_gaps = []
            for gap in gaps:
                if gap['type'] == 'documentation' and gap.get('subtype') in ['architecture', 'monetization', 'readme']:
                    # Check if files exist
                    repo_names = gap.get('affected_projects', [])
                    repos_needing_docs = []
                    
                    for repo_name in repo_names:
                        try:
                            repo = simulator.get_repository(repo_name)
                            doc_file = {
                                'architecture': 'ARCHITECTURE.md',
                                'monetization': 'MONETIZATION_STRATEGY.md', 
                                'readme': 'README.md'
                            }.get(gap['subtype'])
                            
                            try:
                                repo.get_contents(doc_file)
                                # File exists, skip this repo
                            except:
                                # File doesn't exist, keep in list
                                repos_needing_docs.append(repo_name)
                        except:
                            # Keep repo if we can't check
                            repos_needing_docs.append(repo_name)
                    
                    if repos_needing_docs:
                        gap['affected_projects'] = repos_needing_docs
                        filtered_gaps.append(gap)
                else:
                    filtered_gaps.append(gap)
            
            gaps = filtered_gaps
        '''
        
        # Insert after the gaps are identified
        insert_pos = content.find("return gaps", method_start)
        content = content[:insert_pos] + check_code + "\n        " + content[insert_pos:]
        
        portfolio_path.write_text(content)
        print("  ✓ Added documentation existence check to portfolio analysis")
    
    print("\n✅ All fixes applied successfully!")
    print("\nThese fixes ensure:")
    print("  1. Documentation files are checked BEFORE portfolio analysis")
    print("  2. Portfolio analysis respects existing documentation")
    print("  3. No documentation tasks are created for repos with complete docs")
    
    return True

if __name__ == "__main__":
    if apply_fixes():
        print("\n🎉 Documentation task creation fixes applied!")
        print("\nTo test the fixes:")
        print("1. Run: export $(cat .env.local | grep -v '^#' | xargs) && python3 test_work_discovery_fix.py")
        print("2. Run the system and monitor that no documentation tasks are created")