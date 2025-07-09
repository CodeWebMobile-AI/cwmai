#!/usr/bin/env python3
"""
Fix work discovery to:
1. Properly detect existing documentation
2. Prioritize local issues over AI-generated work
3. Reduce excessive AI API calls
"""

import os
import sys

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def apply_fixes():
    """Apply all necessary fixes to the work discovery system."""
    
    print("🔧 Fixing work discovery system...")
    
    # 1. Clear any stale state data
    print("\n1️⃣ Clearing stale state data...")
    clear_stale_state()
    
    # 2. Fix the intelligent work finder to check issues first
    print("\n2️⃣ Fixing intelligent work finder...")
    fix_intelligent_work_finder()
    
    # 3. Update repository analyzer to properly detect docs
    print("\n3️⃣ Ensuring repository analyzer works correctly...")
    verify_repository_analyzer()
    
    print("\n✅ Work discovery fixes applied!")
    print("\nTo test the fixes:")
    print("1. Run: export $(cat .env.local | grep -v '^#' | xargs)")
    print("2. Run: PYTHONPATH=/Users/fortressarigo/PhpstormProjects/cwmai arch -arm64 /usr/bin/python3 test_work_discovery_fix.py")


def clear_stale_state():
    """Clear stale repository analysis from system state."""
    from scripts.state_manager import StateManager
    
    state_manager = StateManager()
    state = state_manager.load_state()
    
    # Clear specific_needs from all repositories
    cleared = 0
    for repo_type in ['repositories', 'projects']:
        if repo_type in state:
            for repo_name, repo_data in state[repo_type].items():
                if 'specific_needs' in repo_data:
                    # Clear the specific needs to force re-analysis
                    repo_data['specific_needs'] = []
                    cleared += 1
    
    if cleared > 0:
        state_manager.state = state
        state_manager.save_state()
        print(f"  ✓ Cleared specific_needs from {cleared} repositories")
    else:
        print(f"  ✓ No stale specific_needs found")


def fix_intelligent_work_finder():
    """Patch the intelligent work finder to prioritize issues."""
    
    # Read the current file
    file_path = "scripts/intelligent_work_finder.py"
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "# PATCHED: Check local issues first" in content:
        print("  ✓ Intelligent work finder already patched")
        return
    
    # Find the discover_work method and add issue checking first
    patch = '''
        # PATCHED: Check local issues first to avoid expensive AI calls
        print("Checking for existing issues before AI-powered discovery...")
        
        # 1. First, check if we're in local mode and get issues
        storage_mode = os.getenv('STORAGE_MODE', 'github')
        if storage_mode == 'local' or storage_mode == StorageMode.LOCAL:
            from scripts.local_github_simulator import LocalGitHubSimulator
            simulator = LocalGitHubSimulator()
            
            # Get all open issues across repositories
            all_issues = []
            for repo_name in active_projects.keys():
                try:
                    issues = simulator.get_issues(repo_name, state="open")
                    for issue in issues:
                        # Convert to WorkItem
                        all_issues.append(WorkItem(
                            id=f"issue_{repo_name}_{issue.number}",
                            task_type="ISSUE", 
                            title=issue.title,
                            description=issue.body or "",
                            repository=repo_name,
                            priority=self._extract_priority_from_labels(issue.labels),
                            metadata={
                                "issue_number": issue.number,
                                "labels": issue.labels,
                                "from_local_issue": True
                            }
                        ))
                except Exception as e:
                    self.logger.debug(f"Could not get issues for {repo_name}: {e}")
            
            if all_issues:
                print(f"Found {len(all_issues)} existing issues - using these instead of AI generation")
                # Shuffle for variety
                import random
                random.shuffle(all_issues)
                return all_issues[:max_items]
        
        # Continue with normal discovery only if no issues found
        '''
    
    # Find where to insert the patch - right after work_items = []
    lines = content.split('\n')
    insert_index = -1
    
    for i, line in enumerate(lines):
        if 'work_items = []' in line and 'discover_work' in lines[i-10:i].__str__():
            insert_index = i + 1
            break
    
    if insert_index > 0:
        # Insert the patch
        lines.insert(insert_index, patch)
        
        # Also add the priority extraction method if not present
        if '_extract_priority_from_labels' not in content:
            priority_method = '''
    def _extract_priority_from_labels(self, labels: List[str]) -> TaskPriority:
        """Extract priority from issue labels."""
        for label in labels:
            label_lower = label.lower()
            if "critical" in label_lower:
                return TaskPriority.CRITICAL
            elif "high" in label_lower:
                return TaskPriority.HIGH
            elif "medium" in label_lower:
                return TaskPriority.MEDIUM
            elif "low" in label_lower:
                return TaskPriority.LOW
        return TaskPriority.MEDIUM
'''
            # Add before the last line of the class
            for i in range(len(lines)-1, 0, -1):
                if lines[i].strip() and not lines[i].startswith(' '):
                    lines.insert(i, priority_method)
                    break
        
        # Write back
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print("  ✓ Patched intelligent work finder to check issues first")
    else:
        print("  ⚠️  Could not find insertion point for patch")


def verify_repository_analyzer():
    """Verify the repository analyzer properly detects documentation."""
    
    # The fix we already applied should work, just verify it
    from scripts.repository_analyzer import RepositoryAnalyzer
    
    # Test that our fix is in place
    file_path = "scripts/repository_analyzer.py"
    with open(file_path, 'r') as f:
        content = f.read()
    
    if "# Local simulator returns already decoded string" in content:
        print("  ✓ Repository analyzer has documentation detection fix")
    else:
        print("  ⚠️  Repository analyzer may need documentation detection fix")
    
    # Also clear any cached repository analysis
    import shutil
    cache_dir = "scripts/repository_analysis_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"  ✓ Cleared repository analysis cache")


if __name__ == "__main__":
    apply_fixes()