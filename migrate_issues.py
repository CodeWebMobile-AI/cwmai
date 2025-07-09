#!/usr/bin/env python3
"""
Migrate issues from incorrect issues.json files to proper individual issue files
"""
import json
import os
from pathlib import Path
from datetime import datetime, timezone

def migrate_issues():
    """Migrate all issues.json files to proper individual issue files."""
    
    base_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories")
    migrated_count = 0
    
    print("Starting issue migration...")
    print("=" * 80)
    
    # Find all repositories
    for org_dir in base_path.iterdir():
        if not org_dir.is_dir():
            continue
            
        for repo_dir in org_dir.iterdir():
            if not repo_dir.is_dir():
                continue
                
            # Check for legacy issues.json file
            legacy_file = repo_dir / "issues.json"
            if not legacy_file.exists():
                continue
                
            repo_name = f"{org_dir.name}/{repo_dir.name}"
            print(f"\nFound issues.json in: {repo_name}")
            
            # Read the legacy file
            try:
                with open(legacy_file, 'r') as f:
                    legacy_issues = json.load(f)
                
                if not legacy_issues:
                    print(f"  - No issues to migrate, removing empty file")
                    legacy_file.unlink()
                    continue
                    
                print(f"  - Found {len(legacy_issues)} issues to migrate")
                
                # Create issues directory if it doesn't exist
                issues_dir = repo_dir / "issues"
                issues_dir.mkdir(exist_ok=True)
                
                # Get metadata for next issue number
                metadata_path = repo_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    next_issue_number = metadata.get("next_issue_number", 1)
                else:
                    next_issue_number = 1
                
                # Migrate each issue to individual files
                for idx, issue_data in enumerate(legacy_issues):
                    issue_number = idx + 1
                    
                    # Create issue dict with proper format
                    issue_dict = {
                        "id": str(issue_number),  # Simple ID for local issues
                        "number": issue_number,
                        "title": issue_data.get('title', f'Issue {issue_number}'),
                        "body": issue_data.get('body', ''),
                        "labels": issue_data.get('labels', []),
                        "state": "open",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "closed_at": None,
                        "assignee": None,
                        "html_url": f"local://issue/{issue_number}"
                    }
                    
                    # Save to individual file
                    issue_file = issues_dir / f"issue-{issue_number}.json"
                    with open(issue_file, 'w') as f:
                        json.dump(issue_dict, f, indent=2)
                    
                    print(f"    ✓ Migrated issue #{issue_number}: {issue_dict['title'][:50]}...")
                    migrated_count += 1
                
                # Update metadata with next issue number
                if metadata_path.exists():
                    metadata["next_issue_number"] = len(legacy_issues) + 1
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                # Remove the legacy file
                legacy_file.unlink()
                print(f"  - Removed legacy issues.json file")
                
            except Exception as e:
                print(f"  ❌ Error migrating issues: {e}")
                continue
    
    print("\n" + "=" * 80)
    print(f"Migration complete! Migrated {migrated_count} issues total.")
    print("All legacy issues.json files have been removed.")

if __name__ == "__main__":
    migrate_issues()