#!/usr/bin/env python3
"""Check which repositories still need markdown parsing."""

import json
from pathlib import Path

def check_repositories():
    """Check which repositories need updating."""
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    needs_update = []
    already_updated = []
    
    for repo_path in local_repos_path.iterdir():
        if repo_path.is_dir() and not repo_path.name.startswith('.'):
            metadata_path = repo_path / 'metadata.json'
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if readme_content is properly structured
                readme_content = metadata.get('readme_content', {})
                monetization_content = metadata.get('monetization_content', {})
                
                # Check if it's the old format (has 'root' key or is a string)
                readme_needs_update = (
                    not readme_content or 
                    'root' in readme_content or 
                    isinstance(readme_content, str) or
                    'error' in readme_content or
                    'content' in readme_content  # This indicates error format
                )
                
                monetization_needs_update = (
                    not monetization_content or
                    'root' in monetization_content or 
                    isinstance(monetization_content, str) or
                    'error' in monetization_content or
                    'content' in monetization_content  # This indicates error format
                )
                
                if readme_needs_update or monetization_needs_update:
                    needs_update.append(repo_path.name)
                    print(f"❌ {repo_path.name} - Needs update")
                    if readme_needs_update:
                        print(f"   - README needs parsing")
                    if monetization_needs_update:
                        print(f"   - MONETIZATION needs parsing")
                else:
                    already_updated.append(repo_path.name)
                    print(f"✅ {repo_path.name} - Already updated")
    
    print(f"\n📊 Summary:")
    print(f"  - Already updated: {len(already_updated)} repositories")
    print(f"  - Need update: {len(needs_update)} repositories")
    
    return needs_update

if __name__ == "__main__":
    needs_update = check_repositories()
    
    if needs_update:
        print(f"\n📝 Repositories that need updating:")
        for repo in needs_update:
            print(f"  - {repo}")