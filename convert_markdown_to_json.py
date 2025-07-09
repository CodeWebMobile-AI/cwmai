#!/usr/bin/env python3
"""Convert markdown files to JSON format and update metadata.json."""

import json
import subprocess
import sys
from pathlib import Path

# Install markdown-to-json if not already installed
try:
    import markdown_to_json
except ImportError:
    print("Installing markdown-to-json package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown-to-json"])
    import markdown_to_json

def convert_markdown_file_to_json(file_path: Path) -> dict:
    """Convert a markdown file to JSON structure."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Convert markdown to JSON
        json_data = markdown_to_json.dictify(content)
        return json_data
    except Exception as e:
        print(f"  ⚠️  Error converting {file_path.name}: {e}")
        return {}

def update_repository_metadata(repo_path: Path) -> bool:
    """Update a repository's metadata.json with converted markdown content."""
    metadata_path = repo_path / 'metadata.json'
    readme_path = repo_path / 'README.md'
    monetization_path = repo_path / 'MONETIZATION_STRATEGY.md'
    
    if not metadata_path.exists():
        print(f"  ❌ No metadata.json found")
        return False
    
    try:
        # Load existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        updated = False
        
        # Convert README.md if it exists
        if readme_path.exists():
            readme_json = convert_markdown_file_to_json(readme_path)
            if readme_json:
                metadata['readme_content'] = readme_json
                print(f"  ✅ Converted README.md to JSON")
                updated = True
        else:
            print(f"  ⚠️  No README.md found")
        
        # Convert MONETIZATION_STRATEGY.md if it exists
        if monetization_path.exists():
            monetization_json = convert_markdown_file_to_json(monetization_path)
            if monetization_json:
                metadata['monetization_content'] = monetization_json
                print(f"  ✅ Converted MONETIZATION_STRATEGY.md to JSON")
                updated = True
        else:
            print(f"  ⚠️  No MONETIZATION_STRATEGY.md found")
        
        # Save updated metadata if changes were made
        if updated:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        
        return False
        
    except Exception as e:
        print(f"  ❌ Error updating metadata: {e}")
        return False

def main():
    """Convert markdown files to JSON for all repositories."""
    # Path to local repositories
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    if not local_repos_path.exists():
        print(f"Local repositories directory not found: {local_repos_path}")
        return
    
    print(f"🔍 Processing repositories in: {local_repos_path}")
    print("📝 Converting .md files to JSON format\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # Process each repository
    for repo_path in local_repos_path.iterdir():
        if repo_path.is_dir() and not repo_path.name.startswith('.'):
            print(f"\n📁 Processing: {repo_path.name}")
            result = update_repository_metadata(repo_path)
            if result is True:
                success_count += 1
            elif result is False:
                skip_count += 1
            else:
                error_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  - Successfully updated: {success_count} repositories")
    print(f"  - Skipped: {skip_count} repositories")
    print(f"  - Errors: {error_count} repositories")

if __name__ == "__main__":
    main()