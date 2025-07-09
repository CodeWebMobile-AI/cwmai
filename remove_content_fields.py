#!/usr/bin/env python3
"""Remove all *_content fields from metadata.json files since they're stored as separate .md files."""

import json
import os
from pathlib import Path

def update_metadata_file(metadata_path):
    """Update a single metadata.json file to remove all content fields."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # List of content fields to remove
        content_fields = ['readme_content', 'architecture_content', 'monetization_content']
        
        # Track what was removed
        removed_fields = []
        
        # Remove each content field if it exists
        for field in content_fields:
            if field in metadata:
                del metadata[field]
                removed_fields.append(field)
        
        if removed_fields:
            # Write the updated metadata back
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Updated: {metadata_path}")
            print(f"   Removed: {', '.join(removed_fields)}")
            return True
        else:
            print(f"⏭️  Skipped (no content fields): {metadata_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {metadata_path}: {e}")
        return None

def main():
    """Update all metadata.json files in local repositories."""
    # Path to local repositories
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    if not local_repos_path.exists():
        print(f"Local repositories directory not found: {local_repos_path}")
        return
    
    print(f"🔍 Searching for metadata.json files in: {local_repos_path}")
    print("📝 Will remove: readme_content, architecture_content, monetization_content fields\n")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    # Find all metadata.json files
    for metadata_path in local_repos_path.rglob('metadata.json'):
        result = update_metadata_file(metadata_path)
        if result is True:
            updated_count += 1
        elif result is False:
            skipped_count += 1
        else:
            error_count += 1
    
    print("\n📊 Summary:")
    print(f"  - Updated: {updated_count} files")
    print(f"  - Skipped: {skipped_count} files")
    print(f"  - Errors: {error_count} files")

if __name__ == "__main__":
    main()