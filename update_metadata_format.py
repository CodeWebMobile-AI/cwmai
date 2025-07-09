#!/usr/bin/env python3
"""Update all local repositories' metadata.json files to remove architecture_content field."""

import json
import os
from pathlib import Path

def update_metadata_file(metadata_path):
    """Update a single metadata.json file."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if architecture_content exists and remove it
        if 'architecture_content' in metadata:
            del metadata['architecture_content']
            
            # Write the updated metadata back
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Updated: {metadata_path}")
            return True
        else:
            print(f"⏭️  Skipped (no architecture_content): {metadata_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {metadata_path}: {e}")
        return False

def main():
    """Update all metadata.json files in local repositories."""
    # Path to local repositories
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    if not local_repos_path.exists():
        print(f"Local repositories directory not found: {local_repos_path}")
        return
    
    print(f"🔍 Searching for metadata.json files in: {local_repos_path}")
    
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