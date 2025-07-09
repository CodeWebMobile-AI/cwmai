#!/usr/bin/env python3
"""Test that AI-generated features are properly displayed in project documentation."""

import json
from pathlib import Path

def check_features_in_project(project_path: Path):
    """Check if AI-generated features appear in project documentation."""
    
    # Read metadata to get the AI-generated features
    metadata_path = project_path / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ No metadata.json found in {project_path}")
        return False
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get the initial features from metadata
    initial_features = metadata.get('project_details', {}).get('initial_features', [])
    if not initial_features:
        print(f"❌ No initial_features found in metadata for {project_path}")
        return False
    
    print(f"\n📊 Project: {project_path.name}")
    print(f"🎯 AI-Generated Features from metadata:")
    for i, feature in enumerate(initial_features, 1):
        print(f"   {i}. {feature}")
    
    # Check README.md
    readme_path = project_path / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text()
        print("\n📄 Checking README.md...")
        
        features_found = []
        for feature in initial_features:
            if feature in readme_content:
                features_found.append(feature)
                print(f"   ✅ Found: {feature}")
            else:
                print(f"   ❌ Missing: {feature}")
        
        if len(features_found) == len(initial_features):
            print("   ✨ All AI-generated features found in README!")
        else:
            print(f"   ⚠️  Only {len(features_found)}/{len(initial_features)} features found")
    
    # Check ARCHITECTURE.md
    arch_path = project_path / "ARCHITECTURE.md"
    if arch_path.exists():
        arch_content = arch_path.read_text()
        print("\n📐 Checking ARCHITECTURE.md...")
        
        # Check for dedicated features section
        if "## Key Features" in arch_content or "### AI-Generated Features" in arch_content:
            print("   ✅ Found dedicated Features section")
        else:
            print("   ❌ No dedicated Features section found")
        
        features_found = []
        for feature in initial_features:
            if feature in arch_content:
                features_found.append(feature)
                print(f"   ✅ Found: {feature}")
            else:
                print(f"   ❌ Missing: {feature}")
        
        if len(features_found) == len(initial_features):
            print("   ✨ All AI-generated features found in ARCHITECTURE!")
        else:
            print(f"   ⚠️  Only {len(features_found)}/{len(initial_features)} features found")
    
    return True

def main():
    """Check all local repositories for proper feature display."""
    local_repos_path = Path("local_repos/repositories/CodeWebMobile-AI")
    
    if not local_repos_path.exists():
        print("❌ No local repositories found")
        return
    
    projects_checked = 0
    for project_dir in local_repos_path.iterdir():
        if project_dir.is_dir():
            check_features_in_project(project_dir)
            projects_checked += 1
    
    print(f"\n✅ Checked {projects_checked} projects")

if __name__ == "__main__":
    main()