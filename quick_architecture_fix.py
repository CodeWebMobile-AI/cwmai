#!/usr/bin/env python3
"""Quick fix to add ARCHITECTURE.md to repositories missing it."""

import os
import asyncio
from github import Github
from scripts.ai_brain import AIBrain
from scripts.repository_analyzer import RepositoryAnalyzer
from scripts.architecture_generator import ArchitectureGenerator

async def fix_architecture_for_repo(repo_name: str):
    """Fix architecture for a single repository."""
    try:
        # Initialize components
        github_token = os.environ.get('GITHUB_TOKEN')
        g = Github(github_token)
        ai_brain = AIBrain()
        
        # Get repository
        repo = g.get_repo(f"CodeWebMobile-AI/{repo_name}")
        print(f"\n🔧 Fixing {repo_name}...")
        
        # Check if ARCHITECTURE.md already exists
        try:
            repo.get_contents("ARCHITECTURE.md")
            print(f"✅ {repo_name} already has ARCHITECTURE.md")
            return True
        except:
            pass
        
        # Generate architecture
        analyzer = RepositoryAnalyzer(github_token, ai_brain)
        generator = ArchitectureGenerator(github_token, ai_brain)
        
        # Analyze repository
        print(f"📊 Analyzing {repo_name}...")
        repo_analysis = await analyzer.analyze_repository(repo.full_name)
        
        # Generate architecture document
        print(f"🏗️ Generating architecture for {repo_name}...")
        architecture_data = await generator.generate_architecture_for_project(repo.full_name, repo_analysis)
        architecture_content = architecture_data.get('content', '')
        
        if architecture_content:
            # Create the file
            result = repo.create_file(
                "ARCHITECTURE.md",
                "Add comprehensive architecture documentation",
                architecture_content,
                branch="main"
            )
            print(f"✅ Created ARCHITECTURE.md for {repo_name}")
            return True
        else:
            print(f"❌ Failed to generate content for {repo_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {repo_name}: {e}")
        return False

async def main():
    """Main function to fix all repositories."""
    repos_to_fix = [
        "summarize-ai-mobile",
        "brand-guardian-ai",
        "reputation-ai",
        "eco-track-ai", 
        "ai-powered-inventory-sync",
        "community-connect-platform",
        "mindleap-ai",
        "vid-gen-ai",
        "review-sentry"
    ]
    
    print(f"🚀 Starting architecture fix for {len(repos_to_fix)} repositories...")
    
    # Process repositories one by one to avoid rate limits
    success_count = 0
    for repo_name in repos_to_fix:
        if await fix_architecture_for_repo(repo_name):
            success_count += 1
        # Small delay to avoid rate limits
        await asyncio.sleep(2)
    
    print(f"\n📊 Summary: Fixed {success_count}/{len(repos_to_fix)} repositories")

if __name__ == "__main__":
    asyncio.run(main())