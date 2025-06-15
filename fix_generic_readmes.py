#!/usr/bin/env python3
"""Fix all generic READMEs with proper project-specific content."""

import os
import asyncio
from github import Github
from scripts.ai_brain import AIBrain
from scripts.http_ai_client import HTTPAIClient
import json


async def generate_readme_content(repo_name: str, description: str, ai_client: HTTPAIClient) -> str:
    """Generate a proper README for the repository."""
    
    prompt = f"""
    Generate a comprehensive README.md for the following project:
    
    Repository Name: {repo_name}
    Description: {description}
    
    The repository name is HIGHLY DESCRIPTIVE of what the project does. Use it to understand the project's purpose.
    
    Create a professional README that includes:
    1. Project title and description (based on the repo name and description)
    2. Key Features (infer from the name/description)
    3. Tech Stack (Laravel backend, React frontend, TypeScript, Tailwind CSS)
    4. Prerequisites
    5. Installation instructions
    6. Development setup
    7. Usage examples
    8. API documentation structure
    9. Contributing guidelines
    10. License (MIT)
    
    Make it specific to THIS project, not generic. The content should reflect what the project actually does based on its name.
    
    Return ONLY the markdown content, no explanations.
    """
    
    response = await ai_client.generate_enhanced_response(prompt)
    
    if isinstance(response, dict):
        content = response.get('content', '')
    else:
        content = str(response)
    
    return content.strip()


async def fix_repository_readme(repo, ai_client: HTTPAIClient):
    """Fix a single repository's README."""
    try:
        print(f"\n📝 Fixing README for {repo.name}...")
        
        # Generate new README content
        readme_content = await generate_readme_content(
            repo.name,
            repo.description or f"{repo.name} application",
            ai_client
        )
        
        if not readme_content:
            print(f"❌ Failed to generate content for {repo.name}")
            return False
        
        # Update the README
        try:
            readme_file = repo.get_contents("README.md")
            repo.update_file(
                "README.md",
                f"Update README with project-specific content for {repo.name}",
                readme_content,
                readme_file.sha,
                branch="main"
            )
            print(f"✅ Updated README for {repo.name}")
            return True
        except Exception as e:
            print(f"❌ Error updating README for {repo.name}: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {repo.name}: {e}")
        return False


async def main():
    """Fix all repositories with generic READMEs."""
    
    # Repositories that need fixing
    repos_to_fix = [
        "project-analytics-dashboard",
        "business-analytics-dashboard", 
        "summarize-ai-mobile",
        "mindleap-ai",
        "vid-gen-ai",
        "review-sentry"
    ]
    
    print(f"🚀 Fixing {len(repos_to_fix)} repositories with generic READMEs...\n")
    
    # Initialize GitHub and AI
    github_token = os.environ.get('GITHUB_TOKEN')
    g = Github(github_token)
    
    # Initialize AI client directly
    ai_client = HTTPAIClient()
    
    success_count = 0
    
    for repo_name in repos_to_fix:
        try:
            repo = g.get_repo(f"CodeWebMobile-AI/{repo_name}")
            if await fix_repository_readme(repo, ai_client):
                success_count += 1
            
            # Small delay to avoid rate limits
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ Error accessing {repo_name}: {e}")
    
    print(f"\n📊 Summary: Fixed {success_count}/{len(repos_to_fix)} repositories")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())