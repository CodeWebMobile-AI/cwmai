#!/usr/bin/env python3
"""Update only the remaining repositories that need markdown parsing."""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_markdown_with_gemini import GeminiMarkdownParser

async def update_specific_repos():
    """Update only the repositories that need it."""
    
    # Repositories that need updating
    repos_to_update = [
        "chronoai-time-tracker",
        "mindful-ai-platform", 
        "skill-bridge-ai"
    ]
    
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    print(f"🔍 Updating {len(repos_to_update)} remaining repositories")
    print("🤖 Using Gemini AI to parse .md files into structured JSON format\n")
    
    # Initialize parser
    parser = GeminiMarkdownParser()
    
    if not parser.gemini_api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return
    
    success_count = 0
    error_count = 0
    
    for repo_name in repos_to_update:
        repo_path = local_repos_path / repo_name
        
        if repo_path.exists():
            print(f"\n📁 Processing: {repo_name}")
            result = await parser.update_repository_metadata(repo_path)
            
            if result:
                success_count += 1
            else:
                error_count += 1
            
            # Add delay to avoid rate limiting
            await asyncio.sleep(2)
        else:
            print(f"\n❌ Repository not found: {repo_name}")
            error_count += 1
    
    print(f"\n📊 Final Summary:")
    print(f"  - Successfully updated: {success_count} repositories")
    print(f"  - Errors: {error_count} repositories")

if __name__ == "__main__":
    # Set up environment
    os.environ['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    # Enable logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(update_specific_repos())