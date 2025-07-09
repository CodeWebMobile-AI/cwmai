#!/usr/bin/env python3
"""Test Gemini markdown parsing on a single repository."""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_markdown_with_gemini import GeminiMarkdownParser

async def test_single_repo():
    """Test parsing a single repository."""
    parser = GeminiMarkdownParser()
    
    if not parser.gemini_api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return
    
    # Test with chronoai-time-tracker
    repo_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/chronoai-time-tracker")
    
    if not repo_path.exists():
        print(f"❌ Repository not found: {repo_path}")
        return
    
    print(f"📁 Testing with: {repo_path.name}")
    
    # Test README parsing
    readme_path = repo_path / "README.md"
    if readme_path.exists():
        print("\n📖 Testing README.md parsing...")
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   Content length: {len(content)} characters")
        
        result = await parser.parse_readme_to_dict(content)
        
        if result:
            print("✅ Successfully parsed README.md")
            print("\nStructured output:")
            import json
            print(json.dumps(result, indent=2)[:1000] + "...")
        else:
            print("❌ Failed to parse README.md")
    
    # Test MONETIZATION parsing
    monetization_path = repo_path / "MONETIZATION_STRATEGY.md"
    if monetization_path.exists():
        print("\n💰 Testing MONETIZATION_STRATEGY.md parsing...")
        with open(monetization_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   Content length: {len(content)} characters")
        
        result = await parser.parse_monetization_to_dict(content)
        
        if result:
            print("✅ Successfully parsed MONETIZATION_STRATEGY.md")
            print("\nStructured output:")
            import json
            print(json.dumps(result, indent=2)[:1000] + "...")
        else:
            print("❌ Failed to parse MONETIZATION_STRATEGY.md")

if __name__ == "__main__":
    # Set up environment
    os.environ['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    # Enable logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(test_single_repo())