#!/usr/bin/env python3
"""Fix chronoai-time-tracker monetization parsing."""

import asyncio
import json
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_markdown_with_gemini import GeminiMarkdownParser

async def fix_chronoai_monetization():
    """Fix the monetization parsing for chronoai-time-tracker."""
    
    repo_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI/chronoai-time-tracker")
    metadata_path = repo_path / 'metadata.json'
    monetization_path = repo_path / 'MONETIZATION_STRATEGY.md'
    
    print("📁 Fixing chronoai-time-tracker monetization parsing...")
    
    parser = GeminiMarkdownParser()
    
    if not parser.gemini_api_key:
        print("❌ GOOGLE_API_KEY not found")
        return
    
    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Read monetization content
    with open(monetization_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📄 Monetization content length: {len(content)} characters")
    
    # Try parsing with a simpler schema first
    simple_schema = {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "revenue_models": {
                "type": "array",
                "items": {"type": "string"}
            },
            "pricing_tiers": {
                "type": "array", 
                "items": {"type": "string"}
            },
            "market_analysis": {"type": "string"},
            "customer_acquisition": {
                "type": "array",
                "items": {"type": "string"}
            },
            "growth_projections": {"type": "string"},
            "key_metrics": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["executive_summary"]
    }
    
    print("💰 Attempting to parse with simplified schema...")
    
    try:
        result = await parser._call_gemini_structured(
            f"""Parse this MONETIZATION_STRATEGY.md content into a structured format.
            
Monetization Content:
{content}

Extract:
1. Executive summary as a string
2. Revenue models as an array of strings
3. Pricing tiers as an array of strings
4. Market analysis as a string summary
5. Customer acquisition strategies as an array of strings
6. Growth projections as a string summary
7. Key metrics as an array of strings

Keep the output simple and structured.""",
            simple_schema
        )
        
        if result:
            metadata['monetization_content'] = result
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("✅ Successfully parsed and saved monetization content")
            
            # Show a sample of the result
            print("\nParsed structure:")
            for key in list(result.keys())[:3]:
                value = result[key]
                if isinstance(value, list):
                    print(f"  - {key}: {len(value)} items")
                else:
                    print(f"  - {key}: {type(value).__name__}")
        else:
            print("❌ Failed to parse monetization content")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.environ['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(fix_chronoai_monetization())