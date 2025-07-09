#!/usr/bin/env python3
"""
Debug AI Response for Duplicate Detection
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.enhanced_duplicate_detector import EnhancedDuplicateDetector
from scripts.ai_brain import IntelligentAIBrain


async def debug_ai_response():
    """Debug what the AI is actually returning."""
    print("Debugging AI Response for Duplicate Detection")
    print("=" * 60)
    
    # Initialize
    ai_brain = IntelligentAIBrain()
    repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    detector = EnhancedDuplicateDetector(repos_path, ai_brain, github_token=None)
    
    # Load projects
    detector.load_all_metadata()
    print(f"Loaded {len(detector._metadata_cache)} projects")
    
    # Simple test project
    test_project = {
        "name": "subscription-tracker-pro",
        "description": "Track and manage all your subscriptions",
        "problem_statement": "People forget about subscriptions and waste money"
    }
    
    # Simplified prompt and schema
    simple_prompt = f"""
    Is this new project a duplicate of any existing projects?

    New project:
    - Name: {test_project['name']}
    - Description: {test_project['description']}
    - Problem: {test_project['problem_statement']}

    Existing projects include:
    - sub-wise: AI-powered subscription tracking and management app
    - subscrip-ai: AI-enhanced subscription manager

    Respond with a simple JSON object with these fields:
    - is_duplicate (boolean): true if this is similar to existing projects
    - confidence (number): 0.0 to 1.0
    - similar_to (string): name of the most similar project
    - reason (string): brief explanation
    """
    
    simple_schema = {
        "type": "object",
        "properties": {
            "is_duplicate": {"type": "boolean"},
            "confidence": {"type": "number"},
            "similar_to": {"type": "string"},
            "reason": {"type": "string"}
        }
    }
    
    # Test with AI
    print("\nSending simplified request to AI...")
    
    try:
        response = await ai_brain.generate_enhanced_response(simple_prompt, response_schema=simple_schema)
        
        print("\nFull response object:")
        print(json.dumps(response, indent=2))
        
        if response and response.get('structured_output'):
            print("\nStructured output detected!")
            print("Parsed content:", response.get('parsed_content'))
        
        if response and response.get('content'):
            print("\nRaw content:")
            print(response['content'][:500])
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run debug."""
    try:
        await debug_ai_response()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())