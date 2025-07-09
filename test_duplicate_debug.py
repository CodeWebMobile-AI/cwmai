#!/usr/bin/env python3
"""
Debug duplicate prevention to see why similarities aren't detected
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.duplicate_prevention_system import DuplicatePreventionSystem
from scripts.ai_brain import IntelligentAIBrain

async def debug_duplicate_detection():
    """Debug why duplicate detection isn't working."""
    
    # Initialize system
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    ai_brain = IntelligentAIBrain()
    
    duplicate_system = DuplicatePreventionSystem(local_repos_path, ai_brain)
    
    print("🔍 Duplicate Detection Debug")
    print("=" * 60)
    
    # Load existing metadata
    existing_metadata = duplicate_system.load_all_metadata()
    
    # Test with a project very similar to ai-text-summarizer
    test_project = {
        'name': 'Smart Content Digest',
        'description': 'AI tool that summarizes articles, videos, and podcasts for busy professionals',
        'problem_statement': 'Information overload - people dont have time to consume all content',
        'target_market': 'Busy professionals and students',
        'initial_features': [
            'Text summarization',
            'Audio transcription and summary',
            'Video summary generation',
            'Key points extraction',
            'Multi-language support'
        ],
        'monetization_model': 'Freemium with premium features'
    }
    
    print("\n📋 Test Project:")
    print(f"Name: {test_project['name']}")
    print(f"Description: {test_project['description']}")
    print(f"Target: {test_project['target_market']}")
    
    # Get the ai-text-summarizer metadata
    ai_summarizer = existing_metadata.get('ai-text-summarizer')
    if ai_summarizer:
        print("\n📦 Existing AI Text Summarizer:")
        print(f"Name: {ai_summarizer.display_name}")
        print(f"Description: {ai_summarizer.description}")
        
        # Calculate similarity manually
        score, reasons = duplicate_system._calculate_similarity(test_project, ai_summarizer)
        
        print(f"\n🔢 Similarity Score: {score:.2%}")
        print(f"📝 Reasons: {reasons}")
        
        # Debug individual similarities
        print("\n🔍 Detailed Similarity Analysis:")
        
        # Name similarity
        name_sim = duplicate_system._string_similarity(
            test_project['name'].lower(),
            ai_summarizer.display_name.lower()
        )
        print(f"  Name similarity: {name_sim:.2%}")
        
        # Description similarity
        desc_sim = duplicate_system._string_similarity(
            test_project['description'].lower(),
            ai_summarizer.description.lower()
        )
        print(f"  Description similarity: {desc_sim:.2%}")
        
        # Problem/semantic similarity
        problem_sim = duplicate_system._semantic_similarity(
            test_project['problem_statement'],
            ai_summarizer.description
        )
        print(f"  Problem similarity: {problem_sim:.2%}")
        
        # Extract target from description
        target_from_desc = duplicate_system._extract_target_from_description(ai_summarizer.description)
        print(f"  Extracted target audience: '{target_from_desc}'")
        
        target_sim = duplicate_system._target_audience_similarity(
            test_project['target_market'],
            target_from_desc
        )
        print(f"  Target audience similarity: {target_sim:.2%}")
        
        # Feature overlap
        print("\n  Test features:")
        for f in test_project['initial_features']:
            print(f"    - {f}")
            
        print("\n  Existing features:")
        for f in ai_summarizer.key_features[:3]:
            print(f"    - {f}")
            
        feature_overlap = duplicate_system._feature_overlap(
            test_project['initial_features'],
            ai_summarizer.key_features
        )
        print(f"\n  Feature overlap: {feature_overlap:.2%}")
        
        # Show semantic analysis
        print("\n📊 Semantic Analysis:")
        
        # Extract key terms from test project
        test_terms = duplicate_system._extract_key_terms(test_project['description'].lower())
        print(f"  Test project key terms: {test_terms}")
        
        # Extract key terms from existing
        existing_terms = duplicate_system._extract_key_terms(ai_summarizer.description.lower())
        print(f"  Existing project key terms: {existing_terms}")
        
        # Show overlap
        overlap = test_terms.intersection(existing_terms)
        print(f"  Common terms: {overlap}")
        
    else:
        print("\n❌ Could not find ai-text-summarizer in metadata")


def main():
    """Run the debug."""
    try:
        asyncio.run(debug_duplicate_detection())
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()