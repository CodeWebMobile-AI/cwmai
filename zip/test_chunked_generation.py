#!/usr/bin/env python3
"""Test the chunked generation for project documentation."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.project_creator import ProjectCreator
from scripts.ai_brain import IntelligentAIBrain
from scripts.state_manager import StateManager

async def test_chunked_generation():
    """Test the chunked generation implementation."""
    print("Testing chunked generation for project documentation...")
    
    # Create AI brain
    state_manager = StateManager()
    state = state_manager.load_state()
    ai_brain = IntelligentAIBrain(state, {})
    
    # Create project creator with AI brain
    creator = ProjectCreator(None, ai_brain)  # None for local mode
    
    # Test project details
    test_details = {
        'name': 'test-chunked-generation',
        'description': 'A test project to verify chunked documentation generation',
        'problem_statement': 'Documentation generation can fail with very long prompts',
        'target_audience': 'Developers who need reliable documentation generation',
        'initial_features': ['Chunked Generation', 'Parallel Processing', 'Multi-Model Support'],
        'tech_stack': 'Laravel, React, TypeScript',
        'core_entities': ['Document', 'Chunk', 'Generator', 'Model']
    }
    
    try:
        print("\n1. Testing Architecture Generation (4 parallel chunks)...")
        architecture = await creator._generate_structured_architecture(test_details)
        print(f"✓ Architecture generated with {len(architecture)} sections")
        for key in architecture:
            if isinstance(architecture[key], dict):
                print(f"  - {key}: {len(architecture[key])} sub-sections")
            else:
                print(f"  - {key}: {type(architecture[key]).__name__}")
        
        print("\n2. Testing Monetization Strategy Generation (4 parallel chunks)...")
        monetization = await creator._generate_structured_monetization(test_details)
        print(f"✓ Monetization strategy generated with {len(monetization)} sections")
        for key in monetization:
            if isinstance(monetization[key], dict):
                print(f"  - {key}: {len(monetization[key])} sub-sections")
            else:
                print(f"  - {key}: {type(monetization[key]).__name__}")
        
        print("\n3. Testing README Generation (single prompt - reasonable length)...")
        readme = await creator._generate_structured_readme(test_details)
        print(f"✓ README generated with {len(readme)} sections")
        
        # Save results
        results = {
            'test_details': test_details,
            'architecture_sections': list(architecture.keys()),
            'monetization_sections': list(monetization.keys()),
            'readme_sections': list(readme.keys()),
            'success': True
        }
        
        with open('test_chunked_generation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ All tests passed! Chunked generation is working correctly.")
        print("Results saved to test_chunked_generation_results.json")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Set up environment
    os.environ['PYTHONPATH'] = str(Path(__file__).parent)
    
    # Run test
    success = asyncio.run(test_chunked_generation())
    sys.exit(0 if success else 1)