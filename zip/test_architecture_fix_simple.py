#!/usr/bin/env python3
"""Simple test to demonstrate the architecture fix works."""

import asyncio
import json
from scripts.project_creator import ProjectCreator
from scripts.ai_brain_factory import AIBrainFactory

async def test_architecture_generation():
    """Test that architecture now includes detailed feature descriptions."""
    
    # Initialize components
    ai_brain = AIBrainFactory.create_for_development()
    project_creator = ProjectCreator(github_token=None, ai_brain=ai_brain)
    
    # Test project details
    test_details = {
        'name': 'test-architecture-fix',
        'description': 'Testing architecture generation fix',
        'problem_statement': 'Testing if features are properly detailed',
        'target_audience': 'Developers',
        'initial_features': [
            'Smart Feature One',
            'Advanced Feature Two'
        ],
        'metadata': {
            'market_research': {
                'problem_description': 'Testing problem',
                'affected_audience': 'Test users',
                'market_size': '$1M-$5M'
            }
        }
    }
    
    print("🧪 Testing architecture generation fix...")
    print(f"📋 Features to test: {test_details['initial_features']}")
    
    try:
        # Generate architecture
        architecture = await project_creator._generate_structured_architecture(test_details)
        
        print("\n✅ Architecture generated successfully!")
        
        # Check the key_features structure
        if 'key_features' in architecture:
            key_features = architecture['key_features']
            print(f"\n🔍 key_features type: {type(key_features)}")
            print(f"🔍 key_features keys: {list(key_features.keys()) if isinstance(key_features, dict) else 'Not a dict'}")
            
            # Print the whole key_features structure for debugging
            print("\n🔍 Full key_features structure:")
            print(json.dumps(key_features, indent=2)[:500] + "...")
            
            if isinstance(key_features, dict) and 'ai_generated_features_from_market_research' in key_features:
                features = key_features['ai_generated_features_from_market_research']
                print(f"\n✅ Found {len(features)} detailed features")
                
                # Print first feature structure
                if features and isinstance(features[0], dict):
                    print("\n📋 First feature structure:")
                    for key, value in features[0].items():
                        print(f"  - {key}: {str(value)[:80]}...")
                    
                    print("\n✅ Fix confirmed: Features now have detailed technical descriptions!")
                else:
                    print("❌ Features are not in detailed format")
            else:
                print("❌ 'ai_generated_features_from_market_research' key not found")
        else:
            print("❌ 'key_features' section not found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_architecture_generation())