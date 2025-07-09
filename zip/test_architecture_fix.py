#!/usr/bin/env python3
"""Test the fixed architecture generation with detailed features."""

import asyncio
import json
from scripts.project_creator import ProjectCreator
from scripts.ai_brain_factory import AIBrainFactory
# Remove unused imports

async def test_architecture_generation():
    """Test that architecture now includes detailed feature descriptions."""
    
    # Initialize components
    ai_brain = AIBrainFactory.create_for_development()
    
    # ProjectCreator expects github_token and ai_brain
    project_creator = ProjectCreator(
        github_token=None,  # Local mode uses None
        ai_brain=ai_brain
    )
    
    # Test project details with features from market research
    test_details = {
        'name': 'test-project',
        'description': 'Test project for architecture generation',
        'problem_statement': 'Testing architecture generation with detailed features',
        'target_audience': 'Developers testing the system',
        'initial_features': [
            'AI-Powered Design Mockup Generator',
            'Automated Personalization Data Extraction',
            'Smart Inventory Allocation For Customization',
            'Predictive Order Completion Time Estimates',
            'Dynamic Proof Approval Workflow'
        ],
        'metadata': {
            'market_research': {
                'problem_description': 'Streamlining order fulfillment for personalized products',
                'affected_audience': 'Artisans and makers',
                'market_size': '$10M-$50M',
                'competition_level': 'Moderate',
                'current_solutions': 'Manual processes and generic e-commerce platforms'
            },
            'validation_metrics': {
                'validation_score': 8.5,
                'revenue_potential': '$100,000-$500,000 ARR'
            }
        }
    }
    
    print("🔧 Testing architecture generation...")
    print(f"📋 Features to include: {len(test_details['initial_features'])} features")
    
    try:
        # Generate architecture
        architecture = await project_creator._generate_structured_architecture(test_details)
        
        # Check if features are properly detailed
        print("\n✅ Architecture generated successfully!")
        
        # Verify key_features structure
        if 'key_features' in architecture:
            key_features = architecture['key_features']
            print(f"\n📊 Key Features Structure:")
            print(json.dumps(key_features, indent=2)[:500] + "...")
            
            # Check for the new detailed structure
            if 'ai_generated_features_from_market_research' in key_features:
                features_list = key_features['ai_generated_features_from_market_research']
                print(f"\n✅ Found {len(features_list)} detailed features")
                
                # Check first feature structure
                if features_list and isinstance(features_list[0], dict):
                    first_feature = features_list[0]
                    print(f"\n📋 First feature structure:")
                    print(f"  - Feature Name: {first_feature.get('feature_name')}")
                    print(f"  - Technical Description: {first_feature.get('technical_description', 'N/A')[:100]}...")
                    print(f"  - Architecture Impact: {first_feature.get('architecture_impact', 'N/A')[:100]}...")
                    print(f"  - Complexity: {first_feature.get('implementation_complexity')}")
                    print(f"  - Components: {len(first_feature.get('key_components', []))} components")
                else:
                    print("❌ Features are not in the expected detailed format")
            else:
                print("❌ 'ai_generated_features_from_market_research' key not found")
        else:
            print("❌ 'key_features' section not found in architecture")
            
        # Test markdown conversion
        print("\n🔄 Testing markdown conversion...")
        markdown = project_creator._transform_structured_architecture_to_markdown(architecture)
        
        # Check if markdown contains the features section
        if 'AI Generated Features From Market Research' in markdown:
            print("✅ Features section properly formatted in markdown")
            
            # Check for feature details in markdown
            feature_checks = ['Feature Name:', 'Technical Description:', 'Architecture Impact:']
            found_checks = sum(1 for check in feature_checks if check in markdown)
            print(f"✅ Found {found_checks}/{len(feature_checks)} expected feature fields in markdown")
        else:
            print("❌ Features section not found in markdown")
            
        print("\n✅ Architecture fix test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Architecture Generation Fix...")
    asyncio.run(test_architecture_generation())