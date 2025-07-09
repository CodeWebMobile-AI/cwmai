#!/usr/bin/env python3
"""Test the enhanced architecture generation."""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.ai_brain import IntelligentAIBrain
from scripts.project_creator import ProjectCreator
from scripts.storage_mode import StorageMode


async def test_enhanced_architecture():
    """Test the enhanced architecture generation."""
    print("🧪 Testing Enhanced Architecture Generation")
    print("=" * 50)
    
    # Initialize AI brain
    print("🧠 Initializing AI Brain...")
    ai_brain = IntelligentAIBrain()
    
    # Create test project details
    test_details = {
        'name': 'test-architecture-project',
        'problem_statement': 'Small businesses struggle to manage their inventory efficiently across multiple sales channels',
        'target_audience': 'Small retail businesses with 1-50 employees',
        'initial_features': [
            'Inventory tracking across channels',
            'Low stock alerts',
            'Sales analytics',
            'Supplier management',
            'Barcode scanning'
        ],
        'value_proposition': 'AI-powered inventory management that prevents stockouts and reduces waste',
        'metadata': {
            'market_research': {
                'problem_description': 'Small retailers lose 20% of potential sales due to stockouts and waste 15% of inventory due to overstocking',
                'affected_audience': 'Small retail businesses, especially those selling on multiple platforms',
                'market_size': '$5.2B addressable market in North America',
                'current_solutions': 'Excel spreadsheets, basic POS systems, manual counting',
                'unique_angle': 'AI predictions based on local events, weather, and trends',
                'key_innovation': 'Predictive AI that learns from local market patterns'
            },
            'validation_metrics': {
                'validation_score': 9.2
            }
        }
    }
    
    # Initialize project creator
    print("📁 Initializing Project Creator...")
    # Set storage mode
    os.environ['STORAGE_MODE'] = 'local'
    creator = ProjectCreator(
        github_token=None,
        ai_brain=ai_brain
    )
    
    # Test architecture generation
    print("\n🏗️ Generating Enhanced Architecture...")
    try:
        architecture = await creator._generate_structured_architecture(test_details)
        
        # Verify all sections are present
        print("\n✅ Checking Required Sections:")
        required_sections = [
            'executive_summary',
            'system_context', 
            'business_rules',
            'domain_models',
            'state_machines',
            'api_documentation',
            'implementation_examples',
            'infrastructure',
            'testing_strategy',
            'security',
            'performance',
            'monitoring'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section in architecture:
                print(f"  ✓ {section}")
            else:
                print(f"  ✗ {section} - MISSING")
                missing_sections.append(section)
        
        # Check domain model completeness
        print(f"\n📊 Domain Models: {len(architecture.get('domain_models', []))} defined")
        if 'domain_models' in architecture:
            for model in architecture['domain_models'][:3]:  # Show first 3
                print(f"  - {model.get('name', 'Unknown')}: {len(model.get('fields', []))} fields")
        
        # Check business rules
        if 'business_rules' in architecture:
            print(f"\n📋 Business Rules: {len(architecture['business_rules'].get('core_rules', {}))} categories")
            print(f"  - Edge cases defined: {len(architecture['business_rules'].get('edge_cases', []))}")
        
        # Check state machines
        if 'state_machines' in architecture:
            print(f"\n🔄 State Machines: {len(architecture['state_machines'])} defined")
            for entity, machine in list(architecture['state_machines'].items())[:2]:
                states = len(machine.get('states', []))
                transitions = len(machine.get('transitions', []))
                print(f"  - {entity}: {states} states, {transitions} transitions")
        
        # Save the generated architecture
        output_dir = Path('test_output')
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON
        json_path = output_dir / 'test_architecture.json'
        with open(json_path, 'w') as f:
            json.dump(architecture, f, indent=2)
        print(f"\n💾 Saved JSON to: {json_path}")
        
        # Convert to markdown
        try:
            from scripts.architecture_markdown_converter import ArchitectureMarkdownConverter
            converter = ArchitectureMarkdownConverter()
            markdown_content = converter.convert_to_markdown(architecture)
            
            md_path = output_dir / 'test_architecture.md'
            with open(md_path, 'w') as f:
                f.write(markdown_content)
            print(f"📄 Saved Markdown to: {md_path}")
            
            # Show preview
            print("\n📋 Markdown Preview (first 1000 chars):")
            print("-" * 50)
            print(markdown_content[:1000])
            print("...")
            
        except Exception as e:
            print(f"⚠️ Markdown conversion error: {e}")
        
        # Final assessment
        print("\n🎯 Architecture Completeness Assessment:")
        total_sections = len(required_sections)
        present_sections = total_sections - len(missing_sections)
        completeness = (present_sections / total_sections) * 100
        
        print(f"  Completeness: {completeness:.1f}%")
        
        if completeness >= 90:
            print("  ✅ Architecture is comprehensive!")
        elif completeness >= 70:
            print("  ⚠️ Architecture is mostly complete but needs work")
        else:
            print("  ❌ Architecture is incomplete")
        
        # Check balance
        if architecture:
            section_sizes = {k: len(str(v)) for k, v in architecture.items()}
            total_size = sum(section_sizes.values())
            print(f"\n📊 Section Balance (total size: {total_size} chars):")
            for section, size in sorted(section_sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = (size / total_size) * 100
                print(f"  - {section}: {percentage:.1f}% ({size} chars)")
        
        return completeness >= 90
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    success = await test_enhanced_architecture()
    
    if success:
        print("\n✅ Enhanced architecture generation test PASSED!")
        print("\n📋 Next Steps:")
        print("1. Check test_output/test_architecture.md for the full document")
        print("2. Verify all sections have appropriate detail")
        print("3. Confirm business rules and state machines are present")
        print("4. Check that all domain models are defined")
    else:
        print("\n❌ Enhanced architecture generation test FAILED!")
        print("Please check the errors above and fix the implementation.")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())