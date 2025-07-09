#!/usr/bin/env python3
"""Test script to verify all fixes are working correctly."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_verisafe_json():
    """Test that verisafe JSON files are valid."""
    print("\n1. Testing verisafe JSON files...")
    issues_dir = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip/local_repos/repositories/CodeWebMobile-AI/verisafe/issues")
    
    if not issues_dir.exists():
        print("   ❌ Issues directory not found")
        return False
    
    json_files = list(issues_dir.glob("issue-*.json"))
    print(f"   Found {len(json_files)} JSON files")
    
    all_valid = True
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                json.load(f)
            print(f"   ✅ {json_file.name} is valid JSON")
        except json.JSONDecodeError as e:
            print(f"   ❌ {json_file.name} has JSON error: {e}")
            all_valid = False
    
    return all_valid

async def test_architecture_generator():
    """Test the enhanced architecture generator."""
    print("\n2. Testing enhanced architecture generator...")
    
    try:
        from scripts.enhanced_architecture_generator import EnhancedArchitectureGenerator
        from scripts.http_ai_client import HTTPAIClient
        
        # Create a mock AI brain for testing
        class MockAIBrain:
            def __init__(self):
                self.client = HTTPAIClient()
                
            async def generate_enhanced_response(self, prompt, model='gemini'):
                # Test different response types
                if "infrastructure" in prompt.lower():
                    # Test case 1: Valid JSON response
                    return {"content": json.dumps({
                        "environments": {
                            "local": {"setup": "Docker", "services": ["web", "db"]},
                            "production": {"provider": "AWS", "services": ["ECS", "RDS"]}
                        },
                        "ci_cd": {"pipeline": "GitHub Actions", "stages": ["test", "deploy"]},
                        "scaling": {"strategy": "Auto-scaling"}
                    })}
                elif "testing" in prompt.lower():
                    # Test case 2: JSON in markdown block
                    return {"content": "```json\n" + json.dumps({
                        "test_types": {
                            "unit": {"coverage_target": "80%", "tools": ["PHPUnit"]},
                            "integration": {"coverage_target": "70%", "tools": ["Pest"]}
                        }
                    }) + "\n```"}
                elif "code examples" in prompt.lower():
                    # Test case 3: Returns PHP code instead of JSON (should retry)
                    # Count retries based on CRITICAL in prompt
                    if not hasattr(self, 'code_example_attempts'):
                        self.code_example_attempts = 0
                    self.code_example_attempts += 1
                    
                    if self.code_example_attempts >= 2:  # Second or later attempt
                        return {"content": json.dumps({
                            "controller_example": {
                                "description": "Example controller",
                                "code": "<?php\\nclass Controller {}"
                            }
                        })}
                    else:  # First attempt - return PHP code
                        return {"content": "```php\n<?php\nclass Controller {}\n```"}
                else:
                    # Default valid response
                    return {"content": json.dumps({"data": "test"})}
        
        mock_brain = MockAIBrain()
        generator = EnhancedArchitectureGenerator(mock_brain)
        
        # Test infrastructure generation
        print("   Testing infrastructure generation...")
        infra = await generator._generate_infrastructure({"name": "TestProject"})
        assert "environments" in infra, "Infrastructure missing environments"
        print("   ✅ Infrastructure generation works")
        
        # Test testing strategy generation
        print("   Testing testing strategy generation...")
        testing = await generator._generate_testing_strategy({"name": "TestProject"})
        assert "test_types" in testing, "Testing strategy missing test_types"
        print("   ✅ Testing strategy generation works")
        
        # Test code examples (with retry)
        print("   Testing code examples generation (with retry)...")
        try:
            # Reset attempts counter for new test
            mock_brain.code_example_attempts = 0
            
            examples = await generator._generate_code_examples({"name": "TestProject"})
            print(f"   Generated examples: {list(examples.keys())}")
            print(f"   Mock brain attempts: {mock_brain.code_example_attempts}")
            
            # Check for retry markers
            if "_retry_needed" in examples:
                print("   ❌ Got retry signal instead of actual content")
            
            assert "controller_example" in examples, "Code examples missing controller_example"
            print("   ✅ Code examples generation works with retry")
        except Exception as e:
            print(f"   Code examples error: {e}")
            print(f"   Examples result: {examples if 'examples' in locals() else 'Not generated'}")
            raise
        
        return True
        
    except Exception as e:
        print(f"   ❌ Architecture generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_work_discovery():
    """Test work discovery system."""
    print("\n3. Testing work discovery...")
    
    try:
        from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
        
        # Just verify the module imports correctly and has the right methods
        print("   ✅ Hybrid orchestrator imports successfully")
        
        # Check if local issue discovery method exists
        if hasattr(EnhancedHybridOrchestrator, '_get_local_issues'):
            print("   ✅ Local issue discovery method exists")
        else:
            print("   ❌ Local issue discovery method missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Work discovery test failed: {e}")
        return False

async def test_local_simulator():
    """Test local GitHub simulator."""
    print("\n4. Testing local GitHub simulator...")
    
    try:
        from scripts.local_github_simulator import LocalGitHubSimulator
        
        simulator = LocalGitHubSimulator()
        
        # Test getting issues from a repository with issues
        print("   Testing issue retrieval...")
        try:
            issues = simulator.get_issues("CodeWebMobile-AI/verisafe", state="open")
            print(f"   ✅ Found {len(issues)} issues in verisafe repository")
        except Exception as e:
            print(f"   ❌ Failed to get issues: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Local simulator test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing All Fixes")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Verisafe JSON", await test_verisafe_json()))
    results.append(("Architecture Generator", await test_architecture_generator()))
    results.append(("Work Discovery", await test_work_discovery()))
    results.append(("Local Simulator", await test_local_simulator()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)