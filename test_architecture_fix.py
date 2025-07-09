#!/usr/bin/env python3
"""Test that architecture detection always returns LARAVEL_REACT"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.task_types import ArchitectureType
from scripts.hybrid_orchestrator_enhanced import EnhancedHybridOrchestrator
from scripts.project_context_model import ProjectContext, TechnicalContext

def test_architecture_enum():
    """Test that ArchitectureType has LARAVEL_REACT"""
    print("Testing ArchitectureType enum...")
    print(f"Available architecture types: {[e.value for e in ArchitectureType]}")
    print(f"LARAVEL_REACT value: {ArchitectureType.LARAVEL_REACT.value}")
    
    # Test that MONOLITH doesn't exist
    try:
        _ = ArchitectureType.MONOLITH
        print("ERROR: MONOLITH still exists in enum!")
    except AttributeError:
        print("✓ MONOLITH correctly does not exist in enum")

def test_detect_architecture():
    """Test that _detect_architecture always returns LARAVEL_REACT"""
    print("\nTesting _detect_architecture method...")
    
    # Create orchestrator instance
    orchestrator = EnhancedHybridOrchestrator(max_workers=1)
    
    # Test with empty analysis
    result = orchestrator._detect_architecture({})
    print(f"Empty analysis result: {result}")
    assert result == ArchitectureType.LARAVEL_REACT, f"Expected LARAVEL_REACT, got {result}"
    
    # Test with various file patterns (should still return LARAVEL_REACT)
    test_cases = [
        {"structure": {"files": ["package.json", "composer.json"]}},
        {"structure": {"files": ["django_app.py", "vue.config.js"]}},
        {"structure": {"files": ["next.config.js", "pages/index.js"]}},
        {"structure": {"files": ["serverless.yml", "lambda.py"]}},
        {"structure": {"files": ["microservice1/", "docker-compose.yml"]}},
        {"structure": {"files": ["setup.py", "__init__.py"]}},
    ]
    
    for i, test_case in enumerate(test_cases):
        result = orchestrator._detect_architecture(test_case)
        print(f"Test case {i+1} result: {result}")
        assert result == ArchitectureType.LARAVEL_REACT, f"Test case {i+1}: Expected LARAVEL_REACT, got {result}"
    
    print("✓ All architecture detection tests passed!")

def test_default_context():
    """Test that default context uses LARAVEL_REACT"""
    print("\nTesting default ProjectContext...")
    
    # Create a technical context with defaults
    tech_context = TechnicalContext()
    print(f"Default architecture type: {tech_context.architecture_type}")
    assert tech_context.architecture_type == ArchitectureType.LARAVEL_REACT, \
        f"Expected default to be LARAVEL_REACT, got {tech_context.architecture_type}"
    
    print("✓ Default context test passed!")

if __name__ == "__main__":
    print("=== Testing Architecture Fix ===\n")
    
    try:
        test_architecture_enum()
        test_detect_architecture()
        test_default_context()
        
        print("\n✅ All tests passed! The MONOLITH error should be fixed.")
        print("\nThe system will now correctly identify all projects as LARAVEL_REACT architecture.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()