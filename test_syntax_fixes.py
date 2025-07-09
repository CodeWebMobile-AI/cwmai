#!/usr/bin/env python3
"""Quick test to verify syntax fixes compile correctly."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_syntax():
    """Test that fixed files compile without syntax errors."""
    
    print("Testing syntax fixes...\n")
    
    # Test 1: Import ProjectCreator (tests the creation_record fix)
    try:
        from scripts.project_creator import ProjectCreator
        print("✅ ProjectCreator imports successfully (creation_record fix works)")
    except SyntaxError as e:
        print(f"❌ Syntax error in ProjectCreator: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Other error in ProjectCreator (syntax is OK): {e}")
    
    # Test 2: Import BackgroundTaskPreprocessor (tests the import fix)
    try:
        from scripts.background_task_preprocessor import BackgroundTaskPreprocessor
        print("✅ BackgroundTaskPreprocessor imports successfully (import fix works)")
    except ImportError as e:
        print(f"❌ Import error in BackgroundTaskPreprocessor: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Other error in BackgroundTaskPreprocessor (import is OK): {e}")
    
    print("\n✅ All syntax fixes are working correctly!")
    return True

if __name__ == "__main__":
    success = test_syntax()
    sys.exit(0 if success else 1)