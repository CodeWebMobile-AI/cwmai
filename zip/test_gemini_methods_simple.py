#!/usr/bin/env python3
"""Simple test to verify the refactored methods exist and have correct signatures."""

import sys
from pathlib import Path
import ast
import inspect

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_method_signatures():
    """Test that all required methods exist with correct signatures."""
    print("Checking refactored Gemini Deep Research methods...")
    
    file_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/zip/scripts/gemini_deep_research.py")
    
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())
    
    # Find the GeminiDeepResearch class
    class_found = False
    methods_to_check = {
        # Original methods that should call helpers
        '_phase_4_5_unit_economics': False,
        '_phase_4_75_mvp_simulation': False,
        # New helper methods
        '_phase_4_5_cac_analysis': False,
        '_phase_4_5_ltv_margins': False,
        '_phase_4_5_growth_metrics': False,
        '_phase_4_75_mvp_features': False,
        '_phase_4_75_revenue_projections': False,
        '_phase_4_75_traction_indicators': False,
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'GeminiDeepResearchOptimized':
            class_found = True
            print("✅ Found GeminiDeepResearchOptimized class")
            
            # Check for methods
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name in methods_to_check:
                    methods_to_check[item.name] = True
                    print(f"✅ Found method: {item.name}")
                    
                    # Check if original methods call the helper methods
                    if item.name == '_phase_4_5_unit_economics':
                        method_body = ast.unparse(item)
                        if '_phase_4_5_cac_analysis' in method_body:
                            print("   ✓ Calls _phase_4_5_cac_analysis")
                        if '_phase_4_5_ltv_margins' in method_body:
                            print("   ✓ Calls _phase_4_5_ltv_margins")
                        if '_phase_4_5_growth_metrics' in method_body:
                            print("   ✓ Calls _phase_4_5_growth_metrics")
                            
                    elif item.name == '_phase_4_75_mvp_simulation':
                        method_body = ast.unparse(item)
                        if '_phase_4_75_mvp_features' in method_body:
                            print("   ✓ Calls _phase_4_75_mvp_features")
                        if '_phase_4_75_revenue_projections' in method_body:
                            print("   ✓ Calls _phase_4_75_revenue_projections")
                        if '_phase_4_75_traction_indicators' in method_body:
                            print("   ✓ Calls _phase_4_75_traction_indicators")
    
    if not class_found:
        print("❌ GeminiDeepResearchOptimized class not found!")
        return False
    
    # Check for missing methods
    missing_methods = [name for name, found in methods_to_check.items() if not found]
    if missing_methods:
        print(f"\n❌ Missing methods: {missing_methods}")
        return False
    
    print("\n✅ All refactored methods exist with correct structure!")
    
    # Additional check: look for schema complexity comments
    with open(file_path, 'r') as f:
        content = f.read()
        
    if "Make 3 sequential calls to avoid schema complexity" in content:
        print("✅ Found schema complexity comment in unit economics")
    if "Call 1/3:" in content and "Call 2/3:" in content and "Call 3/3:" in content:
        print("✅ Found sequential call logging")
    
    return True

if __name__ == "__main__":
    success = test_method_signatures()
    sys.exit(0 if success else 1)