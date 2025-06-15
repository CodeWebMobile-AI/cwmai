#!/usr/bin/env python3
"""
Debug the enum issue in continuous system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_enum_issue():
    """Test the specific enum issue."""
    print("Testing enum parsing...")
    
    try:
        from work_item_types import TaskPriority
        print("TaskPriority enum imported successfully")
        
        # Test creating with different values
        print("Testing enum creation:")
        
        p1 = TaskPriority(1)
        print(f"TaskPriority(1) = {p1}")
        
        p2 = TaskPriority.HIGH
        print(f"TaskPriority.HIGH = {p2}")
        
        # Test the problematic case
        try:
            p3 = TaskPriority("RESEARCH")
            print(f"TaskPriority('RESEARCH') = {p3}")
        except Exception as e:
            print(f"TaskPriority('RESEARCH') failed: {e}")
            
        try:
            p4 = TaskPriority["RESEARCH"]
            print(f"TaskPriority['RESEARCH'] = {p4}")
        except Exception as e:
            print(f"TaskPriority['RESEARCH'] failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_work_finder_issue():
    """Test the work finder specifically."""
    print("\nTesting work finder enum usage...")
    
    try:
        from intelligent_work_finder import IntelligentWorkFinder
        from ai_brain import IntelligentAIBrain
        
        # Create minimal state
        system_state = {'projects': {}}
        ai_brain = IntelligentAIBrain(system_state, {})
        
        work_finder = IntelligentWorkFinder(ai_brain, system_state)
        print("Work finder created successfully")
        
        return True
        
    except Exception as e:
        print(f"Work finder error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Debugging Enum Issue ===")
    test_enum_issue()
    test_work_finder_issue()