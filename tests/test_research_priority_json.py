#!/usr/bin/env python3
"""
Test script to verify ResearchPriority JSON serialization fix.
"""

import json
import sys
import os
sys.path.append('scripts')

from research_scheduler import ResearchPriority, ResearchJSONEncoder


def test_research_priority_serialization():
    """Test that ResearchPriority enum can be JSON serialized."""
    
    # Test data that would previously fail
    test_data = {
        'priority': ResearchPriority.IMMEDIATE,
        'secondary_priority': ResearchPriority.HIGH,
        'priority_list': [
            ResearchPriority.CRITICAL,
            ResearchPriority.MEDIUM,
            ResearchPriority.LOW
        ],
        'nested': {
            'priority': ResearchPriority.IMMEDIATE,
            'metadata': {
                'fallback_priority': ResearchPriority.LOW
            }
        }
    }
    
    print("Testing ResearchPriority JSON serialization...")
    
    # Test json.dumps
    try:
        json_str = json.dumps(test_data, cls=ResearchJSONEncoder, indent=2)
        print("✓ json.dumps() works with ResearchJSONEncoder")
        
        # Parse it back
        parsed = json.loads(json_str)
        assert parsed['priority'] == 'immediate'
        assert parsed['secondary_priority'] == 'high'
        assert parsed['priority_list'] == ['critical', 'medium', 'low']
        assert parsed['nested']['priority'] == 'immediate'
        assert parsed['nested']['metadata']['fallback_priority'] == 'low'
        print("✓ Round-trip serialization/deserialization works")
        
    except Exception as e:
        print(f"✗ json.dumps() failed: {e}")
        return False
    
    # Test json.dump to file
    try:
        test_file = '/tmp/test_research_priority.json'
        with open(test_file, 'w') as f:
            json.dump(test_data, f, cls=ResearchJSONEncoder, indent=2)
        
        # Read it back
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data['priority'] == 'immediate'
        print("✓ json.dump() to file works with ResearchJSONEncoder")
        
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        print(f"✗ json.dump() to file failed: {e}")
        return False
    
    # Test without ResearchJSONEncoder (should fail)
    try:
        json.dumps(test_data)
        print("✗ Unexpected: json.dumps() worked without ResearchJSONEncoder")
        return False
    except TypeError as e:
        if "not JSON serializable" in str(e):
            print("✓ Confirmed: json.dumps() fails without ResearchJSONEncoder (as expected)")
        else:
            print(f"✗ Unexpected error without ResearchJSONEncoder: {e}")
            return False
    
    print("\n✅ All tests passed! ResearchPriority JSON serialization is fixed.")
    return True


if __name__ == "__main__":
    success = test_research_priority_serialization()
    sys.exit(0 if success else 1)