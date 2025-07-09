#!/usr/bin/env python3
"""
Quick test of JSON serialization fix.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from scripts.custom_json_encoder import safe_json_dumps, CustomJSONEncoder
from scripts.gemini_deep_research import MarketOpportunity


def test_json_encoder():
    """Test the custom JSON encoder."""
    print("🧪 Testing Custom JSON Encoder")
    print("=" * 50)
    
    # Create a test MarketOpportunity
    opp = MarketOpportunity(
        problem_description="Test problem",
        affected_audience="Test users",
        current_solutions="Current solutions",
        opportunity="Test opportunity",
        monetization_model="SaaS",
        market_size="$1B",
        competition_level="Medium",
        validation_sources=["source1", "source2"],
        key_insights=["insight1"],
        implementation_difficulty="Medium",
        potential_revenue="$100k/month",
        unique_advantages=["advantage1"],
        search_evidence={"test": "data"}
    )
    
    # Test direct serialization with custom encoder
    try:
        json_str = safe_json_dumps(opp, indent=2)
        print("✓ MarketOpportunity serialization successful")
        print(f"  JSON length: {len(json_str)} chars")
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        print(f"✓ JSON parsing successful")
        print(f"  Keys: {list(parsed.keys())[:5]}...")
        
    except Exception as e:
        print(f"✗ Serialization failed: {e}")
        return False
        
    # Test nested object serialization
    nested_data = {
        "id": "test123",
        "metadata": {
            "opportunity": opp,
            "list_of_opps": [opp, opp],
            "nested": {
                "deep": {
                    "opp": opp
                }
            }
        }
    }
    
    try:
        json_str = safe_json_dumps(nested_data, indent=2)
        print("\n✓ Nested object serialization successful")
        print(f"  JSON length: {len(json_str)} chars")
        
    except Exception as e:
        print(f"\n✗ Nested serialization failed: {e}")
        return False
        
    print("\n✅ All JSON encoder tests passed!")
    return True


if __name__ == "__main__":
    success = test_json_encoder()
    sys.exit(0 if success else 1)