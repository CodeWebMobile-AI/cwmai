#!/usr/bin/env python3
"""
Clear the research cache and start fresh.
Backs up existing cache before clearing.
"""

import os
import json
from datetime import datetime
from pathlib import Path


def clear_research_cache():
    """Clear the research cache and start fresh."""
    # Check both possible locations
    locations = [
        "research_cache.json",
        "zip/research_cache.json"
    ]
    
    cache_cleared = False
    
    for cache_file in locations:
        if os.path.exists(cache_file):
            print(f"\n📁 Found cache at: {cache_file}")
            
            # Read existing cache
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Show current cache status
            problems_count = len(data.get('problems', []))
            print(f"📊 Current cache has {problems_count} problems")
            
            if problems_count > 0:
                # Create backup
                backup_file = f"{cache_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"💾 Backed up to: {backup_file}")
                
                # Show problem details
                print("\n📋 Problems being cleared:")
                for i, problem in enumerate(data.get('problems', []), 1):
                    print(f"{i}. {problem.get('name', 'Unknown')}")
                    print(f"   Target: {problem.get('target_audience', 'Unknown')}")
                    print(f"   Features: {len(problem.get('features', []))} features")
            
            # Create empty cache
            empty_cache = {
                "problems": [],
                "last_research_time": None
            }
            
            with open(cache_file, 'w') as f:
                json.dump(empty_cache, f, indent=2)
            
            print(f"\n✅ Cache cleared at: {cache_file}")
            cache_cleared = True
    
    if not cache_cleared:
        print("❌ No cache file found in current directory or zip/")
        print("Creating new empty cache in zip/")
        
        # Create zip directory if it doesn't exist
        os.makedirs("zip", exist_ok=True)
        
        # Create empty cache
        empty_cache = {
            "problems": [],
            "last_research_time": None
        }
        
        with open("zip/research_cache.json", 'w') as f:
            json.dump(empty_cache, f, indent=2)
        
        print("✅ Created new empty cache at: zip/research_cache.json")


if __name__ == "__main__":
    print("🧹 Research Cache Cleaner")
    print("=" * 50)
    
    clear_research_cache()
    
    print("\n🎉 Cache clearing complete!")
    print("The system will now generate fresh research with proper features.")