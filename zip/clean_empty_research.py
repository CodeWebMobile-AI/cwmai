#!/usr/bin/env python3
"""
Clean up empty research entries from research memory.
"""

import json
from pathlib import Path
from datetime import datetime


def clean_research_memory():
    """Remove empty research entries from research_memory.json."""
    
    # Path to research memory file
    memory_file = Path("research_knowledge/research_summary.json")
    
    if not memory_file.exists():
        print("❌ research_memory.json not found")
        return
    
    # Load the file
    with open(memory_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loading research memory...")
    print(f"Total research sessions: {data.get('total_research_sessions', 0)}")
    print(f"Total problems discovered: {data.get('total_problems_discovered', 0)}")
    
    # Find empty entries
    all_problems = data.get('all_problems_discovered', [])
    empty_entries = []
    valid_entries = []
    
    for problem in all_problems:
        if (not problem.get('problem') and 
            not problem.get('audience') and 
            not problem.get('opportunity')):
            empty_entries.append(problem)
        else:
            valid_entries.append(problem)
    
    print(f"\nFound {len(empty_entries)} empty entries")
    print(f"Found {len(valid_entries)} valid entries")
    
    if empty_entries:
        print("\nEmpty entries:")
        for entry in empty_entries:
            print(f"  - {entry.get('timestamp', 'No timestamp')} - Market size: {entry.get('market_size', 'N/A')}")
        
        # Create backup
        backup_file = memory_file.with_suffix('.json.backup')
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ Created backup at {backup_file}")
        
        # Update data
        data['all_problems_discovered'] = valid_entries
        data['total_problems_discovered'] = len(valid_entries)
        
        # Save cleaned data
        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Removed {len(empty_entries)} empty entries")
        print(f"✅ Updated total_problems_discovered to {len(valid_entries)}")
    else:
        print("\n✅ No empty entries found - research memory is clean!")
    
    # Also clean up duplicate sectors
    all_sectors = data.get('all_sectors_discovered', [])
    unique_sectors = list(set(all_sectors))
    
    if len(unique_sectors) < len(all_sectors):
        print(f"\nFound {len(all_sectors) - len(unique_sectors)} duplicate sectors")
        data['all_sectors_discovered'] = sorted(unique_sectors)
        data['total_unique_sectors'] = len(unique_sectors)
        
        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Reduced sectors from {len(all_sectors)} to {len(unique_sectors)} unique entries")


if __name__ == "__main__":
    clean_research_memory()