#!/usr/bin/env python3
"""
Query Research Knowledge Base

Simple utility to explore saved market research data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

RESEARCH_DIR = Path("/Users/fortressarigo/PhpstormProjects/cwmai/research_knowledge")

def load_research_summary() -> Dict[str, Any]:
    """Load the research summary file."""
    summary_file = RESEARCH_DIR / "research_summary.json"
    if not summary_file.exists():
        print("No research data found yet.")
        return {}
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def show_summary():
    """Display overall research summary."""
    summary = load_research_summary()
    if not summary:
        return
    
    print("\n📊 Research Knowledge Summary")
    print("=" * 50)
    print(f"Total research sessions: {summary.get('total_research_sessions', 0)}")
    print(f"Total unique sectors: {summary.get('total_unique_sectors', 0)}")
    print(f"Total problems discovered: {summary.get('total_problems_discovered', 0)}")
    print(f"Last research: {summary.get('last_research', 'Never')}")
    
    print("\n🎯 Top Sectors Discovered:")
    sectors = summary.get('all_sectors_discovered', [])
    for i, sector in enumerate(sectors[:10], 1):
        print(f"  {i}. {sector}")

def search_problems(keyword: str):
    """Search for problems containing a keyword."""
    summary = load_research_summary()
    if not summary:
        return
    
    problems = summary.get('all_problems_discovered', [])
    matches = [p for p in problems if keyword.lower() in p['problem'].lower()]
    
    print(f"\n🔍 Found {len(matches)} problems matching '{keyword}':")
    print("=" * 50)
    for p in matches[:10]:  # Show first 10
        print(f"\n• {p['problem']}")
        print(f"  Audience: {p['audience']}")
        print(f"  Date: {p['timestamp']}")

def show_recent_research(count: int = 5):
    """Show the most recent research sessions."""
    files = sorted(RESEARCH_DIR.glob("research_*.json"), reverse=True)[:count]
    
    print(f"\n📅 Last {count} Research Sessions:")
    print("=" * 50)
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📄 {file.name}")
        print(f"  Source: {data.get('source', 'Unknown')}")
        print(f"  Problems found: {data.get('problems_discovered', 0)}")
        print(f"  Sectors: {', '.join(data.get('sectors_discovered', []))}")

def analyze_sectors():
    """Analyze sector distribution."""
    summary = load_research_summary()
    if not summary:
        return
    
    # Count problems per sector
    sector_counts = {}
    for problem in summary.get('all_problems_discovered', []):
        audience = problem.get('audience', 'Unknown')
        sector_counts[audience] = sector_counts.get(audience, 0) + 1
    
    print("\n📈 Sector Analysis (Problems per Audience):")
    print("=" * 50)
    
    # Sort by count
    sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
    for sector, count in sorted_sectors[:15]:
        bar = "█" * min(count, 40)
        print(f"{sector:30} {bar} ({count})")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python query_research_knowledge.py <command> [args]")
        print("\nCommands:")
        print("  summary     - Show overall research summary")
        print("  search <keyword> - Search problems by keyword")
        print("  recent [n]  - Show n most recent research sessions (default 5)")
        print("  sectors     - Analyze sector distribution")
        return
    
    command = sys.argv[1]
    
    if command == "summary":
        show_summary()
    elif command == "search" and len(sys.argv) > 2:
        search_problems(sys.argv[2])
    elif command == "recent":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        show_recent_research(count)
    elif command == "sectors":
        analyze_sectors()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()