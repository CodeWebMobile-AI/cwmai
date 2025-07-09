#!/usr/bin/env python3
"""
Clean Research Summary Utility

One-time utility to clean the existing research_summary.json file by:
- Removing empty problem entries
- Deduplicating similar problems
- Normalizing audience segments
- Standardizing market sizes
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.research_summary_manager import ResearchSummaryManager
from scripts.problem_similarity_checker import ProblemSimilarityChecker


def main():
    """Clean the research summary file."""
    print("🧹 Research Summary Cleanup Utility")
    print("=" * 50)
    
    # Initialize manager
    summary_file = Path("research_knowledge/research_summary.json")
    
    if not summary_file.exists():
        print("❌ No research_summary.json found!")
        return
    
    # Backup original file
    backup_file = summary_file.with_suffix('.json.backup')
    print(f"📋 Creating backup: {backup_file}")
    
    with open(summary_file, 'r') as f:
        original_data = json.load(f)
    
    with open(backup_file, 'w') as f:
        json.dump(original_data, f, indent=2)
    
    # Show original stats
    print(f"\n📊 Original Statistics:")
    print(f"  - Total problems: {len(original_data.get('all_problems_discovered', []))}")
    print(f"  - Total sectors: {len(original_data.get('all_sectors_discovered', []))}")
    print(f"  - Research sessions: {original_data.get('total_research_sessions', 0)}")
    
    # Initialize manager and clean
    print(f"\n🔧 Cleaning data...")
    manager = ResearchSummaryManager()
    
    # Clean existing data
    stats = manager.clean_existing_data()
    
    print(f"\n✅ Cleanup Results:")
    print(f"  - Problems removed (empty/invalid): {stats['problems_removed']}")
    print(f"  - Problems deduplicated: {stats['problems_deduplicated']}")
    print(f"  - Sectors deduplicated: {stats['sectors_deduplicated']}")
    
    # Show new stats
    new_stats = manager.get_summary_stats()
    print(f"\n📊 New Statistics:")
    print(f"  - Total problems: {new_stats['total_problems']}")
    print(f"  - Unique sectors: {new_stats['unique_sectors']}")
    print(f"  - Research sessions: {new_stats['total_sessions']}")
    
    # Generate similarity report
    print(f"\n🔍 Generating Similarity Report...")
    similarity_checker = ProblemSimilarityChecker()
    
    # Get cleaned problems
    with open(summary_file, 'r') as f:
        cleaned_data = json.load(f)
    
    problems = cleaned_data.get('all_problems_discovered', [])
    if problems:
        report = similarity_checker.get_similarity_report(problems)
        
        print(f"\n📈 Similarity Analysis:")
        print(f"  - Total problems: {report['total_problems']}")
        print(f"  - Unique groups: {report['unique_groups']}")
        print(f"  - Duplication rate: {report['duplication_rate']:.1f}%")
        print(f"  - Largest similar group: {report['largest_group_size']} problems")
        
        print(f"\n🏷️ Top Problem Themes:")
        for word, count in report['common_themes'][:10]:
            print(f"  - {word}: {count} occurrences")
    
    # Show example problems
    print(f"\n📝 Sample Cleaned Problems:")
    for i, problem in enumerate(problems[:3]):
        print(f"\n{i+1}. {problem.get('problem', 'Unknown')[:100]}...")
        print(f"   Audience: {problem.get('audience', 'Unknown')}")
        print(f"   Market: {problem.get('market_size', 'Unknown')}")
    
    print(f"\n✨ Cleanup complete! Original backed up to: {backup_file}")
    
    # Ask if user wants to see duplicates
    if problems and report['duplicate_count'] > 0:
        response = input("\nWould you like to see similar problem groups? (y/n): ")
        if response.lower() == 'y':
            groups = similarity_checker.group_similar_problems(problems)
            
            print(f"\n🔗 Similar Problem Groups:")
            for i, group in enumerate(groups[:5]):  # Show first 5 groups
                if len(group) > 1:  # Only show groups with duplicates
                    print(f"\n  Group {i+1} ({len(group)} problems):")
                    for idx in group[:3]:  # Show first 3 in each group
                        print(f"    - {problems[idx]['problem'][:80]}...")


if __name__ == "__main__":
    main()