#!/usr/bin/env python3
"""
Recover improvement code from knowledge evolution reports
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def recover_improvements_from_report(report_file: str):
    """Extract and save improvements from a report file."""
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    # Create improvements directory
    improvement_dir = Path("scripts/improvements")
    improvement_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract improvements
    improvements = report.get('evolution_report', {}).get('best_improvements', [])
    
    if not improvements:
        print(f"No improvements found in {report_file}")
        return
    
    print(f"Found {len(improvements)} improvements in {report_file}")
    
    for imp in improvements:
        if not imp.get('code'):
            print(f"  ⚠️  No code for {imp.get('name', 'unnamed')}")
            continue
            
        # Save code
        file_name = f"{imp['name']}.py"
        file_path = improvement_dir / file_name
        
        with open(file_path, 'w') as f:
            f.write(imp['code'])
        
        print(f"  ✅ Saved {imp['name']} to {file_path}")
        
        # Save metadata
        metadata_path = improvement_dir / f"{imp['name']}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': imp.get('name'),
                'type': imp.get('type'),
                'description': imp.get('description'),
                'fitness': imp.get('fitness'),
                'created_at': imp.get('created_at'),
                'generation': imp.get('generation'),
                'file_path': str(file_path),
                'recovered_from': report_file,
                'recovered_at': datetime.now().isoformat()
            }, f, indent=2)

def main():
    if len(sys.argv) > 1:
        # Specific file provided
        report_file = sys.argv[1]
        recover_improvements_from_report(report_file)
    else:
        # Find all report files
        report_files = list(Path('.').glob('knowledge_evolution_report_*.json'))
        
        if not report_files:
            print("No knowledge evolution reports found")
            return
            
        print(f"Found {len(report_files)} report files")
        
        for report_file in report_files:
            print(f"\nProcessing {report_file}...")
            try:
                recover_improvements_from_report(str(report_file))
            except Exception as e:
                print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    main()