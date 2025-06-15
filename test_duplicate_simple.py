#!/usr/bin/env python3
"""
Simple test to check duplicate handling in logs.
"""

import subprocess
import re

def analyze_duplicate_handling():
    """Analyze the continuous_ai.log for duplicate handling patterns."""
    print("=== Analyzing Duplicate Task Handling ===\n")
    
    # Get recent log entries
    try:
        # Count duplicates
        dup_count = subprocess.check_output(
            ['grep', '-c', 'Duplicate task already exists:', '/workspaces/cwmai/continuous_ai.log'],
            text=True
        ).strip()
        
        # Count alternatives generated
        alt_count = subprocess.check_output(
            ['grep', '-c', 'Generated alternative task:', '/workspaces/cwmai/continuous_ai.log'],
            text=True
        ).strip()
        
        # Count AI failures
        ai_fail_count = subprocess.check_output(
            ['grep', '-c', 'AI alternative generation failed:', '/workspaces/cwmai/continuous_ai.log'],
            text=True
        ).strip()
        
        # Get recent alternatives
        recent_alts = subprocess.check_output(
            ['grep', 'Generated alternative task:', '/workspaces/cwmai/continuous_ai.log', '|', 'tail', '-5'],
            shell=True,
            text=True
        ).strip()
        
        print(f"Duplicate Detection Summary:")
        print(f"- Total duplicates detected: {dup_count}")
        print(f"- Alternative tasks generated: {alt_count}")
        print(f"- AI generation failures: {ai_fail_count}")
        print(f"- Success rate: {int(alt_count) / max(int(dup_count), 1) * 100:.1f}%")
        
        print("\nRecent Alternative Tasks Generated:")
        for line in recent_alts.split('\n'):
            if line:
                match = re.search(r'Generated alternative task: (.+)$', line)
                if match:
                    print(f"  ✓ {match.group(1)}")
        
        # Check for repeat offenders
        print("\nRepeat Duplicate Tasks:")
        repeat_output = subprocess.check_output(
            ['grep', 'Duplicate task already exists:', '/workspaces/cwmai/continuous_ai.log', '|', 
             'sed', 's/.*TASK-[0-9]* - //', '|', 'sort', '|', 'uniq', '-c', '|', 'sort', '-nr', '|', 'head', '-5'],
            shell=True,
            text=True
        ).strip()
        
        for line in repeat_output.split('\n'):
            if line:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    count, task = parts
                    print(f"  - {task}: {count} times")
        
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing logs: {e}")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    analyze_duplicate_handling()