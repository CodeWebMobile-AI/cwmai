#!/usr/bin/env python3
"""
Research System Monitor
Tracks and displays research activity in real-time
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path


def monitor_research():
    """Monitor research system activity."""
    print("🔬 Research System Monitor")
    print("=" * 50)
    
    log_file = "continuous_ai_output_new.log"
    research_dir = Path("research_knowledge")
    
    last_check = 0
    research_cycles = 0
    last_research_time = None
    
    while True:
        try:
            # Check log file for research activity
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                # Count research mentions since last check
                new_research = 0
                for i, line in enumerate(lines):
                    if i > last_check and 'research' in line.lower():
                        if 'Starting research cycle' in line:
                            research_cycles += 1
                            last_research_time = datetime.now()
                            print(f"\n🚀 Research cycle #{research_cycles} started!")
                        elif 'research_conducted' in line:
                            print(f"  📊 Research conducted")
                        elif 'insights_extracted' in line:
                            print(f"  💡 Insights extracted")
                        elif 'tasks_generated' in line:
                            print(f"  ✅ Tasks generated from research")
                        new_research += 1
                
                last_check = len(lines)
            
            # Check research knowledge directory
            knowledge_files = []
            if research_dir.exists():
                for subdir in research_dir.iterdir():
                    if subdir.is_dir():
                        files = list(subdir.glob("**/*.json"))
                        knowledge_files.extend(files)
            
            # Display status
            print(f"\r📈 Status: Cycles: {research_cycles} | Knowledge Files: {len(knowledge_files)} | Last Activity: {last_research_time or 'None'}", end='', flush=True)
            
            # Check state file for research metrics
            if os.path.exists("system_state.json"):
                with open("system_state.json", 'r') as f:
                    state = json.load(f)
                    research_metrics = state.get("research_metrics", {})
                    if research_metrics:
                        print(f"\n📊 Research Metrics: {json.dumps(research_metrics, indent=2)}")
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\n👋 Research monitor stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    monitor_research()