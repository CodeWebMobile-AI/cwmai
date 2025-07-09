#!/usr/bin/env python3
"""
Test the batch research logic without making API calls.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def calculate_opportunities_needed(available_workers: int, existing_projects: int) -> dict:
    """Calculate how many opportunities to research based on workers and portfolio."""
    
    # Smart calculation logic from intelligent_work_finder.py
    if existing_projects < 5:
        # Early stage: aggressive project creation
        base_opportunities = min(available_workers, 10)
    else:
        # Established portfolio: moderate creation
        base_opportunities = min(available_workers // 2, 5)
    
    # Add buffer for quality filtering (20% extra)
    opportunities_with_buffer = int(base_opportunities * 1.2)
    final_count = max(opportunities_with_buffer, 3)  # At least 3
    
    return {
        "available_workers": available_workers,
        "existing_projects": existing_projects,
        "base_opportunities": base_opportunities,
        "with_buffer": opportunities_with_buffer,
        "final_count": final_count,
        "strategy": "aggressive" if existing_projects < 5 else "moderate"
    }


def simulate_worker_distribution(workers: list, opportunities: list) -> dict:
    """Simulate how opportunities are distributed to workers."""
    assignments = {}
    unassigned_workers = []
    buffer_opportunities = []
    
    # Assign opportunities to workers
    for i, worker in enumerate(workers):
        if i < len(opportunities):
            assignments[worker] = opportunities[i]
        else:
            unassigned_workers.append(worker)
    
    # Identify buffer opportunities
    if len(opportunities) > len(workers):
        buffer_opportunities = opportunities[len(workers):]
    
    return {
        "assignments": assignments,
        "unassigned_workers": unassigned_workers,
        "buffer_opportunities": buffer_opportunities
    }


def main():
    """Run the batch logic demonstration."""
    print("🧪 Batch Research Logic Demonstration")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Startup Phase",
            "workers": 10,
            "projects": 2,
            "description": "Many workers, few projects - aggressive growth"
        },
        {
            "name": "Growth Phase", 
            "workers": 6,
            "projects": 4,
            "description": "Moderate workers, some projects - balanced growth"
        },
        {
            "name": "Established Phase",
            "workers": 8,
            "projects": 15,
            "description": "Good worker count, many projects - selective growth"
        },
        {
            "name": "Solo Developer",
            "workers": 1,
            "projects": 0,
            "description": "Single worker starting fresh"
        },
        {
            "name": "Large Team",
            "workers": 20,
            "projects": 10,
            "description": "Large team with established portfolio"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n\n📊 Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 50)
        
        # Calculate opportunities
        calc = calculate_opportunities_needed(scenario['workers'], scenario['projects'])
        
        print(f"Available Workers: {calc['available_workers']}")
        print(f"Existing Projects: {calc['existing_projects']}")
        print(f"Strategy: {calc['strategy'].upper()}")
        print(f"\nCalculation:")
        print(f"  Base opportunities: {calc['base_opportunities']}")
        print(f"  With 20% buffer: {calc['with_buffer']}")
        print(f"  Final count: {calc['final_count']}")
        
        # Simulate distribution
        workers = [f"worker-{i}" for i in range(scenario['workers'])]
        opportunities = [f"opportunity-{i}" for i in range(calc['final_count'])]
        
        distribution = simulate_worker_distribution(workers, opportunities)
        
        print(f"\nDistribution:")
        print(f"  Workers assigned: {len(distribution['assignments'])}")
        print(f"  Workers unassigned: {len(distribution['unassigned_workers'])}")
        print(f"  Buffer opportunities: {len(distribution['buffer_opportunities'])}")
        
        # Show some assignments
        if distribution['assignments']:
            print(f"\nSample assignments:")
            for worker, opp in list(distribution['assignments'].items())[:3]:
                print(f"  {worker} → {opp}")
            if len(distribution['assignments']) > 3:
                print(f"  ... and {len(distribution['assignments']) - 3} more")
        
        if distribution['buffer_opportunities']:
            print(f"\nBuffer (for quality filtering):")
            for opp in distribution['buffer_opportunities'][:2]:
                print(f"  {opp}")
    
    # Show the benefits
    print("\n\n✅ Benefits of Batch Research Approach:")
    print("1. ONE research session serves multiple workers")
    print("2. Each worker gets a unique, validated opportunity")
    print("3. Buffer ensures quality (can discard weak opportunities)")
    print("4. Scales intelligently with team size")
    print("5. Respects portfolio maturity (no over-creation)")
    
    # Show the flow
    print("\n📋 System Flow:")
    print("1. Orchestrator detects idle workers")
    print("2. Calculates opportunities needed (with buffer)")
    print("3. Gemini researches N opportunities in one batch")
    print("4. Opportunities distributed to workers")
    print("5. Buffer opportunities used if any fail validation")


if __name__ == "__main__":
    main()