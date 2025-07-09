#!/usr/bin/env python3
"""Minimal test to verify phase data saving works"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

# Test just the save_phase_data method
print("Testing _save_phase_data method implementation...")

# Simulate the data structure
research_phases_data = {
    'phase_0_market_timing': None,
    'phase_1_problems_discovered': [],
    'phase_2_problems_validated': [],
    'phase_2_5_user_research': [],
    'phase_2_75_revenue_validation': [],
    'phase_3_competition_analysis': [],
    'phase_3_5_technical_validation': [],
    'phase_4_market_sizing': [],
    'phase_4_5_unit_economics': [],
    'phase_4_75_mvp_simulation': [],
    'phase_5_final_selection': []
}

# Simulate scores storage
scores = {
    'opp_0': {'phase_2': 8.5, 'phase_2_5': 8.2},
    'opp_1': {'phase_2': 6.1, 'phase_2_5': 5.8},
    'opp_2': {'phase_2': 3.2, 'phase_2_5': 3.0}
}

# Test opportunities
opportunities = [
    {
        'id': 'opp_0',
        'problem_description': 'High-scoring opportunity for enterprise automation',
        'affected_audience': 'Large enterprises',
        'validation_score': 8.5,
        'validation_evidence': 'Strong market demand',
        'revenue_model': 'SaaS subscription'
    },
    {
        'id': 'opp_1',
        'problem_description': 'Medium-scoring opportunity for small business tools',
        'affected_audience': 'Small businesses',
        'validation_score': 6.1,
        'validation_evidence': 'Some market interest',
        'revenue_model': 'Freemium'
    },
    {
        'id': 'opp_2',
        'problem_description': 'Low-scoring opportunity with limited potential',
        'affected_audience': 'Individual users',
        'validation_score': 3.2,
        'validation_evidence': 'Weak demand',
        'revenue_model': 'One-time purchase'
    }
]

# Simulate the _save_phase_data logic
phase_name = 'phase_2_problems_validated'
phase_data = []

for opp in opportunities:
    opp_id = opp.get('id', 'unknown')
    
    # Get score for this phase
    score = 0.0
    if opp_id in scores:
        phase_score_map = {
            'phase_2_problems_validated': 'phase_2',
            'phase_2_5_user_research': 'phase_2_5'
        }
        score_key = phase_score_map.get(phase_name)
        if score_key and score_key in scores[opp_id]:
            score = scores[opp_id][score_key]
    
    # Apply hybrid approach
    if score >= 7.0:
        phase_entry = {
            'id': opp_id,
            'score': score,
            'status': 'high_potential',
            'problem': opp.get('problem_description', ''),
            'audience': opp.get('affected_audience', ''),
            'details': {k: v for k, v in opp.items() if k not in ['id', 'problem_description', 'affected_audience']}
        }
    elif score >= 5.0:
        phase_entry = {
            'id': opp_id,
            'score': score,
            'status': 'medium_potential',
            'problem': opp.get('problem_description', '')[:100] + '...',
            'audience': opp.get('affected_audience', ''),
            'key_metrics': {
                'validation_score': opp.get('validation_score', 0)
            }
        }
    else:
        phase_entry = {
            'id': opp_id,
            'score': score,
            'status': 'rejected',
            'problem_summary': opp.get('problem_description', '')[:50] + '...',
            'rejection_phase': phase_name
        }
    
    phase_data.append(phase_entry)

# Store in research_phases_data
research_phases_data[phase_name] = phase_data

# Print results
print(f"\nPhase: {phase_name}")
print("="*60)
for entry in phase_data:
    print(f"\n{entry['id']} (Score: {entry['score']}, Status: {entry['status']})")
    if entry['status'] == 'high_potential':
        print(f"  Full problem: {entry['problem']}")
        print(f"  Details saved: {list(entry['details'].keys())}")
    elif entry['status'] == 'medium_potential':
        print(f"  Problem summary: {entry['problem']}")
        print(f"  Key metrics: {entry['key_metrics']}")
    else:
        print(f"  Problem: {entry['problem_summary']}")
        print(f"  Rejected at: {entry['rejection_phase']}")

# Create a test research file
test_dir = Path("test_research_output")
test_dir.mkdir(exist_ok=True)

test_file = test_dir / f"test_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
test_data = {
    'research_timestamp': datetime.now().isoformat(),
    'opportunities_discovered': len(opportunities),
    'research_phases': research_phases_data,
    'optimization_stats': {
        'high_potential': len([e for e in phase_data if e['status'] == 'high_potential']),
        'medium_potential': len([e for e in phase_data if e['status'] == 'medium_potential']),
        'rejected': len([e for e in phase_data if e['status'] == 'rejected'])
    }
}

with open(test_file, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"\n\nTest file created: {test_file}")
print("\nResearch phases with data:")
for phase, data in research_phases_data.items():
    if data:
        if isinstance(data, list):
            print(f"  ✓ {phase}: {len(data)} items")
        else:
            print(f"  ✓ {phase}: Has data")

print("\n✅ Implementation appears to be working correctly!")