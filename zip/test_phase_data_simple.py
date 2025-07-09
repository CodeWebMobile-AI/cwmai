#!/usr/bin/env python3
"""Simple test to verify phase data saving logic"""

import json
from pathlib import Path

# Test the hybrid saving logic
def test_hybrid_saving():
    """Test the hybrid approach for different score levels"""
    
    print("Testing hybrid phase data saving logic...\n")
    
    # Simulate opportunities with different scores
    test_opportunities = [
        {
            'id': 'opp_high_score',
            'problem_description': 'This is a high-scoring opportunity with great potential',
            'affected_audience': 'Large enterprises',
            'validation_score': 8.5,
            'other_data': 'Lots of detailed information here'
        },
        {
            'id': 'opp_medium_score',
            'problem_description': 'This is a medium-scoring opportunity that might work',
            'affected_audience': 'Small businesses',
            'validation_score': 6.2,
            'other_data': 'Some information here'
        },
        {
            'id': 'opp_low_score',
            'problem_description': 'This is a low-scoring opportunity that will be rejected',
            'affected_audience': 'Individual users',
            'validation_score': 3.5,
            'other_data': 'Not much potential'
        }
    ]
    
    # Simulate the hybrid saving logic
    phase_data = []
    
    for opp in test_opportunities:
        score = opp.get('validation_score', 0)
        
        if score >= 7.0:  # High scorers
            phase_entry = {
                'id': opp['id'],
                'score': score,
                'status': 'high_potential',
                'problem': opp['problem_description'],
                'audience': opp['affected_audience'],
                'details': {k: v for k, v in opp.items() if k not in ['id', 'problem_description', 'affected_audience']}
            }
            print(f"HIGH SCORER ({score}): Saving full details")
            print(f"  - Entry: {json.dumps(phase_entry, indent=2)}\n")
            
        elif score >= 5.0:  # Medium scorers
            phase_entry = {
                'id': opp['id'],
                'score': score,
                'status': 'medium_potential',
                'problem': opp['problem_description'][:50] + '...',
                'audience': opp['affected_audience'],
                'key_metrics': {
                    'validation_score': score
                }
            }
            print(f"MEDIUM SCORER ({score}): Saving summary only")
            print(f"  - Entry: {json.dumps(phase_entry, indent=2)}\n")
            
        else:  # Low scorers
            phase_entry = {
                'id': opp['id'],
                'score': score,
                'status': 'rejected',
                'problem_summary': opp['problem_description'][:30] + '...',
                'rejection_phase': 'phase_2_problems_validated'
            }
            print(f"LOW SCORER ({score}): Saving minimal data")
            print(f"  - Entry: {json.dumps(phase_entry, indent=2)}\n")
        
        phase_data.append(phase_entry)
    
    # Summary
    print("="*60)
    print("SUMMARY:")
    print("="*60)
    high_count = len([p for p in phase_data if p.get('status') == 'high_potential'])
    medium_count = len([p for p in phase_data if p.get('status') == 'medium_potential'])
    rejected_count = len([p for p in phase_data if p.get('status') == 'rejected'])
    
    print(f"High potential: {high_count}")
    print(f"Medium potential: {medium_count}")
    print(f"Rejected: {rejected_count}")
    print(f"\nTotal data size comparison:")
    print(f"  - Full data: {len(json.dumps(test_opportunities))} bytes")
    print(f"  - Hybrid approach: {len(json.dumps(phase_data))} bytes")
    print(f"  - Savings: {100 - (len(json.dumps(phase_data)) / len(json.dumps(test_opportunities)) * 100):.1f}%")
    
    # Check a recent research file
    print("\n" + "="*60)
    print("CHECKING RECENT RESEARCH FILES:")
    print("="*60)
    research_dir = Path("research_knowledge")
    if research_dir.exists():
        files = sorted(research_dir.glob("research_*.json"), reverse=True)
        # Skip summary files
        files = [f for f in files if not str(f).endswith('summary.json')]
        
        if files:
            latest_file = files[0]
            print(f"Latest research file: {latest_file.name}")
            
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            phases = data.get('research_phases', {})
            print(f"\nPhase data in file:")
            for phase, content in phases.items():
                if content:
                    if isinstance(content, list):
                        print(f"  ✓ {phase}: {len(content)} items")
                    else:
                        print(f"  ✓ {phase}: Has data")
                else:
                    print(f"  ✗ {phase}: Empty")
        else:
            print("No research files found")

if __name__ == "__main__":
    test_hybrid_saving()