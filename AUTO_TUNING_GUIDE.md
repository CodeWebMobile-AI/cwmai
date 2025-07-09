# Auto-Tuning Guide for Knowledge-Driven Evolution

## Overview

The auto-tuning feature automatically adjusts evolutionary parameters based on performance metrics to optimize the improvement generation process. It monitors for stagnation and dynamically modifies parameters to maintain progress.

## Features

### 1. **Automatic Parameter Adjustment**
- **Mutation Rate**: Dynamically adjusted between 0.05 and 0.5
- **Population Size**: Adjusted between 5 and 30 individuals
- **Synthesis Frequency**: Modified between every 3 to 20 generations
- **Learning Rate**: Fine-tuned between 0.001 and 0.1 (if applicable)

### 2. **Stagnation Detection**
- Monitors fitness improvement over the last 10 generations
- Detects when progress has plateaued
- Triggers interventions to restore progress

### 3. **Performance-Based Recommendations**
- Analyzes current performance metrics
- Generates context-aware recommendations
- Applies changes automatically without manual intervention

## How It Works

### Detection Process
Every 5 generations, the system:
1. Calculates current best and average fitness
2. Compares with historical performance
3. Identifies stagnation patterns
4. Generates appropriate recommendations

### Auto-Tuning Logic
```python
# Stagnation detected when:
# - No fitness improvement in 10 generations
# - Breakthrough rate too low
# - Population diversity insufficient

# Interventions:
# - Increase mutation rate → Add diversity
# - Adjust population size → Balance exploration/exploitation
# - Modify synthesis frequency → Control breakthrough attempts
```

## Usage

### Enable Auto-Tuning (Default)
```bash
arch -arm64 python3 run_knowledge_driven_evolution.py --minutes 30
```

### Run Without Auto-Tuning
```bash
arch -arm64 python3 run_knowledge_driven_evolution.py --minutes 30 --no-auto-tuning
```

## Parameter Bounds

| Parameter | Min | Max | Step | Purpose |
|-----------|-----|-----|------|---------|
| Mutation Rate | 0.05 | 0.5 | 20% change | Controls genetic diversity |
| Population Size | 5 | 30 | +/- 2 | Balances breadth of search |
| Synthesis Frequency | 3 | 20 | +/- 2 | Manages breakthrough attempts |
| Learning Rate | 0.001 | 0.1 | 10% change | Fine-tunes adaptation speed |

## Example Output

```
🔄 Generation 15 - 450s remaining

🔧 Running auto-tuning analysis...
  ⚠️ Stagnation detected: 2 cycles without improvement

💡 AUTO-TUNING RECOMMENDATIONS:
  • Consider increasing mutation rate to add diversity
  • Consider increasing population size for broader exploration

🔧 Auto-adjusted mutation rate: 0.10 → 0.12
🔧 Auto-adjusted population size: 10 → 12
  ✅ Auto-tuning applied, resetting stagnation counter
```

## Auto-Tuning Report

The final summary includes a comprehensive auto-tuning report:

```
🔧 AUTO-TUNING SUMMARY:
  • Enabled: True
  • Current mutation rate: 0.18
  • Current population size: 14
  • Adjustments applied: 3

  Applied adjustments:
    - Consider increasing mutation rate to add diversity
    - Consider increasing population size for broader exploration
    - Consider increasing synthesis frequency for more breakthrough attempts
```

## Benefits

1. **Adaptive Performance**: System automatically adapts to changing conditions
2. **No Manual Intervention**: Parameters adjust without user input
3. **Stagnation Prevention**: Detects and resolves performance plateaus
4. **Optimized Resource Usage**: Balances exploration vs exploitation
5. **Historical Tracking**: All adjustments are logged for analysis

## Integration with Research

When research is enabled, auto-tuning also considers:
- Research insight generation rate
- Knowledge integration effectiveness
- Cross-domain synthesis success

## Troubleshooting

### No Auto-Tuning Activity
- Check if performance is already optimal (no stagnation)
- Verify auto-tuning is enabled (check logs)
- Ensure sufficient generations have passed (5+ needed)

### Too Frequent Adjustments
- System may be in difficult optimization landscape
- Consider manual parameter override if needed
- Check fitness function for instability

### Parameter Hitting Bounds
- Natural behavior when exploring extremes
- System will adapt within safe bounds
- Manual intervention possible if needed

## Future Enhancements

1. **Meta-Learning**: Learn optimal tuning strategies over time
2. **Multi-Objective Tuning**: Consider multiple performance metrics
3. **Predictive Tuning**: Anticipate stagnation before it occurs
4. **Custom Strategies**: User-defined tuning behaviors