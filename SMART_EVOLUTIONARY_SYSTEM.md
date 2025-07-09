# Smart Evolutionary System Documentation

## Overview

The Smart Evolutionary System is an enhanced version of `run_evolutionary_aggressive.py` that eliminates hard-coded values in favor of adaptive, intelligent parameter management and dynamic discovery.

## Key Improvements

### 1. **Dynamic Parameter Adaptation**
Instead of fixed values like `population_size=75`, the system now:
- Monitors performance metrics in real-time
- Adjusts parameters based on fitness stagnation, diversity, and success rates
- Uses multi-armed bandit algorithms to select optimal strategies
- Learns from past performance to improve future runs

### 2. **Intelligent Type Discovery**
Replaces the hard-coded list of improvement types with:
- Automatic discovery by analyzing codebase patterns
- Learning from successful improvements
- Gap analysis to identify missing capabilities
- Creative generation of novel improvement types
- Clustering to categorize and organize discoveries

### 3. **Meta-Learning Memory**
The system now remembers across runs:
- Successful parameter configurations
- Improvement patterns that work well
- Context-specific strategies
- Common failure modes to avoid

### 4. **Plugin Architecture**
Extensible system that allows:
- Custom improvement generators
- New fitness metrics
- Alternative selection strategies
- Domain-specific validators

### 5. **Context-Aware Configuration**
Instead of hard-coded ensemble preferences:
- Performance-based model selection
- Adaptive temperature ranges
- Dynamic role assignment based on success rates

## Usage

### Basic Usage (Adaptive Mode)
```bash
# Uses intelligent adaptation - recommended!
python run_evolutionary_aggressive.py --minutes 10
```

### Learning from History
```bash
# Initialize parameters from past successful runs
python run_evolutionary_aggressive.py --learn-from-history
```

### Using Profiles
```bash
# Still supports profiles, but they're now starting points
python run_evolutionary_aggressive.py --profile aggressive
```

### With Custom Plugins
```bash
# Load additional plugins for domain-specific improvements
python run_evolutionary_aggressive.py --plugins ./my_plugins ./team_plugins
```

## Configuration

The system uses `evolutionary_config.json` which includes:

```json
{
  "parameters": {
    "population_size": {
      "min": 5,
      "max": 200,
      "default": 20,
      "adaptive": true
    }
  },
  "improvement_types": {
    "core": ["optimization", "error_fix"],
    "discovered": ["cache_optimizer", "ml_enhancer"]
  },
  "fitness_objectives": {
    "performance": {
      "weight": 0.25,
      "adaptive": true
    }
  }
}
```

## Architecture

### Core Components

1. **EvolutionaryConfig** - Manages adaptive configuration
2. **AdaptiveParameterEngine** - Dynamically adjusts parameters
3. **ImprovementDiscoveryEngine** - Discovers new improvement types
4. **EvolutionaryMemory** - Long-term learning system
5. **EvolutionaryPluginManager** - Handles plugins

### Data Flow

```
Start → Load Config → Check Memory → Initialize Parameters
  ↓
Evolution Loop:
  → Generate Improvements (with plugins/discovery)
  → Evaluate Fitness (adaptive weights)
  → Adapt Parameters (based on performance)
  → Remember Outcomes (for future learning)
  ↓
Save Memory → Update Config → Generate Report
```

## Plugin Development

Create custom plugins by extending base classes:

```python
from scripts.evolutionary_plugins import ImprovementGeneratorPlugin

class MyCustomGenerator(ImprovementGeneratorPlugin):
    @property
    def name(self):
        return "my_generator"
        
    async def generate_improvement(self, context):
        # Your custom generation logic
        return improvement_dict
```

## Monitoring

The system provides rich feedback:
- Real-time parameter adaptations
- Discovery notifications
- Performance metrics
- Learning summaries

## Benefits Over Hard-Coded System

1. **Self-Improving**: Gets better with each run
2. **Adaptive**: Adjusts to your specific codebase
3. **Discoverable**: Finds new opportunities automatically
4. **Extensible**: Easy to add custom behaviors
5. **Intelligent**: Makes decisions based on data, not guesses

## Migration from Old System

The new system is backward compatible:
- Old command-line arguments still work
- Profiles are supported as starting points
- Can run without history (cold start)

## Future Enhancements

- Neural network-based parameter prediction
- Distributed evolution across multiple machines
- Real-time visualization dashboard
- Integration with CI/CD pipelines