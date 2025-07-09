# Missing Data Persistence in Knowledge-Driven Evolution

## Critical Data Not Being Saved

### 1. Generated Improvements Code
**Issue**: Improvements have code but it's never written to disk
**Fix**: Add to `apply_best_improvement()`:
```python
# Save improvement code to file
improvement_dir = Path("scripts/improvements")
improvement_dir.mkdir(exist_ok=True)
file_path = improvement_dir / f"{improvement['name']}.py"
with open(file_path, 'w') as f:
    f.write(improvement['code'])
```

### 2. Research Knowledge Store
**Issue**: ResearchKnowledgeStore has no persistence
**Fix**: Add to ResearchEvolutionEngine:
```python
def save_knowledge_store(self):
    """Persist research knowledge to disk."""
    Path("research_knowledge").mkdir(exist_ok=True)
    with open("research_knowledge/store.json", 'w') as f:
        json.dump({
            'research_results': self.knowledge_store.research_results,
            'insights': self.knowledge_store.insights,
            'topics': self.knowledge_store.topics
        }, f, indent=2)
```

### 3. Evolutionary Memory State
**Issue**: EvolutionaryMemory not persisted between runs
**Fix**: Add to KnowledgeDrivenEvolutionRunner:
```python
def save_memory_state(self):
    """Save evolutionary memory."""
    self.memory.save_to_disk("evolutionary_memory.json")
    self.synthesis_engine.save_state("synthesis_state.json")
```

### 4. Population Snapshots
**Issue**: Population lost between runs
**Fix**: Save after each generation:
```python
with open(f"population_gen_{generation}.json", 'w') as f:
    json.dump(self.improvement_population, f, indent=2)
```

### 5. Pattern Learning State
**Issue**: Learned patterns not persisted
**Fix**: Already exists but not called:
```python
# In run_knowledge_driven_evolution.py, add after each generation:
self.evolutionary_engine.knowledge_store.save_state()
```

## Implementation Priority

1. **HIGH**: Save generated improvement code files
2. **HIGH**: Persist ResearchKnowledgeStore
3. **MEDIUM**: Save population snapshots
4. **MEDIUM**: Persist pattern learning state
5. **LOW**: Save intermediate evolution states

## Quick Fix Script

Create `save_all_state.py`:
```python
def save_all_state(runner):
    """Save all critical state from evolution runner."""
    # Save improvements
    for imp in runner.improvements_created:
        save_improvement_code(imp)
    
    # Save research state
    if runner.research_engine:
        runner.research_engine.knowledge_store.save_state()
    
    # Save evolutionary state
    runner.evolutionary_engine.save_population()
    runner.knowledge_store.save_state()
    
    # Save synthesis state
    runner.synthesis_engine.save_knowledge_graph()
```

## Data Recovery

For the current test run, the data exists in:
- `knowledge_evolution_report_20250622_184225.json` - Contains improvement details
- Memory of the Python process (if still running)

To extract the code from the report:
```bash
cat knowledge_evolution_report_*.json | jq -r '.evolution_report.best_improvements[].code' > recovered_code.py
```