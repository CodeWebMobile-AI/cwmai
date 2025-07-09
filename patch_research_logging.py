#!/usr/bin/env python3
"""
Simple patch to add logging to research engine
"""

# Copy the original file
import shutil
shutil.copy('scripts/research_evolution_engine.py', 'scripts/research_evolution_engine_logged.py')

# Read the file
with open('scripts/research_evolution_engine_logged.py', 'r') as f:
    lines = f.readlines()

# Find specific lines and add logging
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    
    # Add logger attribute after class definition
    if 'class ResearchEvolutionEngine:' in line:
        new_lines.append('    """Orchestrates all research components for self-improvement."""\n')
        new_lines.append('    \n')
        new_lines.append('    logger = logging.getLogger("ResearchEvolutionEngine")\n')
    
    # Add logger initialization in __init__
    if 'self.state_manager = state_manager' in line and i > 40:
        new_lines.append('        self.logger = logging.getLogger("ResearchEvolutionEngine")\n')
        new_lines.append('        self.logger.info("Initializing Research Evolution Engine...")\n')
    
    # Replace print statements with logger
    if 'print(f"Starting research cycle:' in line:
        new_lines[-1] = line.replace('print(', 'self.logger.info(')
    
    if 'print(f"Completed research cycle:' in line:
        new_lines[-1] = line.replace('print(', 'self.logger.info(')
    
    # Add logging before each step
    if '# Step 1: Analyze current system needs' in line:
        new_lines.append('            self.logger.info("Step 1: Analyzing system needs...")\n')
    
    if '# Step 2: Select research topics' in line:
        new_lines.append('            self.logger.info("Step 2: Selecting research topics...")\n')
    
    if '# Step 3: Execute research' in line:
        new_lines.append('            self.logger.info("Step 3: Executing research...")\n')
    
    if '# Step 4: Extract actionable insights' in line:
        new_lines.append('            self.logger.info("Step 4: Extracting insights...")\n')
    
    if '# Step 5: Generate implementation tasks' in line:
        new_lines.append('            self.logger.info("Step 5: Generating implementation tasks...")\n')
    
    if '# Step 6: Implement high-confidence improvements' in line:
        new_lines.append('            self.logger.info("Step 6: Implementing improvements...")\n')
    
    if '# Step 7: Measure performance changes' in line:
        new_lines.append('            self.logger.info("Step 7: Measuring performance changes...")\n')
    
    if '# Step 9: Learn from outcomes' in line:
        new_lines.append('            self.logger.info("Step 9: Learning from cycle outcomes...")\n')
    
    # Add logging after some operations
    if 'system_needs = await self._analyze_system_needs()' in line:
        new_lines.append('            self.logger.info(f"System needs identified: {len(system_needs) if isinstance(system_needs, list) else type(system_needs)}")\n')
    
    if 'research_topics = await self._select_research_topics(system_needs)' in line:
        new_lines.append('            self.logger.info(f"Research topics selected: {len(research_topics) if isinstance(research_topics, list) else type(research_topics)}")\n')
    
    if 'research_results = await self._execute_research(research_topics)' in line:
        new_lines.append('            self.logger.info(f"Research results obtained: {len(research_results) if isinstance(research_results, list) else type(research_results)}")\n')
    
    if 'insights = await self._extract_insights(research_results)' in line:
        new_lines.append('            self.logger.info(f"Insights extracted: {len(insights) if isinstance(insights, list) else type(insights)}")\n')
    
    if 'tasks = await self._generate_implementation_tasks(insights)' in line:
        new_lines.append('            self.logger.info(f"Tasks generated: {len(tasks) if isinstance(tasks, list) else type(tasks)}")\n')
    
    if 'implementations = await self._implement_improvements(tasks)' in line:
        new_lines.append('            self.logger.info(f"Implementations completed: {len(implementations) if isinstance(implementations, list) else type(implementations)}")\n')
    
    # Add traceback to error handling
    if 'self.logger.error(f"Error in research cycle {cycle_id}: {e}")' in line:
        new_lines.append('            import traceback\n')
        new_lines.append('            self.logger.error(f"Traceback:\\n{traceback.format_exc()}")\n')

# Write the modified file
with open('scripts/research_evolution_engine_logged.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Patched research_evolution_engine_logged.py with logging")