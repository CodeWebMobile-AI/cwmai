#!/usr/bin/env python3
"""
Add comprehensive logging to the Research Evolution Engine
"""

import re

def add_logging_to_research_engine():
    """Add detailed logging to research_evolution_engine.py"""
    
    with open('scripts/research_evolution_engine.py', 'r') as f:
        content = f.read()
    
    # Add logging setup at the beginning of the class
    content = re.sub(
        r'(class ResearchEvolutionEngine:)',
        r'\1\n    """Orchestrates all research components for self-improvement."""\n    \n    logger = logging.getLogger("ResearchEvolutionEngine")',
        content
    )
    
    # Add logging to __init__
    content = re.sub(
        r'(def __init__.*?\n.*?# Core system components)',
        r'\1\n        self.logger = logging.getLogger("ResearchEvolutionEngine")\n        self.logger.info("Initializing Research Evolution Engine...")',
        content,
        flags=re.DOTALL
    )
    
    # Add logging to execute_research_cycle
    content = re.sub(
        r'(print\(f"Starting research cycle: {cycle_id}"\))',
        r'self.logger.info(f"Starting research cycle: {cycle_id}")\n        \1',
        content
    )
    
    # Add logging for each step
    content = re.sub(
        r'(# Step 1: Analyze current system needs\n\s+)(system_needs = await self\._analyze_system_needs\(\))',
        r'\1self.logger.info("Step 1: Analyzing system needs...")\n        \2\n        self.logger.info(f"System needs identified: {len(system_needs) if isinstance(system_needs, list) else type(system_needs)}")',
        content
    )
    
    content = re.sub(
        r'(# Step 2: Select research topics\n\s+)(research_topics = await self\._select_research_topics\(system_needs\))',
        r'\1self.logger.info("Step 2: Selecting research topics...")\n        \2\n        self.logger.info(f"Research topics selected: {len(research_topics) if isinstance(research_topics, list) else type(research_topics)}")',
        content
    )
    
    content = re.sub(
        r'(# Step 3: Execute research\n\s+)(research_results = await self\._execute_research\(research_topics\))',
        r'\1self.logger.info("Step 3: Executing research...")\n        \2\n        self.logger.info(f"Research results obtained: {len(research_results) if isinstance(research_results, list) else type(research_results)}")',
        content
    )
    
    content = re.sub(
        r'(# Step 4: Extract actionable insights\n\s+)(insights = await self\._extract_insights\(research_results\))',
        r'\1self.logger.info("Step 4: Extracting insights...")\n        \2\n        self.logger.info(f"Insights extracted: {len(insights) if isinstance(insights, list) else type(insights)}")',
        content
    )
    
    content = re.sub(
        r'(# Step 5: Generate implementation tasks\n\s+)(tasks = await self\._generate_implementation_tasks\(insights\))',
        r'\1self.logger.info("Step 5: Generating implementation tasks...")\n        \2\n        self.logger.info(f"Tasks generated: {len(tasks) if isinstance(tasks, list) else type(tasks)}")',
        content
    )
    
    content = re.sub(
        r'(# Step 6: Implement high-confidence improvements\n\s+)(implementations = await self\._implement_improvements\(tasks\))',
        r'\1self.logger.info("Step 6: Implementing improvements...")\n        \2\n        self.logger.info(f"Implementations completed: {len(implementations) if isinstance(implementations, list) else type(implementations)}")',
        content
    )
    
    content = re.sub(
        r'(# Step 7: Measure performance changes\n\s+)(performance_changes = await self\._measure_performance_changes\(cycle_start_time\))',
        r'\1self.logger.info("Step 7: Measuring performance changes...")\n        \2\n        self.logger.info(f"Performance changes: {performance_changes}")',
        content
    )
    
    content = re.sub(
        r'(# Step 8: External agent research.*?\n\s+if.*?:)',
        r'\1\n            self.logger.info("Step 8: Checking external agent research...")',
        content,
        flags=re.DOTALL
    )
    
    content = re.sub(
        r'(# Step 9: Learn from outcomes\n\s+)(await self\._learn_from_cycle\(cycle_results\))',
        r'\1self.logger.info("Step 9: Learning from cycle outcomes...")\n        \2\n        self.logger.info("Learning complete")',
        content
    )
    
    content = re.sub(
        r'(# Step 10: Perform cross-research analysis.*?\n\s+if.*?:)',
        r'\1\n            self.logger.info("Step 10: Performing cross-research analysis...")',
        content,
        flags=re.DOTALL
    )
    
    # Add timeout logging for each async method
    content = re.sub(
        r'(async def _analyze_system_needs\(self\).*?\n)',
        r'\1        """Analyze current system needs for research."""\n        self.logger.debug("_analyze_system_needs starting...")\n        start_time = time.time()\n',
        content
    )
    
    content = re.sub(
        r'(async def _select_research_topics\(self.*?\).*?\n)',
        r'\1        """Select research topics based on needs."""\n        self.logger.debug("_select_research_topics starting...")\n        start_time = time.time()\n',
        content
    )
    
    content = re.sub(
        r'(async def _execute_research\(self.*?\).*?\n)',
        r'\1        """Execute research on selected topics."""\n        self.logger.debug("_execute_research starting...")\n        start_time = time.time()\n',
        content
    )
    
    # Add error logging
    content = re.sub(
        r'(except Exception as e:\n\s+)(self\.logger\.error\(f"Error in research cycle {cycle_id}: {e}"\))',
        r'\1import traceback\n        self.logger.error(f"Error in research cycle {cycle_id}: {e}")\n        self.logger.error(f"Traceback:\\n{traceback.format_exc()}")\n        \2',
        content
    )
    
    # Write the updated file
    with open('scripts/research_evolution_engine_logged.py', 'w') as f:
        f.write(content)
    
    print("✅ Created research_evolution_engine_logged.py with comprehensive logging")

if __name__ == "__main__":
    add_logging_to_research_engine()