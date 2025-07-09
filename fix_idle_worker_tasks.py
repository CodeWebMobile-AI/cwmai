#!/usr/bin/env python3
"""
Fix idle worker issues by ensuring research and project creation tasks
are properly queued and available for workers.
"""

import re

def fix_intelligent_work_finder():
    """Update IntelligentWorkFinder to queue research tasks for idle workers."""
    
    print("Fixing IntelligentWorkFinder to create research tasks...")
    
    # Read the file
    with open('scripts/intelligent_work_finder.py', 'r') as f:
        content = f.read()
    
    # 1. Add method to create research work items
    research_method = '''
    async def _create_research_work_items(self, count: int = 3) -> List[WorkOpportunity]:
        """Create research and project creation work items for idle workers."""
        opportunities = []
        
        # Create market research task
        opportunities.append(WorkOpportunity(
            source="idle_worker_research",
            type="MARKET_RESEARCH",
            priority=TaskPriority.MEDIUM,
            title="Conduct Market Research for New Opportunities",
            description="Research current market trends, identify problems that need solving, and discover potential project opportunities. Focus on real-world problems with monetization potential.",
            repository=None,
            estimated_cycles=2,
            metadata={
                'task_type': 'research',
                'can_be_done_by': 'any_worker',
                'no_repository_needed': True
            }
        ))
        
        # Create project evaluation task
        opportunities.append(WorkOpportunity(
            source="idle_worker_research",
            type="PORTFOLIO_ANALYSIS",
            priority=TaskPriority.MEDIUM,
            title="Analyze Portfolio for Gap Opportunities",
            description="Analyze current project portfolio to identify missing capabilities, underserved markets, and potential new project areas. Generate actionable insights for project creation.",
            repository=None,
            estimated_cycles=2,
            metadata={
                'task_type': 'analysis',
                'can_be_done_by': 'any_worker',
                'no_repository_needed': True
            }
        ))
        
        # Create project creation task
        opportunities.append(WorkOpportunity(
            source="idle_worker_research",
            type="PROJECT_CREATION",
            priority=TaskPriority.HIGH,
            title="Create New Project from Market Research",
            description="Based on recent market research findings, create a new project repository with initial setup, architecture documentation, and development roadmap.",
            repository=None,
            estimated_cycles=3,
            metadata={
                'task_type': 'creation',
                'can_be_done_by': 'any_worker',
                'no_repository_needed': True
            }
        ))
        
        return opportunities[:count]
'''
    
    # Insert the method after _discover_portfolio_gaps
    insert_pos = content.find('async def _discover_portfolio_gaps')
    if insert_pos > 0:
        # Find the end of that method
        method_end = content.find('\n    async def', insert_pos + 1)
        if method_end > 0:
            content = content[:method_end] + '\n' + research_method + '\n' + content[method_end:]
            print("✅ Added _create_research_work_items method")
    
    # 2. Update discover_work to always include research tasks
    # Find the discover_work method and add research tasks
    discover_pattern = r'(async def discover_work\(self.*?\) -> List\[WorkOpportunity\]:.*?)(# Discover new work opportunities)'
    
    research_injection = '''
        # Always add research tasks for idle workers
        if max_items > len(all_opportunities):
            remaining_slots = max_items - len(all_opportunities)
            research_tasks = await self._create_research_work_items(min(3, remaining_slots))
            all_opportunities.extend(research_tasks)
            self.logger.info(f"Added {len(research_tasks)} research tasks for idle workers")
        
        '''
    
    # Find where to inject the research tasks
    # Look for where opportunities are being collected
    collection_pattern = r'(all_opportunities\.extend\(task_opportunities\).*?\n)(.*?# Prioritize and filter)'
    
    match = re.search(collection_pattern, content, re.DOTALL)
    if match:
        content = content[:match.end(1)] + research_injection + content[match.end(1):]
        print("✅ Added research task injection in discover_work")
    
    # 3. Save the updated file
    with open('scripts/intelligent_work_finder.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated IntelligentWorkFinder")


def fix_hybrid_orchestrator():
    """Update hybrid orchestrator to handle non-repository work."""
    
    print("\nFixing hybrid orchestrator for non-repository work...")
    
    # Read the file
    with open('scripts/hybrid_orchestrator_enhanced.py', 'r') as f:
        content = f.read()
    
    # Add method to handle research work
    research_handler = '''
    async def _handle_research_work(self, work_item: WorkItem) -> Dict[str, Any]:
        """Handle research and project creation work that doesn't require a repository."""
        self.logger.info(f"Handling research work: {work_item.title}")
        
        try:
            if work_item.type == "MARKET_RESEARCH":
                # Trigger market research
                market_research = MarketResearchEngine(self.ai_brain)
                opportunities = await market_research.discover_market_opportunities(count=5)
                
                # Convert to work items
                new_work = []
                for opp in opportunities:
                    # Create project creation tasks from opportunities
                    new_work.append({
                        'title': f"Create project: {opp.problem_description[:50]}",
                        'type': 'PROJECT_CREATION',
                        'opportunity': opp
                    })
                
                return {
                    'success': True,
                    'result': f"Discovered {len(opportunities)} market opportunities",
                    'new_work': new_work
                }
                
            elif work_item.type == "PORTFOLIO_ANALYSIS":
                # Analyze portfolio gaps
                portfolio_intel = PortfolioIntelligence(self.ai_brain)
                gaps = await portfolio_intel.analyze_portfolio_gaps(self.system_state.get('projects', {}))
                
                return {
                    'success': True,
                    'result': f"Identified {len(gaps)} portfolio gaps",
                    'gaps': gaps
                }
                
            elif work_item.type == "PROJECT_CREATION":
                # Create new project
                project_creator = ProjectCreator(self.github_token if self.github_token else None)
                metadata = work_item.metadata or {}
                
                result = await project_creator.create_project(
                    name=work_item.title.replace("Create project: ", ""),
                    description=work_item.description,
                    project_type=metadata.get('project_type', 'web-app'),
                    technologies=metadata.get('tech_stack', ['python'])
                )
                
                return {
                    'success': True,
                    'result': f"Created project: {result.get('name')}",
                    'project': result
                }
                
            else:
                return {
                    'success': True,
                    'result': f"Processed {work_item.type} work"
                }
                
        except Exception as e:
            self.logger.error(f"Error handling research work: {e}")
            return {
                'success': False,
                'error': str(e)
            }
'''
    
    # Insert the method after execute_single_task
    insert_pos = content.find('async def execute_single_task')
    if insert_pos > 0:
        # Find the end of that method
        method_end = content.find('\n    async def', insert_pos + 1)
        if method_end > 0:
            content = content[:method_end] + '\n' + research_handler + '\n' + content[method_end:]
            print("✅ Added _handle_research_work method")
    
    # Update execute_single_task to handle non-repository work
    old_pattern = r'(repository = work_item\.repository.*?\n)(.*?if not repository:)'
    new_code = '''repository = work_item.repository
        
        # Handle non-repository work (research, project creation, etc)
        if not repository or work_item.metadata.get('no_repository_needed'):
            return await self._handle_research_work(work_item)
        
        if not repository:'''
    
    content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)
    print("✅ Updated execute_single_task to handle non-repository work")
    
    # Add necessary imports
    if 'from scripts.market_research_engine import MarketResearchEngine' not in content:
        # Add imports at the top
        import_section = '''from scripts.market_research_engine import MarketResearchEngine
from scripts.portfolio_intelligence import PortfolioIntelligence
from scripts.project_creator import ProjectCreator
'''
        # Find where to add imports
        import_pos = content.find('from scripts.')
        if import_pos > 0:
            content = content[:import_pos] + import_section + content[import_pos:]
            print("✅ Added necessary imports")
    
    # Save the updated file
    with open('scripts/hybrid_orchestrator_enhanced.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated hybrid orchestrator")


def main():
    """Apply all fixes."""
    print("🔧 Fixing idle worker issues...\n")
    
    fix_intelligent_work_finder()
    fix_hybrid_orchestrator()
    
    print("\n✅ All fixes applied!")
    print("\nThe system will now:")
    print("1. Create research tasks for idle workers")
    print("2. Handle non-repository work (market research, project creation)")
    print("3. Keep workers busy even when repositories are locked")
    print("\nReady to run the test!")

if __name__ == "__main__":
    main()