"""
Enhanced Continuous Orchestrator with MCP Integration

Extends the continuous orchestrator with MCP support while maintaining
backward compatibility with existing systems.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from scripts.continuous_orchestrator import ContinuousOrchestrator
from scripts.mcp_integration import MCPIntegrationHub
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.work_item_types import WorkItem


class EnhancedContinuousOrchestrator(ContinuousOrchestrator):
    """Continuous orchestrator enhanced with MCP integrations."""
    
    def __init__(self, max_workers: int = 10, enable_parallel: bool = True, 
                 enable_research: bool = True, enable_round_robin: bool = False,
                 enable_mcp: bool = True):
        """Initialize enhanced orchestrator.
        
        Args:
            max_workers: Maximum number of parallel workers
            enable_parallel: Whether to enable parallel processing
            enable_research: Whether to enable research evolution engine
            enable_round_robin: Whether to enable round-robin AI provider selection
            enable_mcp: Whether to enable MCP integrations
        """
        super().__init__(max_workers, enable_parallel, enable_research, enable_round_robin)
        
        self.enable_mcp = enable_mcp and os.getenv('MCP_ENABLED', 'true').lower() == 'true'
        self.mcp_hub: Optional[MCPIntegrationHub] = None
        self.mcp_issue_creator: Optional[MCPGitHubIssueCreator] = None
        self._mcp_initialized = False
        
        self.logger.info(f"MCP integration: {'enabled' if self.enable_mcp else 'disabled'}")
        
    async def initialize(self):
        """Initialize with MCP support."""
        if self.enable_mcp:
            await self._initialize_mcp()
            
    async def _initialize_mcp(self):
        """Initialize MCP integrations."""
        try:
            self.mcp_hub = MCPIntegrationHub()
            await self.mcp_hub.initialize()
            
            # Use MCP-enabled issue creator if GitHub MCP is available
            if self.mcp_hub.github:
                self.mcp_issue_creator = MCPGitHubIssueCreator(self.mcp_hub)
                self.logger.info("✅ MCP GitHub integration enabled")
            
            self._mcp_initialized = True
            self.logger.info("✅ MCP integrations initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP: {e}")
            self.enable_mcp = False
            
    async def create_github_issue(self, work_item: WorkItem) -> Dict:
        """Create GitHub issue with MCP fallback."""
        # Try MCP first if available
        if self.mcp_issue_creator and self._mcp_initialized:
            try:
                self.logger.info("Creating issue via MCP integration")
                return await self.mcp_issue_creator.execute_work_item(work_item)
            except Exception as e:
                self.logger.error(f"MCP issue creation failed: {e}, falling back to direct API")
        
        # Fallback to original method
        return await super().create_github_issue(work_item)
    
    async def discover_repositories_with_mcp(self) -> List[str]:
        """Discover repositories using MCP GitHub search."""
        repositories = []
        
        if self.mcp_hub and self.mcp_hub.github and self._mcp_initialized:
            try:
                # Define search queries for different categories
                search_queries = [
                    {
                        "query": "ai automation in:readme stars:>50 pushed:>2024-01-01",
                        "category": "AI Automation"
                    },
                    {
                        "query": "continuous integration tool stars:>100 language:python",
                        "category": "CI/CD Tools"
                    },
                    {
                        "query": "developer productivity stars:>25 topics:automation",
                        "category": "Developer Tools"
                    },
                    {
                        "query": "machine learning framework stars:>200",
                        "category": "ML Frameworks"
                    }
                ]
                
                for search in search_queries:
                    self.logger.info(f"🔍 Searching {search['category']} repositories...")
                    results = await self.mcp_hub.github.search_repositories(
                        search['query'], 
                        limit=10
                    )
                    
                    for repo in results:
                        repo_name = repo.get('full_name')
                        if repo_name and repo_name not in repositories:
                            repositories.append(repo_name)
                            
                            # Store discovery metadata in memory
                            if self.mcp_hub.memory:
                                await self.mcp_hub.memory.store_context(
                                    key=f"repo_discovery_{repo_name.replace('/', '_')}",
                                    value={
                                        "discovered_at": datetime.now().isoformat(),
                                        "category": search['category'],
                                        "stars": repo.get('stargazers_count', 0),
                                        "language": repo.get('language'),
                                        "description": repo.get('description'),
                                        "topics": repo.get('topics', [])
                                    }
                                )
                
                self.logger.info(f"✅ Discovered {len(repositories)} repositories via MCP")
                
            except Exception as e:
                self.logger.error(f"Error discovering repos via MCP: {e}")
        
        return repositories
    
    async def fetch_market_intelligence_with_mcp(self) -> Dict:
        """Fetch market intelligence using MCP Fetch."""
        intelligence = {}
        
        if self.mcp_hub and self.mcp_hub.fetch and self._mcp_initialized:
            try:
                # Define intelligence sources
                sources = [
                    {
                        "name": "hackernews_ai",
                        "url": "https://hn.algolia.com/api/v1/search?tags=story&query=AI%20automation&hitsPerPage=10"
                    },
                    {
                        "name": "github_trending",
                        "url": "https://api.github.com/search/repositories?q=created:>2024-01-01&sort=stars&order=desc&per_page=10"
                    },
                    {
                        "name": "producthunt_ai",
                        "url": "https://api.producthunt.com/v2/api/graphql"  # Would need proper GraphQL query
                    }
                ]
                
                for source in sources:
                    try:
                        self.logger.info(f"📊 Fetching intelligence from {source['name']}...")
                        data = await self.mcp_hub.fetch.fetch_json(source['url'])
                        
                        if data:
                            intelligence[source['name']] = data
                            
                            # Analyze and store insights
                            if self.mcp_hub.memory:
                                await self.mcp_hub.memory.store_context(
                                    key=f"market_intelligence_{source['name']}_{datetime.now().strftime('%Y%m%d')}",
                                    value={
                                        "source": source['name'],
                                        "timestamp": datetime.now().isoformat(),
                                        "data": data,
                                        "analyzed": False
                                    }
                                )
                    except Exception as e:
                        self.logger.warning(f"Failed to fetch from {source['name']}: {e}")
                
                self.logger.info(f"✅ Collected intelligence from {len(intelligence)} sources via MCP")
                
            except Exception as e:
                self.logger.error(f"Error fetching market intelligence via MCP: {e}")
        
        return intelligence
    
    async def persist_state_with_mcp(self, state: Dict):
        """Persist state using multiple MCP backends."""
        # First use parent's persistence (Redis/file)
        await super().persist_state(state)
        
        if not self._mcp_initialized:
            return
        
        # Store in MCP memory for fast access
        if self.mcp_hub and self.mcp_hub.memory:
            try:
                await self.mcp_hub.memory.store_context(
                    key="orchestrator_state_latest",
                    value=state,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "cycle_count": state.get('cycle_count', 0)
                    }
                )
            except Exception as e:
                self.logger.error(f"Failed to persist state to MCP memory: {e}")
        
        # Store in MySQL for long-term persistence
        if self.mcp_hub and self.mcp_hub.mysql:
            try:
                # Ensure table exists
                await self.mcp_hub.mysql.create_table("orchestrator_states", {
                    "id": "INT AUTO_INCREMENT PRIMARY KEY",
                    "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "cycle_count": "INT",
                    "tasks_created": "INT",
                    "workers_active": "INT",
                    "state_json": "JSON",
                    "metrics_json": "JSON"
                })
                
                # Extract metrics
                metrics = {
                    "total_value_created": state.get('total_value_created', 0),
                    "success_rate": state.get('success_rate', 0),
                    "repositories_managed": len(state.get('repositories', []))
                }
                
                # Insert state record
                await self.mcp_hub.mysql.insert_record("orchestrator_states", {
                    "cycle_count": state.get('cycle_count', 0),
                    "tasks_created": state.get('total_tasks_created', 0),
                    "workers_active": len([w for w in state.get('workers', {}).values() 
                                         if w.get('status') == 'working']),
                    "state_json": state,
                    "metrics_json": metrics
                })
                
                self.logger.info("💾 Persisted state to MySQL via MCP")
                
            except Exception as e:
                self.logger.error(f"Failed to persist state to MySQL: {e}")
    
    async def analyze_repository_files_with_mcp(self, repo_path: str) -> Dict:
        """Analyze repository files using MCP filesystem."""
        analysis = {"files": [], "languages": {}, "structure": {}}
        
        if self.mcp_hub and self.mcp_hub.filesystem and self._mcp_initialized:
            try:
                # List repository contents
                entries = await self.mcp_hub.filesystem.list_directory(repo_path, recursive=True)
                
                for entry in entries:
                    if entry.get('type') == 'file':
                        file_path = entry.get('path')
                        analysis['files'].append(file_path)
                        
                        # Analyze file extensions
                        ext = os.path.splitext(file_path)[1]
                        if ext:
                            analysis['languages'][ext] = analysis['languages'].get(ext, 0) + 1
                
                # Read key files for deeper analysis
                key_files = ['README.md', 'package.json', 'requirements.txt', 'setup.py']
                for key_file in key_files:
                    full_path = os.path.join(repo_path, key_file)
                    try:
                        content = await self.mcp_hub.filesystem.read_file(full_path)
                        if content:
                            analysis['structure'][key_file] = {
                                'exists': True,
                                'size': len(content),
                                'preview': content[:200]
                            }
                    except:
                        pass
                
                self.logger.info(f"✅ Analyzed {len(analysis['files'])} files via MCP")
                
            except Exception as e:
                self.logger.error(f"Error analyzing repository via MCP: {e}")
        
        return analysis
    
    async def run_single_cycle(self):
        """Run a single orchestration cycle with MCP enhancements."""
        cycle_start = time.time()
        
        # Discover repositories with MCP if enabled
        if self.enable_mcp and self._mcp_initialized:
            new_repos = await self.discover_repositories_with_mcp()
            if new_repos:
                self.logger.info(f"🔍 Discovered {len(new_repos)} new repositories via MCP")
                # Add to system state
                for repo in new_repos:
                    if repo not in self.system_state.get('repositories', []):
                        self.system_state.setdefault('repositories', []).append(repo)
        
        # Fetch market intelligence
        if self.enable_mcp and self._mcp_initialized:
            intelligence = await self.fetch_market_intelligence_with_mcp()
            if intelligence:
                self.system_state['market_intelligence'] = intelligence
        
        # Run parent's cycle logic
        await super().run_single_cycle()
        
        # Enhanced state persistence
        if self.enable_mcp and self._mcp_initialized:
            await self.persist_state_with_mcp(self.system_state)
        
        cycle_duration = time.time() - cycle_start
        self.logger.info(f"✅ Cycle completed in {cycle_duration:.2f}s with MCP enhancements")
    
    async def cleanup(self):
        """Clean up resources including MCP connections."""
        if self.mcp_hub and self._mcp_initialized:
            try:
                await self.mcp_hub.close()
                self.logger.info("✅ MCP connections closed")
            except Exception as e:
                self.logger.error(f"Error closing MCP connections: {e}")
        
        await super().cleanup()


async def main():
    """Run the enhanced continuous orchestrator."""
    orchestrator = EnhancedContinuousOrchestrator(
        max_workers=3,
        enable_parallel=True,
        enable_research=True,
        enable_round_robin=True,
        enable_mcp=True
    )
    
    try:
        await orchestrator.initialize()
        await orchestrator.run()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())