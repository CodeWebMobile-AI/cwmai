"""
MCP-Enabled Continuous Orchestrator
Example of how to integrate MCPs into the continuous orchestrator
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from scripts.mcp_integration import MCPIntegrationHub
from scripts.mcp_github_issue_creator import MCPGitHubIssueCreator
from scripts.continuous_orchestrator import ContinuousOrchestrator
from scripts.work_item_types import WorkItem

logger = logging.getLogger(__name__)


class MCPContinuousOrchestrator(ContinuousOrchestrator):
    """Enhanced orchestrator using MCP integrations."""
    
    def __init__(self):
        super().__init__()
        self.mcp_hub: Optional[MCPIntegrationHub] = None
        self.mcp_issue_creator: Optional[MCPGitHubIssueCreator] = None
        
    async def initialize(self):
        """Initialize with MCP support."""
        await super().initialize()
        
        # Initialize MCP hub
        self.mcp_hub = MCPIntegrationHub()
        await self.mcp_hub.initialize()
        
        # Use MCP-enabled issue creator
        self.mcp_issue_creator = MCPGitHubIssueCreator(self.mcp_hub)
        
        logger.info("✅ MCP integrations initialized")
        
    async def create_github_issue(self, work_item: WorkItem) -> Dict:
        """Create GitHub issue using MCP."""
        if self.mcp_issue_creator:
            return await self.mcp_issue_creator.execute_work_item(work_item)
        else:
            # Fallback to original method
            return await super().create_github_issue(work_item)
    
    async def discover_repositories(self) -> List[str]:
        """Discover repositories using MCP GitHub search."""
        repositories = []
        
        if self.mcp_hub and self.mcp_hub.github:
            try:
                # Search for AI/automation related repos
                search_queries = [
                    "ai automation in:readme stars:>10",
                    "continuous integration tool stars:>50",
                    "developer productivity stars:>20"
                ]
                
                for query in search_queries:
                    results = await self.mcp_hub.github.search_repositories(query, limit=5)
                    for repo in results:
                        repositories.append(repo['full_name'])
                
                logger.info(f"🔍 Discovered {len(repositories)} repositories via MCP")
                
                # Store discoveries in memory
                if self.mcp_hub.memory:
                    await self.mcp_hub.memory.store_context(
                        key=f"repo_discovery_{datetime.now().isoformat()}",
                        value={
                            "repositories": repositories,
                            "timestamp": datetime.now().isoformat(),
                            "method": "mcp_github_search"
                        }
                    )
                
            except Exception as e:
                logger.error(f"Error discovering repos via MCP: {e}")
                # Fallback to original method
                repositories = await super().discover_repositories()
        else:
            repositories = await super().discover_repositories()
        
        return repositories
    
    async def fetch_market_intelligence(self) -> Dict:
        """Fetch market intelligence using MCP."""
        intelligence = {}
        
        if self.mcp_hub and self.mcp_hub.fetch:
            try:
                # Fetch from multiple sources
                sources = [
                    ("https://api.github.com/trending", "github_trending"),
                    ("https://hn.algolia.com/api/v1/search?tags=story&query=AI", "hackernews_ai")
                ]
                
                for url, key in sources:
                    data = await self.mcp_hub.fetch.fetch_json(url)
                    if data:
                        intelligence[key] = data
                
                # Store in memory for analysis
                if self.mcp_hub.memory and intelligence:
                    await self.mcp_hub.memory.store_context(
                        key=f"market_intelligence_{datetime.now().isoformat()}",
                        value=intelligence
                    )
                
                logger.info(f"📊 Fetched market intelligence from {len(intelligence)} sources via MCP")
                
            except Exception as e:
                logger.error(f"Error fetching market intelligence via MCP: {e}")
        
        return intelligence
    
    async def persist_state(self, state: Dict):
        """Persist state using MCP MySQL."""
        # First, use parent's Redis persistence
        await super().persist_state(state)
        
        # Additionally persist to MySQL for long-term storage
        if self.mcp_hub and self.mcp_hub.mysql:
            try:
                # Ensure table exists
                await self.mcp_hub.mysql.create_table("orchestrator_state", {
                    "id": "INT AUTO_INCREMENT PRIMARY KEY",
                    "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "state_json": "JSON",
                    "tasks_created": "INT",
                    "cycle_count": "INT"
                })
                
                # Insert state record
                await self.mcp_hub.mysql.insert_record("orchestrator_state", {
                    "state_json": state,
                    "tasks_created": state.get('total_tasks_created', 0),
                    "cycle_count": state.get('cycle_count', 0)
                })
                
                logger.info("💾 Persisted state to MySQL via MCP")
                
            except Exception as e:
                logger.error(f"Error persisting to MySQL: {e}")
    
    async def analyze_repository_with_mcp(self, repo_name: str) -> Dict:
        """Analyze repository using MCP tools."""
        analysis = {}
        
        if self.mcp_hub and self.mcp_hub.github:
            try:
                # Get repository info
                repo_info = await self.mcp_hub.github.get_repository_info(repo_name)
                
                # Get recent issues
                issues = await self.mcp_hub.github.list_issues(repo_name, limit=20)
                
                analysis = {
                    "name": repo_info.get("name"),
                    "description": repo_info.get("description"),
                    "stars": repo_info.get("stargazers_count", 0),
                    "open_issues": repo_info.get("open_issues_count", 0),
                    "language": repo_info.get("language"),
                    "recent_issue_titles": [issue["title"] for issue in issues[:5]],
                    "has_ai_mentions": any("ai" in issue["title"].lower() or 
                                         "claude" in issue.get("body", "").lower() 
                                         for issue in issues)
                }
                
                # Store analysis in memory
                if self.mcp_hub.memory:
                    await self.mcp_hub.memory.store_context(
                        key=f"repo_analysis_{repo_name.replace('/', '_')}",
                        value=analysis
                    )
                
            except Exception as e:
                logger.error(f"Error analyzing repo {repo_name} via MCP: {e}")
        
        return analysis
    
    async def cleanup(self):
        """Clean up MCP connections."""
        if self.mcp_hub:
            await self.mcp_hub.close()
        
        await super().cleanup()


async def main():
    """Run MCP-enabled orchestrator."""
    orchestrator = MCPContinuousOrchestrator()
    
    try:
        await orchestrator.initialize()
        
        # Run a single cycle for demonstration
        logger.info("Running single orchestration cycle with MCP...")
        await orchestrator.run_single_cycle()
        
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())