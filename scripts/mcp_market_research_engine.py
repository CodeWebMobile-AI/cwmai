"""
MCP-Enhanced Market Research Engine

Uses Model Context Protocol for external data fetching and analysis,
providing more reliable and comprehensive market intelligence.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import json
import re

from scripts.market_research_engine import MarketResearchEngine, MarketTrend, ProjectOpportunity
from scripts.ai_brain import IntelligentAIBrain
from scripts.mcp_integration import MCPIntegrationHub


class MCPMarketResearchEngine(MarketResearchEngine):
    """Enhanced market research engine using MCP integrations."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, mcp_hub: Optional[MCPIntegrationHub] = None):
        """Initialize MCP-enhanced market research engine.
        
        Args:
            ai_brain: AI brain for intelligent analysis
            mcp_hub: Optional pre-initialized MCP integration hub
        """
        super().__init__(ai_brain)
        self.mcp_hub = mcp_hub
        self._mcp_initialized = False
        
    async def _ensure_mcp_initialized(self):
        """Ensure MCP is initialized."""
        if not self._mcp_initialized and not self.mcp_hub:
            self.mcp_hub = MCPIntegrationHub()
            await self.mcp_hub.initialize(servers=['fetch', 'github', 'memory'])
            self._mcp_initialized = True
    
    async def _analyze_github_trends_mcp(self) -> List[MarketTrend]:
        """Analyze GitHub trends using MCP."""
        trends = []
        
        if not self.mcp_hub or not self.mcp_hub.github:
            return await super()._analyze_github_trends()
        
        try:
            self.logger.info("📊 Analyzing GitHub trends via MCP...")
            
            # Search for trending repositories in different categories
            search_queries = [
                ("AI AND machine learning", "AI/ML"),
                ("developer tools AND productivity", "DevTools"),
                ("automation AND workflow", "Automation"),
                ("web3 AND blockchain", "Web3"),
                ("data engineering", "DataEng"),
                ("security AND cybersecurity", "Security")
            ]
            
            for query, category in search_queries:
                try:
                    # Search for trending repos
                    repos = await self.mcp_hub.github.search_repositories(
                        f"{query} stars:>100 created:>2024-01-01",
                        limit=5
                    )
                    
                    if repos:
                        # Analyze trends from repositories
                        tech_stack = set()
                        problems = []
                        
                        for repo in repos:
                            # Extract technologies from language
                            if repo.get('language'):
                                tech_stack.add(repo['language'])
                            
                            # Extract problems from description
                            desc = repo.get('description', '')
                            if desc:
                                problems.append(desc[:100])
                        
                        trend = MarketTrend(
                            category=category,
                            trend_name=f"{category} Innovation",
                            description=f"Growing interest in {category.lower()} solutions",
                            demand_level="high" if len(repos) >= 3 else "medium",
                            technologies=list(tech_stack)[:5],
                            problem_space=problems[0] if problems else f"{category} challenges",
                            target_audience="Developers and tech companies",
                            competitive_landscape=f"{len(repos)} active projects",
                            opportunity_score=min(0.9, len(repos) * 0.2)
                        )
                        trends.append(trend)
                        
                except Exception as e:
                    self.logger.warning(f"Error searching {category}: {e}")
            
            # Store trends in memory for analysis
            if self.mcp_hub.memory and trends:
                await self.mcp_hub.memory.store_context(
                    key=f"github_trends_{datetime.now().strftime('%Y%m%d')}",
                    value={
                        "trends": [self._trend_to_dict(t) for t in trends],
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            self.logger.info(f"✅ Found {len(trends)} GitHub trends via MCP")
            
        except Exception as e:
            self.logger.error(f"Error analyzing GitHub trends via MCP: {e}")
            return await super()._analyze_github_trends()
        
        return trends
    
    async def _fetch_tech_news_mcp(self) -> List[Dict]:
        """Fetch technology news using MCP Fetch."""
        news_items = []
        
        if not self.mcp_hub or not self.mcp_hub.fetch:
            return []
        
        try:
            # Fetch from multiple news sources
            sources = [
                {
                    "name": "HackerNews",
                    "url": "https://hn.algolia.com/api/v1/search?tags=story&query=AI%20startup&hitsPerPage=10"
                },
                {
                    "name": "ProductHunt",
                    "url": "https://api.producthunt.com/v1/posts?days_ago=1"  # Would need auth
                },
                {
                    "name": "TechCrunch",
                    "url": "https://techcrunch.com/wp-json/wp/v2/posts?per_page=10&categories=449557098"
                }
            ]
            
            for source in sources:
                try:
                    self.logger.info(f"📰 Fetching news from {source['name']}...")
                    
                    data = await self.mcp_hub.fetch.fetch_json(source['url'])
                    
                    if data:
                        # Process based on source format
                        if source['name'] == 'HackerNews' and 'hits' in data:
                            for hit in data['hits'][:5]:
                                news_items.append({
                                    'source': 'HackerNews',
                                    'title': hit.get('title', ''),
                                    'url': hit.get('url', ''),
                                    'points': hit.get('points', 0),
                                    'comments': hit.get('num_comments', 0)
                                })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fetch from {source['name']}: {e}")
            
            # Store news in memory
            if self.mcp_hub.memory and news_items:
                await self.mcp_hub.memory.store_context(
                    key=f"tech_news_{datetime.now().strftime('%Y%m%d_%H')}",
                    value=news_items
                )
            
        except Exception as e:
            self.logger.error(f"Error fetching tech news via MCP: {e}")
        
        return news_items
    
    async def _research_emerging_technologies_mcp(self) -> List[MarketTrend]:
        """Research emerging technologies using MCP."""
        trends = []
        
        await self._ensure_mcp_initialized()
        
        # Fetch tech news first
        news_items = await self._fetch_tech_news_mcp()
        
        # Analyze news for trends
        if news_items:
            tech_mentions = {}
            
            for item in news_items:
                title = item.get('title', '').lower()
                # Extract technology mentions
                tech_keywords = ['ai', 'ml', 'blockchain', 'quantum', 'ar', 'vr', 
                               'iot', '5g', 'edge computing', 'serverless']
                
                for tech in tech_keywords:
                    if tech in title:
                        tech_mentions[tech] = tech_mentions.get(tech, 0) + 1
            
            # Create trends from mentions
            for tech, count in sorted(tech_mentions.items(), key=lambda x: x[1], reverse=True)[:3]:
                trend = MarketTrend(
                    category="Emerging Tech",
                    trend_name=f"{tech.upper()} Applications",
                    description=f"Growing interest in {tech} solutions",
                    demand_level="high" if count > 3 else "medium",
                    technologies=[tech],
                    problem_space=f"Practical {tech} implementation",
                    target_audience="Early adopters and enterprises",
                    competitive_landscape="Rapidly evolving",
                    opportunity_score=min(0.9, count * 0.3)
                )
                trends.append(trend)
        
        # Fallback to AI-based analysis if no news
        if not trends:
            return await super()._research_emerging_technologies()
        
        return trends
    
    async def discover_market_trends(self) -> List[MarketTrend]:
        """Discover market trends using MCP-enhanced methods."""
        await self._ensure_mcp_initialized()
        
        # Check cache first
        if self._is_cache_valid():
            self.logger.info("Using cached market trends")
            return list(self.trend_cache.values())
        
        trends = []
        
        # Use MCP-enhanced methods
        if self.mcp_hub and self._mcp_initialized:
            # 1. GitHub trends via MCP
            github_trends = await self._analyze_github_trends_mcp()
            trends.extend(github_trends)
            
            # 2. Emerging tech via MCP
            tech_trends = await self._research_emerging_technologies_mcp()
            trends.extend(tech_trends)
            
            # 3. Retrieve previous insights from memory
            if self.mcp_hub.memory:
                try:
                    # Search for recent market insights
                    recent_insights = await self.mcp_hub.memory.search_context(
                        "market_insight",
                        limit=5
                    )
                    
                    for insight in recent_insights:
                        # Process stored insights
                        self.logger.info(f"Found stored insight: {insight.get('key')}")
                except:
                    pass
        
        # Fallback to parent methods for remaining analysis
        problem_trends = await self._identify_problem_spaces()
        trends.extend(problem_trends)
        
        community_trends = await self._analyze_developer_needs()
        trends.extend(community_trends)
        
        # Update cache
        self.trend_cache = {trend.trend_name: trend for trend in trends}
        self.last_research_time = datetime.now(timezone.utc)
        
        self.logger.info(f"✅ Discovered {len(trends)} market trends")
        
        return trends
    
    async def analyze_competitor_landscape_mcp(self, space: str) -> Dict[str, Any]:
        """Analyze competitor landscape using MCP."""
        if not self.mcp_hub or not self.mcp_hub.github:
            return {}
        
        try:
            # Search for competitors
            competitors = await self.mcp_hub.github.search_repositories(
                f"{space} stars:>50",
                limit=10
            )
            
            landscape = {
                "total_competitors": len(competitors),
                "top_competitors": [],
                "market_saturation": "low" if len(competitors) < 5 else "medium" if len(competitors) < 10 else "high",
                "opportunity_gaps": []
            }
            
            for comp in competitors[:5]:
                landscape["top_competitors"].append({
                    "name": comp.get("name"),
                    "stars": comp.get("stargazers_count", 0),
                    "description": comp.get("description", "")[:100]
                })
            
            # Store analysis
            if self.mcp_hub.memory:
                await self.mcp_hub.memory.store_context(
                    key=f"competitor_analysis_{space.replace(' ', '_')}",
                    value=landscape
                )
            
            return landscape
            
        except Exception as e:
            self.logger.error(f"Error analyzing competitors: {e}")
            return {}
    
    def _trend_to_dict(self, trend: MarketTrend) -> Dict:
        """Convert trend to dictionary for storage."""
        return {
            "category": trend.category,
            "trend_name": trend.trend_name,
            "description": trend.description,
            "demand_level": trend.demand_level,
            "technologies": trend.technologies,
            "problem_space": trend.problem_space,
            "target_audience": trend.target_audience,
            "competitive_landscape": trend.competitive_landscape,
            "opportunity_score": trend.opportunity_score
        }
    
    async def generate_project_opportunities(self, trends: List[MarketTrend], 
                                           limit: int = 5) -> List[ProjectOpportunity]:
        """Generate project opportunities with MCP-enhanced analysis."""
        # First use parent's generation
        opportunities = await super().generate_project_opportunities(trends, limit)
        
        # Enhance with MCP data if available
        if self.mcp_hub and self._mcp_initialized:
            for opp in opportunities:
                # Analyze competitor landscape
                landscape = await self.analyze_competitor_landscape_mcp(opp.problem_statement)
                
                if landscape:
                    # Adjust opportunity based on competition
                    saturation = landscape.get("market_saturation", "medium")
                    if saturation == "high":
                        opp.market_demand *= 0.7  # Reduce demand if saturated
                    elif saturation == "low":
                        opp.market_demand *= 1.2  # Increase if underserved
                    
                    # Add competitive insights to description
                    competitor_count = landscape.get("total_competitors", 0)
                    opp.description += f"\n\nMarket Analysis: {competitor_count} existing solutions found."
        
        return opportunities
    
    async def close(self):
        """Close MCP connections."""
        if self.mcp_hub and self._mcp_initialized:
            await self.mcp_hub.close()