"""
Market Research Engine for Dynamic Project Discovery

This module provides intelligent market analysis to identify real-world opportunities
and trends for project creation, replacing hard-coded portfolio gap detection.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import json
import re

from scripts.ai_brain import IntelligentAIBrain


@dataclass
class MarketTrend:
    """Represents a market trend or opportunity."""
    category: str
    trend_name: str
    description: str
    demand_level: str  # 'high', 'medium', 'low'
    technologies: List[str]
    problem_space: str
    target_audience: str
    competitive_landscape: str
    opportunity_score: float  # 0.0 to 1.0


@dataclass
class ProjectOpportunity:
    """Represents a specific project opportunity based on market research."""
    title: str
    description: str
    problem_statement: str
    solution_approach: str
    tech_stack: List[str]
    target_market: str
    unique_value_proposition: str
    estimated_complexity: str  # 'low', 'medium', 'high'
    market_demand: float  # 0.0 to 1.0
    innovation_score: float  # 0.0 to 1.0
    monetization_model: Optional[str] = None  # How it makes money 24/7
    revenue_potential: Optional[str] = None  # Expected revenue range


class MarketResearchEngine:
    """Engine for conducting market research and identifying project opportunities."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize the market research engine.
        
        Args:
            ai_brain: AI brain for intelligent analysis
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Cache for research results
        self.trend_cache: Dict[str, MarketTrend] = {}
        self.last_research_time: Optional[datetime] = None
        self.research_interval_hours = 24  # Research every 24 hours
        
    async def discover_market_trends(self) -> List[MarketTrend]:
        """Discover current market trends and opportunities.
        
        Returns:
            List of market trends
        """
        self.logger.info("🔍 Discovering market trends...")
        
        # Check cache validity
        if self._is_cache_valid():
            self.logger.info("Using cached market trends")
            return list(self.trend_cache.values())
        
        trends = []
        
        # 1. Analyze GitHub trending topics
        github_trends = await self._analyze_github_trends()
        trends.extend(github_trends)
        
        # 2. Research emerging technologies
        tech_trends = await self._research_emerging_technologies()
        trends.extend(tech_trends)
        
        # 3. Identify real-world problem spaces
        problem_trends = await self._identify_problem_spaces()
        trends.extend(problem_trends)
        
        # 4. Analyze developer community needs
        community_trends = await self._analyze_developer_needs()
        trends.extend(community_trends)
        
        # Update cache
        self.trend_cache = {trend.trend_name: trend for trend in trends}
        self.last_research_time = datetime.now(timezone.utc)
        
        return trends
    
    async def generate_project_opportunities(
        self, 
        existing_portfolio: Dict[str, Any],
        max_opportunities: int = 5
    ) -> List[ProjectOpportunity]:
        """Generate specific project opportunities based on market research.
        
        Args:
            existing_portfolio: Current portfolio projects
            max_opportunities: Maximum number of opportunities to generate
            
        Returns:
            List of project opportunities
        """
        self.logger.info("🚀 Generating project opportunities...")
        
        # Get current market trends
        trends = await self.discover_market_trends()
        
        # Analyze existing portfolio
        portfolio_analysis = await self._analyze_portfolio_deeply(existing_portfolio)
        
        # Generate opportunities based on gaps and trends
        opportunities = await self._generate_opportunities(
            trends, 
            portfolio_analysis,
            max_opportunities
        )
        
        return opportunities
    
    async def _analyze_github_trends(self) -> List[MarketTrend]:
        """Analyze GitHub trending topics and projects."""
        try:
            prompt = """
            Analyze current GitHub trending topics and successful projects.
            Focus on:
            1. Most starred repositories in the last month
            2. Fastest growing projects
            3. Common problems being solved
            4. Technologies gaining traction
            
            Return 3-5 market trends based on this analysis.
            Each trend should include:
            - category: The market category (e.g., "Developer Tools", "AI Applications", "Web3")
            - trend_name: Specific trend name
            - description: What the trend is about
            - demand_level: high/medium/low
            - technologies: List of relevant technologies
            - problem_space: What problems are being solved
            - target_audience: Who needs this
            - competitive_landscape: Current competition level
            - opportunity_score: 0.0-1.0 rating
            
            Format as JSON array.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                return self._parse_trends(response.get('result', ''))
            
        except Exception as e:
            self.logger.error(f"Error analyzing GitHub trends: {e}")
        
        return []
    
    async def _research_emerging_technologies(self) -> List[MarketTrend]:
        """Research emerging technologies and frameworks."""
        try:
            prompt = """
            Research emerging technologies and frameworks in software development.
            Consider:
            1. New programming languages gaining adoption
            2. Innovative frameworks and libraries
            3. Emerging architectural patterns
            4. Cross-industry technology applications
            
            Identify 3-5 technology trends with real market potential.
            Focus on technologies that solve actual problems, not just hype.
            
            Format as market trends with all required fields in JSON.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                return self._parse_trends(response.get('result', ''))
                
        except Exception as e:
            self.logger.error(f"Error researching technologies: {e}")
        
        return []
    
    async def _identify_problem_spaces(self) -> List[MarketTrend]:
        """Identify real-world problem spaces needing solutions."""
        try:
            prompt = """
            Identify real-world problems that need software solutions.
            Research:
            1. Common pain points in various industries
            2. Inefficiencies in current workflows
            3. Emerging needs from remote work, AI adoption, etc.
            4. Underserved markets or user groups
            
            Focus on problems that:
            - Affect many people or businesses
            - Don't have good existing solutions
            - Can be solved with current technology
            - Have clear monetization potential
            
            Return 3-5 problem-based market trends in JSON format.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                return self._parse_trends(response.get('result', ''))
                
        except Exception as e:
            self.logger.error(f"Error identifying problem spaces: {e}")
        
        return []
    
    async def _analyze_developer_needs(self) -> List[MarketTrend]:
        """Analyze developer community needs and pain points."""
        try:
            prompt = """
            Analyze current developer community needs and pain points.
            Consider:
            1. Common Stack Overflow questions and problems
            2. Popular dev.to and Medium article topics
            3. Recurring issues in developer workflows
            4. Gaps in current developer tools
            
            Identify 3-5 opportunities for developer-focused tools or platforms.
            These should address real frustrations developers face daily.
            
            Format as market trends in JSON.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                return self._parse_trends(response.get('result', ''))
                
        except Exception as e:
            self.logger.error(f"Error analyzing developer needs: {e}")
        
        return []
    
    async def _analyze_portfolio_deeply(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep analysis of existing portfolio."""
        analysis = {
            'technologies_used': set(),
            'problem_domains': set(),
            'project_types': set(),
            'target_markets': set(),
            'architecture_patterns': set(),
            'maturity_levels': {},
            'innovation_areas': set()
        }
        
        for repo_name, repo_data in portfolio.items():
            # Extract technologies
            language = repo_data.get('language')
            if language:
                analysis['technologies_used'].add(language.lower())
            
            topics = repo_data.get('topics', [])
            analysis['technologies_used'].update(topics)
            
            # Use AI to understand the project deeply
            if repo_data.get('description'):
                try:
                    prompt = f"""
                    Analyze this project: {repo_name}
                    Description: {repo_data.get('description')}
                    Language: {language}
                    Topics: {', '.join(topics)}
                    
                    Identify:
                    1. Problem domain (e.g., "e-commerce", "productivity", "developer tools")
                    2. Project type (e.g., "SaaS platform", "CLI tool", "API service")
                    3. Target market (e.g., "developers", "small businesses", "enterprises")
                    4. Architecture pattern (e.g., "microservices", "monolith", "serverless")
                    5. Innovation level (how unique/innovative is this?)
                    
                    Return as JSON object.
                    """
                    
                    response = await self.ai_brain.execute_capability(
                        'problem_analysis',
                        {'prompt': prompt}
                    )
                    
                    if response and response.get('status') == 'success':
                        project_analysis = self._parse_json(response.get('result', ''))
                        if project_analysis:
                            analysis['problem_domains'].add(
                                project_analysis.get('problem_domain', 'unknown')
                            )
                            analysis['project_types'].add(
                                project_analysis.get('project_type', 'unknown')
                            )
                            analysis['target_markets'].add(
                                project_analysis.get('target_market', 'unknown')
                            )
                            analysis['architecture_patterns'].add(
                                project_analysis.get('architecture_pattern', 'unknown')
                            )
                            
                except Exception as e:
                    self.logger.debug(f"Error analyzing project {repo_name}: {e}")
        
        # Convert sets to lists for JSON serialization
        return {
            k: list(v) if isinstance(v, set) else v 
            for k, v in analysis.items()
        }
    
    async def _generate_opportunities(
        self,
        trends: List[MarketTrend],
        portfolio_analysis: Dict[str, Any],
        max_opportunities: int
    ) -> List[ProjectOpportunity]:
        """Generate specific project opportunities."""
        opportunities = []
        
        # Find gaps between trends and current portfolio
        existing_domains = set(portfolio_analysis.get('problem_domains', []))
        existing_types = set(portfolio_analysis.get('project_types', []))
        existing_markets = set(portfolio_analysis.get('target_markets', []))
        
        for trend in trends[:max_opportunities * 2]:  # Consider more trends than needed
            # Check if this trend represents a gap
            is_gap = (
                trend.problem_space not in str(existing_domains) or
                trend.target_audience not in str(existing_markets)
            )
            
            if is_gap and trend.opportunity_score > 0.6:
                # Generate specific project idea for this trend
                project = await self._generate_specific_project(
                    trend,
                    portfolio_analysis
                )
                if project:
                    opportunities.append(project)
            
            if len(opportunities) >= max_opportunities:
                break
        
        # If not enough gap-based opportunities, generate innovative projects
        if len(opportunities) < max_opportunities:
            innovative_projects = await self._generate_innovative_projects(
                max_opportunities - len(opportunities),
                portfolio_analysis
            )
            opportunities.extend(innovative_projects)
        
        return opportunities[:max_opportunities]
    
    async def _generate_specific_project(
        self,
        trend: MarketTrend,
        portfolio_analysis: Dict[str, Any]
    ) -> Optional[ProjectOpportunity]:
        """Generate a specific project idea based on a trend."""
        try:
            prompt = f"""
            Generate a SPECIFIC, UNIQUE project idea based on this market trend:
            
            Trend: {trend.trend_name}
            Problem Space: {trend.problem_space}
            Target Audience: {trend.target_audience}
            Technologies: {', '.join(trend.technologies)}
            
            Current portfolio has these types of projects: {portfolio_analysis.get('project_types', [])}
            
            Create a project that:
            1. Solves a SPECIFIC problem in this space
            2. Has a memorable, creative name (like "Notion", "Slack", not generic)
            3. Uses modern technology appropriately
            4. Has clear 24/7 revenue potential (SaaS, API usage, subscriptions, etc.)
            5. Can generate income while you sleep (automated, scalable business model)
            6. Is different from existing portfolio projects
            
            Provide:
            - title: Creative project name
            - description: 2-3 sentence overview
            - problem_statement: The specific problem being solved
            - solution_approach: How it solves the problem
            - tech_stack: Recommended technologies (be specific)
            - target_market: Who will pay for this
            - unique_value_proposition: What makes it special
            - estimated_complexity: low/medium/high
            - market_demand: 0.0-1.0
            - innovation_score: 0.0-1.0
            - monetization_model: How it generates revenue 24/7 (subscription tiers, API pricing, etc.)
            - revenue_potential: Expected monthly revenue range (e.g., "$10K-$50K MRR")
            
            Return as JSON object.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                project_data = self._parse_json(response.get('result', ''))
                if project_data:
                    # Filter to only expected fields
                    expected_fields = {
                        'title', 'description', 'problem_statement', 'solution_approach',
                        'tech_stack', 'target_market', 'unique_value_proposition',
                        'estimated_complexity', 'market_demand', 'innovation_score',
                        'monetization_model', 'revenue_potential'
                    }
                    filtered_data = {k: v for k, v in project_data.items() if k in expected_fields}
                    return ProjectOpportunity(**filtered_data)
                    
        except Exception as e:
            self.logger.error(f"Error generating project: {e}")
        
        return None
    
    async def _generate_innovative_projects(
        self,
        count: int,
        portfolio_analysis: Dict[str, Any]
    ) -> List[ProjectOpportunity]:
        """Generate innovative project ideas not based on specific trends."""
        projects = []
        
        try:
            prompt = f"""
            Generate {count} INNOVATIVE project ideas that:
            1. Solve problems people don't know they have yet
            2. Combine technologies in novel ways
            3. Target emerging markets or use cases
            4. Are feasible with current technology
            5. Have strong 24/7 passive income potential
            6. Can scale without proportional increase in effort
            
            Focus on:
            - SaaS with subscription models
            - API services with usage-based pricing
            - Automated platforms that run themselves
            - Digital products with recurring revenue
            
            Avoid these existing areas: {portfolio_analysis.get('problem_domains', [])}
            
            Each project must include monetization_model and revenue_potential fields.
            Return as JSON array of project objects.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                projects_data = self._parse_json_array(response.get('result', ''))
                for data in projects_data:
                    try:
                        # Filter to only expected fields
                        expected_fields = {
                            'title', 'description', 'problem_statement', 'solution_approach',
                            'tech_stack', 'target_market', 'unique_value_proposition',
                            'estimated_complexity', 'market_demand', 'innovation_score',
                            'monetization_model', 'revenue_potential'
                        }
                        filtered_data = {k: v for k, v in data.items() if k in expected_fields}
                        projects.append(ProjectOpportunity(**filtered_data))
                    except Exception as e:
                        self.logger.debug(f"Error creating project opportunity: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error generating innovative projects: {e}")
        
        return projects
    
    def _parse_trends(self, result: str) -> List[MarketTrend]:
        """Parse trends from AI response."""
        trends = []
        
        try:
            # Extract JSON array
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                trends_data = json.loads(json_match.group())
                for data in trends_data:
                    try:
                        # Ensure required fields have defaults
                        trend_data = {
                            'category': data.get('category', 'General'),
                            'trend_name': data.get('trend_name', 'Unknown Trend'),
                            'description': data.get('description', ''),
                            'demand_level': data.get('demand_level', 'medium'),
                            'technologies': data.get('technologies', []),
                            'problem_space': data.get('problem_space', ''),
                            'target_audience': data.get('target_audience', ''),
                            'competitive_landscape': data.get('competitive_landscape', ''),
                            'opportunity_score': float(data.get('opportunity_score', 0.5))
                        }
                        trends.append(MarketTrend(**trend_data))
                    except Exception as e:
                        self.logger.debug(f"Error parsing trend: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error parsing trends JSON: {e}")
        
        return trends
    
    def _parse_json(self, result: str) -> Optional[Dict[str, Any]]:
        """Parse JSON object from AI response."""
        try:
            # Extract JSON object
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON: {e}")
        
        return None
    
    def _parse_json_array(self, result: str) -> List[Dict[str, Any]]:
        """Parse JSON array from AI response."""
        try:
            # Extract JSON array
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON array: {e}")
        
        return []
    
    def _is_cache_valid(self) -> bool:
        """Check if the trend cache is still valid."""
        if not self.last_research_time:
            return False
        
        hours_since_research = (
            datetime.now(timezone.utc) - self.last_research_time
        ).total_seconds() / 3600
        
        return hours_since_research < self.research_interval_hours
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get market research statistics."""
        return {
            'trends_discovered': len(self.trend_cache),
            'last_research': self.last_research_time.isoformat() if self.last_research_time else None,
            'cache_valid': self._is_cache_valid(),
            'trend_categories': list(set(t.category for t in self.trend_cache.values())),
            'high_demand_trends': sum(1 for t in self.trend_cache.values() if t.demand_level == 'high')
        }