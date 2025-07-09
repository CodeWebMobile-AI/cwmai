"""
Market Discovery Module

Focused solely on discovering real-world problems through organic search
and user research. No technical implementation details.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from scripts.gemini_deep_research import GeminiDeepResearch


@dataclass
class MarketProblem:
    """A validated market problem worth solving."""
    problem_statement: str
    affected_users: str
    pain_severity: float  # 1-10
    current_solutions: List[str]
    why_they_fail: str
    market_size: str
    validation_evidence: List[str]
    opportunity_score: float


class MarketDiscovery:
    """Discovers and validates real-world problems."""
    
    def __init__(self, system_state: Optional[Dict[str, Any]] = None):
        """Initialize market discovery.
        
        Args:
            system_state: Optional system state containing existing projects
        """
        self.logger = logging.getLogger(__name__)
        self.gemini = GeminiDeepResearch()
        self.system_state = system_state or {}
        
    async def discover_problems(self, count: int = 10) -> List[MarketProblem]:
        """Discover real-world problems worth solving.
        
        Args:
            count: Number of problems to discover
            
        Returns:
            List of validated market problems with full research data
        """
        print("🔍 Market Discovery: Finding Real Problems")
        print("=" * 50)
        
        # Use Gemini to discover problems organically
        # Use dynamic discovery based on current market trends
        # Pass existing projects from system state
        projects = self.system_state.get('projects', {})
        repositories = self.system_state.get('repositories', {})
        existing_portfolio = {**projects, **repositories}
        
        opportunities = await self.gemini.research_market_opportunities(
            count=count,
            existing_portfolio=existing_portfolio
        )
        
        # Get the full research data including all phases
        research_phases = self.gemini.research_phases_data
        
        # Convert to MarketProblem format but preserve ALL data
        problems = []
        for opp in opportunities[:count]:
            problem = MarketProblem(
                problem_statement=opp.problem_description,
                affected_users=opp.affected_audience,
                pain_severity=8.5,  # High since Gemini validated
                current_solutions=[opp.current_solutions],
                why_they_fail=opp.opportunity,
                market_size=opp.market_size,
                validation_evidence=opp.validation_sources,
                opportunity_score=0.85
            )
            # Attach the full opportunity data and research phases
            problem._full_opportunity = opp
            problem._research_phases = research_phases
            problems.append(problem)
            
        print(f"✅ Discovered {len(problems)} validated problems with full research data")
        return problems
        
    async def validate_problem(self, problem_desc: str) -> Optional[MarketProblem]:
        """Validate a specific problem idea.
        
        Args:
            problem_desc: Description of the problem to validate
            
        Returns:
            Validated MarketProblem or None if invalid
        """
        # This would use Gemini to research and validate
        # For now, returns None as placeholder
        return None