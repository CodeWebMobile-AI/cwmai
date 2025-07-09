"""
Solution Designer Module

Focused solely on designing solutions for validated problems.
No technical implementation or repository details.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from scripts.ai_brain import IntelligentAIBrain
from scripts.market_discovery import MarketProblem


@dataclass  
class SolutionDesign:
    """A designed solution for a market problem."""
    # Problem reference
    problem: MarketProblem
    
    # Solution details
    solution_name: str
    tagline: str
    approach: str
    unique_value: str
    
    # Business model
    monetization_type: str  # 'saas', 'marketplace', 'freemium'
    pricing_strategy: List[Dict[str, Any]]
    revenue_potential: str
    
    # Features
    core_features: List[str]
    differentiators: List[str]


class SolutionDesigner:
    """Designs solutions for market problems."""
    
    def __init__(self, ai_brain: IntelligentAIBrain = None):
        """Initialize solution designer."""
        self.ai_brain = ai_brain or IntelligentAIBrain()
        self.logger = logging.getLogger(__name__)
        
    async def design_solution(self, problem: MarketProblem) -> Optional[SolutionDesign]:
        """Design a solution for a validated problem.
        
        Args:
            problem: The market problem to solve
            
        Returns:
            Designed solution or None if design fails
        """
        print(f"🎨 Designing solution for: {problem.problem_statement[:50]}...")
        
        prompt = f"""
        Design an innovative web-based solution for this validated problem.
        
        Problem: {problem.problem_statement}
        Affected Users: {problem.affected_users}
        Current Solutions Fail Because: {problem.why_they_fail}
        Market Size: {problem.market_size}
        
        Create a solution with:
        1. A memorable product name (1-2 words, brandable)
        2. A clear tagline (max 10 words)
        3. The core approach to solving the problem
        4. What makes it 10x better than existing solutions
        5. A sustainable business model
        6. 5 core features that directly address the pain points
        7. 3 key differentiators
        
        Focus on practical implementation and clear monetization.
        
        Return as JSON with all fields.
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt)
            if response and response.get('content'):
                design_data = self._parse_response(response['content'])
                
                if design_data:
                    return SolutionDesign(
                        problem=problem,
                        solution_name=design_data.get('solution_name', 'Solution'),
                        tagline=design_data.get('tagline', ''),
                        approach=design_data.get('approach', ''),
                        unique_value=design_data.get('unique_value', ''),
                        monetization_type=design_data.get('monetization_type', 'saas'),
                        pricing_strategy=design_data.get('pricing_strategy', []),
                        revenue_potential=design_data.get('revenue_potential', '$10K-50K ARR'),
                        core_features=design_data.get('core_features', []),
                        differentiators=design_data.get('differentiators', [])
                    )
        except Exception as e:
            self.logger.error(f"Error designing solution: {e}")
            
        return None
        
    async def design_solutions_batch(self, problems: List[MarketProblem]) -> List[SolutionDesign]:
        """Design solutions for multiple problems.
        
        Args:
            problems: List of market problems
            
        Returns:
            List of designed solutions
        """
        solutions = []
        
        for problem in problems:
            solution = await self.design_solution(problem)
            if solution:
                solutions.append(solution)
                
        return solutions
        
    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response."""
        try:
            # Remove markdown formatting
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Try to parse
            try:
                return json.loads(content)
            except:
                # Extract JSON object
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            
        return None