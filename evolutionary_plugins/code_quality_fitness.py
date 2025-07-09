"""
Example Fitness Metric Plugin

Calculates fitness based on code quality metrics.
"""

from scripts.evolutionary_plugins import FitnessMetricPlugin
from typing import Dict, Any
import ast


class CodeQualityFitness(FitnessMetricPlugin):
    """Calculates fitness based on code quality."""
    
    @property
    def name(self) -> str:
        return "code_quality"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    async def calculate_fitness(self, improvement: Dict[str, Any], 
                               context: Dict[str, Any]) -> float:
        """Calculate fitness based on code quality metrics."""
        code = improvement.get("code", "")
        
        # Simple quality metrics
        score = 0.5  # Base score
        
        try:
            # Parse code
            tree = ast.parse(code)
            
            # Has docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if ast.get_docstring(node):
                        score += 0.1
                        
            # Has type hints
            if ":" in code and "->" in code:
                score += 0.1
                
            # Has error handling
            if "try:" in code:
                score += 0.1
                
            # Reasonable length
            lines = code.split("\n")
            if 10 < len(lines) < 100:
                score += 0.1
                
        except:
            score = 0.3  # Syntax error
            
        return min(1.0, score)
        
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute plugin."""
        improvement = context.get("improvement", {})
        return await self.calculate_fitness(improvement, context)
