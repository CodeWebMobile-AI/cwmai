"""
Dynamic Research Prompt Generator

Generates unique research prompts using AI to ensure diverse problem discovery
without any hardcoded patterns or sectors.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import random

from scripts.research_memory import ResearchMemory


class DynamicResearchPromptGenerator:
    """Generates dynamic research prompts for problem discovery."""
    
    def __init__(self, ai_brain, research_memory: Optional[ResearchMemory] = None):
        """Initialize the prompt generator.
        
        Args:
            ai_brain: AI brain for generating prompts
            research_memory: Research memory for tracking past attempts
        """
        self.ai_brain = ai_brain
        self.research_memory = research_memory or ResearchMemory()
        self.logger = logging.getLogger(__name__)
        
    def generate_research_prompt(self) -> str:
        """Generate a completely unique research prompt.
        
        Returns:
            A dynamically generated research prompt
        """
        # Get analytics from research memory
        analytics = self.research_memory.get_analytics()
        success_patterns = self.research_memory.get_success_patterns()
        
        # Get recent attempts to avoid repetition
        recent_attempts = self.research_memory.get_recent_attempts(hours=24)
        recent_problems = [a.problem_statement for a in recent_attempts[-10:]]
        
        # Build context for AI
        context = {
            "total_attempts": analytics.get("total_attempts", 0),
            "success_rate": success_patterns.get("success_rate", 0),
            "recent_problems": recent_problems,
            "diversity_score": self.research_memory.get_diversity_score()
        }
        
        # Ask AI to generate a unique research direction
        meta_prompt = f"""
        You are a research prompt generator. Your job is to create unique prompts for discovering real-world problems.
        
        Context:
        - We've made {context['total_attempts']} research attempts
        - Success rate: {context['success_rate']:.1%}
        - Diversity score: {context['diversity_score']:.2f}
        
        Recent problem areas explored:
        {json.dumps(context['recent_problems'], indent=2) if context['recent_problems'] else 'None yet'}
        
        Generate a SINGLE research prompt that will help discover NEW types of problems.
        The prompt should:
        1. Explore a different angle than recent attempts
        2. Be specific enough to find real problems
        3. Be broad enough to discover multiple opportunities
        4. Focus on problems people actually have
        
        Return ONLY the research prompt text, nothing else.
        Examples of the TYPE of prompt (but generate something completely different):
        - Search for problems in emerging technologies
        - Find inefficiencies in daily workflows
        - Discover unmet needs in specific demographics
        
        BE CREATIVE and generate something unique.
        """
        
        try:
            response = self.ai_brain.generate_enhanced_response_sync(meta_prompt)
            research_direction = response.get('content', '') if isinstance(response, dict) else str(response)
            research_direction = research_direction.strip()
            self.logger.info(f"Generated research direction: {research_direction}")
            return research_direction if research_direction else "Discover problems that people are actively trying to solve"
        except Exception as e:
            self.logger.error(f"Error generating research prompt: {e}")
            # Fallback to a basic prompt
            return "Discover problems that people are actively trying to solve"
    
    def generate_search_queries(self, research_direction: str) -> List[str]:
        """Generate specific search queries based on the research direction.
        
        Args:
            research_direction: The high-level research direction
            
        Returns:
            List of specific search queries
        """
        prompt = f"""
        Based on this research direction: "{research_direction}"
        
        Generate 15-20 specific search queries that would help discover real problems.
        Include a variety of sources (Reddit, forums, news, social media, etc).
        
        Return a JSON array of search query strings.
        Make them specific and actionable.
        """
        
        try:
            ai_response = self.ai_brain.generate_enhanced_response_sync(prompt)
            response = ai_response.get('content', '') if isinstance(ai_response, dict) else str(ai_response)
            
            # Try to parse as JSON
            if response.strip().startswith('['):
                queries = json.loads(response)
            else:
                # Extract queries from text
                queries = [
                    line.strip().strip('"').strip("'").strip("-").strip() 
                    for line in response.split('\n') 
                    if line.strip() and not line.strip().startswith('#')
                ]
            
            # Ensure we have valid queries
            queries = [q for q in queries if len(q) > 10][:20]
            
            if not queries:
                # Generate some basic queries based on direction
                queries = [
                    f'"{research_direction}" problems',
                    f'"{research_direction}" challenges',
                    f'"{research_direction}" frustrations',
                    f'site:reddit.com "{research_direction}"',
                    f'site:news.ycombinator.com "{research_direction}"'
                ]
            
            return queries
            
        except Exception as e:
            self.logger.error(f"Error generating search queries: {e}")
            # Return basic queries
            return [
                f'"{research_direction}" problems',
                f'"{research_direction}" challenges'
            ]
    
    def adapt_based_on_results(self, found_problems: List[Dict[str, Any]], success: bool):
        """Adapt future prompts based on results.
        
        Args:
            found_problems: Problems discovered in this attempt
            success: Whether any problems led to project creation
        """
        if not found_problems:
            return
            
        # Extract patterns from discovered problems
        prompt = f"""
        Analyze these discovered problems and extract patterns:
        {json.dumps(found_problems[:5], indent=2)}
        
        Success: {success}
        
        What types of problems were these? What patterns do you see?
        Return a brief analysis (2-3 sentences).
        """
        
        try:
            response = self.ai_brain.generate_enhanced_response_sync(prompt)
            analysis = response.get('content', '') if isinstance(response, dict) else str(response)
            self.logger.info(f"Research pattern analysis: {analysis}")
            
            # Store this analysis for future prompt generation
            # This helps the system learn without hardcoding
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
    
    def get_exploration_suggestion(self) -> str:
        """Get a suggestion for what to explore next.
        
        Returns:
            Suggestion for next research direction
        """
        # Use research memory to suggest new direction
        suggestion = self.research_memory.suggest_exploration_direction(self.ai_brain)
        return suggestion