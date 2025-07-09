"""
Project Factory Module

Converts solutions into Laravel React projects with standardized tech stack.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from scripts.solution_designer import SolutionDesign


# Standardized tech stack configuration
LARAVEL_REACT_STACK = {
    "backend": ["php", "laravel"],
    "frontend": ["react", "typescript"], 
    "database": ["mysql"],
    "cache": ["redis"],
    "realtime": ["laravel-echo-server"],
    "styling": ["tailwind-css"],
    "auth": ["laravel-sanctum"]
}


@dataclass
class ProjectSpecification:
    """Complete specification for a Laravel React project."""
    # Basic info
    name: str
    repository_name: str
    description: str
    tagline: str
    
    # Technical specs (always Laravel React)
    tech_stack: List[str]
    
    # Business details
    target_audience: str
    problem_statement: str
    solution_approach: str
    monetization_model: Dict[str, Any]
    
    # Implementation
    initial_features: List[str]
    database_considerations: List[str]
    api_endpoints: List[str]
    
    # Metadata
    created_at: str
    
    # Fields with defaults must come after required fields
    architecture: str = "laravel-react"
    starter_kit_url: str = "https://github.com/laravel/react-starter-kit"
    complexity: str = "medium"
    estimated_dev_time: str = "3 months"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/storage."""
        return {
            "name": self.name,
            "repository_name": self.repository_name,
            "description": self.description,
            "tagline": self.tagline,
            "tech_stack": self.tech_stack,
            "architecture": self.architecture,
            "starter_kit_url": self.starter_kit_url,
            "target_audience": self.target_audience,
            "problem_statement": self.problem_statement,
            "solution_approach": self.solution_approach,
            "monetization_model": self.monetization_model,
            "initial_features": self.initial_features,
            "database_considerations": self.database_considerations,
            "api_endpoints": self.api_endpoints,
            "created_at": self.created_at,
            "complexity": self.complexity,
            "estimated_dev_time": self.estimated_dev_time
        }


class ProjectFactory:
    """Creates standardized Laravel React projects from solutions."""
    
    def __init__(self):
        """Initialize project factory."""
        self.logger = logging.getLogger(__name__)
        
    def create_project(self, solution: SolutionDesign) -> ProjectSpecification:
        """Create a Laravel React project from a solution design.
        
        Args:
            solution: The solution design
            
        Returns:
            Complete project specification
        """
        # Generate repository name
        repo_name = self._generate_repo_name(solution.solution_name)
        
        # Build complete tech stack
        tech_stack = []
        for category, techs in LARAVEL_REACT_STACK.items():
            tech_stack.extend(techs)
            
        # Generate database considerations
        db_considerations = self._generate_db_schema(solution)
        
        # Generate API endpoints
        api_endpoints = self._generate_api_endpoints(solution)
        
        # Create project specification
        project = ProjectSpecification(
            name=solution.solution_name,
            repository_name=repo_name,
            description=f"{solution.tagline}. {solution.approach}"[:200],
            tagline=solution.tagline,
            tech_stack=tech_stack,
            target_audience=solution.problem.affected_users,
            problem_statement=solution.problem.problem_statement,
            solution_approach=solution.approach,
            monetization_model={
                "type": solution.monetization_type,
                "pricing": solution.pricing_strategy,
                "revenue_potential": solution.revenue_potential
            },
            initial_features=solution.core_features[:5],
            database_considerations=db_considerations,
            api_endpoints=api_endpoints,
            created_at=datetime.now().isoformat()
        )
        
        return project
        
    def create_projects_batch(self, solutions: List[SolutionDesign]) -> List[ProjectSpecification]:
        """Create multiple projects from solutions.
        
        Args:
            solutions: List of solution designs
            
        Returns:
            List of project specifications
        """
        projects = []
        
        for solution in solutions:
            try:
                project = self.create_project(solution)
                projects.append(project)
            except Exception as e:
                self.logger.error(f"Error creating project: {e}")
                
        return projects
        
    def _generate_repo_name(self, solution_name: str) -> str:
        """Generate a valid repository name."""
        # Convert to lowercase and replace spaces with hyphens
        repo_name = solution_name.lower()
        repo_name = repo_name.replace(' ', '-')
        
        # Remove special characters
        import re
        repo_name = re.sub(r'[^a-z0-9-]', '', repo_name)
        
        # Remove multiple hyphens
        repo_name = re.sub(r'-+', '-', repo_name)
        
        # Trim hyphens from ends
        repo_name = repo_name.strip('-')
        
        return repo_name or "project"
        
    def _generate_db_schema(self, solution: SolutionDesign) -> List[str]:
        """Generate database schema considerations."""
        # Standard tables for any Laravel React SaaS
        base_tables = [
            "users - User accounts with profiles",
            "subscriptions - Billing and plan management",
            "payments - Payment history and invoices"
        ]
        
        # Add solution-specific tables based on features
        if "marketplace" in solution.monetization_type:
            base_tables.extend([
                "listings - Main content/products",
                "transactions - Marketplace transactions"
            ])
        elif "saas" in solution.monetization_type:
            base_tables.extend([
                "workspaces - Team/organization management",
                "activities - User activity tracking"
            ])
            
        return base_tables[:5]  # Limit to 5 main tables
        
    def _generate_api_endpoints(self, solution: SolutionDesign) -> List[str]:
        """Generate initial API endpoints."""
        # Standard endpoints for Laravel React apps
        endpoints = [
            "POST /api/auth/register - User registration",
            "POST /api/auth/login - User authentication",
            "GET /api/user - Get current user",
            "PUT /api/user - Update user profile"
        ]
        
        # Add billing endpoints for monetized apps
        if solution.monetization_type:
            endpoints.extend([
                "GET /api/billing/plans - Available subscription plans",
                "POST /api/billing/subscribe - Create subscription"
            ])
            
        return endpoints[:6]  # Limit to 6 main endpoints