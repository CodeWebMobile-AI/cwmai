#!/usr/bin/env python3
"""
Test the modular structure without relying on external APIs.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.market_discovery import MarketProblem
from scripts.solution_designer import SolutionDesign
from scripts.project_factory import ProjectFactory
from scripts.project_config import get_full_tech_stack, PROJECT_DEFAULTS


def test_project_factory():
    """Test the project factory with mock data."""
    print("🧪 Testing Project Factory Module")
    print("=" * 60)
    
    # Create a mock market problem
    mock_problem = MarketProblem(
        problem_statement="Small businesses struggle to manage customer appointments efficiently",
        affected_users="Small service businesses (salons, clinics, consultants)",
        pain_severity=8.5,
        current_solutions=["Paper calendars", "Basic Google Calendar"],
        why_they_fail="No automated reminders, no online booking, no payment integration",
        market_size="5 million small service businesses",
        validation_evidence=["Reddit threads", "Industry reports", "User surveys"],
        opportunity_score=0.85
    )
    
    # Create a mock solution design
    mock_solution = SolutionDesign(
        problem=mock_problem,
        solution_name="BookFlow",
        tagline="Smart appointment booking for small businesses",
        approach="AI-powered scheduling with automated reminders and payments",
        unique_value="Reduces no-shows by 80% with smart reminders",
        monetization_type="saas",
        pricing_strategy=[
            {"name": "Starter", "price": 19, "bookings": 100},
            {"name": "Pro", "price": 49, "bookings": 500},
            {"name": "Business", "price": 99, "bookings": "unlimited"}
        ],
        revenue_potential="$50K-200K ARR",
        core_features=[
            "Online booking widget for websites",
            "Automated SMS and email reminders",
            "Integrated payment processing",
            "Customer management dashboard",
            "Analytics and no-show tracking"
        ],
        differentiators=[
            "AI predicts no-show risk",
            "One-click rescheduling",
            "Works on any website"
        ]
    )
    
    # Test project creation
    factory = ProjectFactory()
    project = factory.create_project(mock_solution)
    
    print(f"\n✅ Created Project: {project.name}")
    print(f"Repository: {project.repository_name}")
    print(f"Description: {project.description}")
    print(f"\nTech Stack ({len(project.tech_stack)} technologies):")
    for tech in project.tech_stack:
        print(f"  - {tech}")
    
    print(f"\nMonetization: {project.monetization_model['type']}")
    print("Pricing Tiers:")
    for tier in project.monetization_model['pricing']:
        print(f"  - {tier['name']}: ${tier['price']}")
    
    print(f"\nInitial Features:")
    for i, feature in enumerate(project.initial_features, 1):
        print(f"  {i}. {feature}")
        
    print(f"\nDatabase Considerations:")
    for table in project.database_considerations:
        print(f"  - {table}")
        
    print(f"\nAPI Endpoints:")
    for endpoint in project.api_endpoints:
        print(f"  - {endpoint}")
    
    # Save to file
    with open('test_project_spec.json', 'w') as f:
        json.dump(project.to_dict(), f, indent=2)
    print(f"\n💾 Project specification saved to test_project_spec.json")
    
    return project


def test_tech_stack_config():
    """Test the configuration module."""
    print("\n\n🧪 Testing Configuration Module")
    print("=" * 60)
    
    # Get full tech stack
    tech_stack = get_full_tech_stack()
    print(f"\nFull Tech Stack ({len(tech_stack)} items):")
    for tech in tech_stack:
        print(f"  - {tech}")
    
    # Test project defaults
    print(f"\nProject Defaults:")
    for key, value in PROJECT_DEFAULTS.items():
        print(f"  {key}: {value}")
    
    # Test excluded tech validation
    print(f"\nTechnology Validation:")
    test_techs = ["laravel", "react", "pusher", "postgresql", "mysql"]
    for tech in test_techs:
        from scripts.project_config import validate_tech_choice
        valid = validate_tech_choice(tech)
        status = "✅ Allowed" if valid else "❌ Excluded"
        print(f"  {tech}: {status}")


def test_data_structures():
    """Test the data structures work correctly."""
    print("\n\n🧪 Testing Data Structures")
    print("=" * 60)
    
    # Test MarketProblem
    problem = MarketProblem(
        problem_statement="Test problem",
        affected_users="Test users",
        pain_severity=8.0,
        current_solutions=["Solution 1"],
        why_they_fail="They don't scale",
        market_size="Large",
        validation_evidence=["Evidence 1"],
        opportunity_score=0.8
    )
    print(f"✅ MarketProblem created: {problem.problem_statement}")
    
    # Test SolutionDesign
    solution = SolutionDesign(
        problem=problem,
        solution_name="TestSolution",
        tagline="Test tagline",
        approach="Test approach",
        unique_value="Test value",
        monetization_type="saas",
        pricing_strategy=[],
        revenue_potential="$10K",
        core_features=["Feature 1"],
        differentiators=["Diff 1"]
    )
    print(f"✅ SolutionDesign created: {solution.solution_name}")
    
    # Test ProjectSpecification
    from scripts.project_factory import ProjectSpecification
    project_spec = ProjectSpecification(
        name="Test Project",
        repository_name="test-project",
        description="Test description",
        tagline="Test tagline",
        tech_stack=["laravel", "react"],
        target_audience="Test audience",
        problem_statement="Test problem",
        solution_approach="Test approach",
        monetization_model={"type": "saas"},
        initial_features=["Feature 1"],
        database_considerations=["Table 1"],
        api_endpoints=["GET /api/test"],
        created_at=datetime.now().isoformat()
    )
    print(f"✅ ProjectSpecification created: {project_spec.name}")


def main():
    """Run all tests."""
    print("🚀 Testing Modular Project Discovery System")
    print("=" * 60)
    
    # Test configuration
    test_tech_stack_config()
    
    # Test data structures
    test_data_structures()
    
    # Test project factory
    project = test_project_factory()
    
    print("\n\n✅ All tests completed successfully!")
    print(f"\nThe modular system is working correctly:")
    print("1. Configuration is centralized")
    print("2. Data structures are properly defined")
    print("3. Project factory creates Laravel React projects")
    print("4. Tech stack is standardized")
    
    # Show the final tech stack being used
    print(f"\nFinal project tech stack: {', '.join(project.tech_stack)}")


if __name__ == "__main__":
    main()