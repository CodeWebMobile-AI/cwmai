#!/usr/bin/env python3
"""
Test the new unified project discovery flow.
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.unified_project_discovery import UnifiedProjectDiscovery
from scripts.market_discovery import MarketDiscovery
from scripts.solution_designer import SolutionDesigner
from scripts.project_factory import ProjectFactory


async def test_modular_flow():
    """Test the modular approach."""
    print("🧪 Testing Modular Project Discovery Flow")
    print("=" * 60)
    
    # Step 1: Discover problems
    print("\n📍 Step 1: Market Discovery")
    market = MarketDiscovery()
    problems = await market.discover_problems(count=3)
    
    if not problems:
        print("❌ No problems discovered")
        return
        
    print(f"✅ Found {len(problems)} problems")
    for i, problem in enumerate(problems, 1):
        print(f"   {i}. {problem.problem_statement[:60]}...")
        
    # Step 2: Design solutions
    print("\n📍 Step 2: Solution Design")
    designer = SolutionDesigner()
    solutions = await designer.design_solutions_batch(problems)
    
    print(f"✅ Designed {len(solutions)} solutions")
    for i, solution in enumerate(solutions, 1):
        print(f"   {i}. {solution.solution_name}: {solution.tagline}")
        
    # Step 3: Create projects
    print("\n📍 Step 3: Project Creation")
    factory = ProjectFactory()
    projects = factory.create_projects_batch(solutions)
    
    print(f"✅ Created {len(projects)} project specifications")
    for i, project in enumerate(projects, 1):
        print(f"   {i}. {project.repository_name}")
        print(f"      Tech: {', '.join(project.tech_stack[:5])}...")
        print(f"      Model: {project.monetization_model['type']}")
        
    # Save results
    results = [p.to_dict() for p in projects]
    with open('modular_flow_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to modular_flow_results.json")


async def test_unified_flow():
    """Test the unified approach."""
    print("\n\n🧪 Testing Unified Project Discovery")
    print("=" * 60)
    
    discovery = UnifiedProjectDiscovery()
    projects = await discovery.discover_and_create_projects(count=3)
    
    print(f"\n✅ Generated {len(projects)} complete projects")
    for i, project in enumerate(projects, 1):
        print(f"\n{i}. {project.title}")
        print(f"   Problem: {project.problem_description[:60]}...")
        print(f"   Solution: {project.solution_approach[:60]}...")
        print(f"   Tech: {', '.join(project.tech_stack[:5])}...")
        print(f"   Score: {project.opportunity_score:.2f}")


async def main():
    """Run all tests."""
    # Test modular flow
    await test_modular_flow()
    
    # Test unified flow  
    await test_unified_flow()
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    asyncio.run(main())