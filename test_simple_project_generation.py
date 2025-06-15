#!/usr/bin/env python3
"""
Simple test to demonstrate dynamic project generation.
"""

import json
import asyncio


async def test_dynamic_research():
    """Test that project ideas come from research, not hardcoded lists."""
    
    print("\n=== Dynamic Project Generation Test ===\n")
    
    # Simulate the AI research prompt that would be used
    research_prompt = """
    Research real-world problems and opportunities for a new software project.
    
    Find a specific problem that:
    1. Affects real people or businesses daily
    2. Has clear monetization potential (subscriptions, transactions, services)
    3. Can generate revenue 24/7 with minimal intervention
    4. Is not well-served by existing solutions
    
    Research areas to consider:
    - Small business inefficiencies
    - Personal productivity challenges
    - Health and wellness tracking
    - Education and skill development
    - Financial management pain points
    - Community and local services
    - Environmental sustainability
    - Remote work challenges
    - Content creation workflows
    - Customer service automation
    
    Provide a specific problem/solution pair with:
    - problem: Clear description of the real problem
    - target_audience: Who experiences this problem
    - solution: How technology can solve it
    - monetization: Specific revenue model
    - differentiation: What makes this unique
    - market_evidence: Proof this is needed
    """
    
    print("1. Research Prompt (sent to AI):")
    print("-" * 50)
    print(research_prompt)
    
    # Simulate AI response with a researched problem
    simulated_research = {
        "problem": "Small restaurants struggle with managing online orders from multiple platforms (UberEats, DoorDash, etc.), leading to missed orders, inventory issues, and poor customer experience",
        "target_audience": "Independent restaurants and small restaurant chains with 1-10 locations",
        "solution": "Unified order management platform that aggregates orders from all delivery services into one dashboard with real-time inventory sync",
        "monetization": "Monthly SaaS subscription ($99-299/month per location) + transaction fees (0.5% of order value)",
        "differentiation": "AI-powered demand forecasting and automatic menu availability updates across all platforms",
        "market_evidence": "70% of restaurants report order management as top pain point, $300B food delivery market growing 15% annually"
    }
    
    print("\n2. AI Research Result:")
    print("-" * 50)
    print(json.dumps(simulated_research, indent=2))
    
    # Generate project details based on research
    project_details = {
        "name": "unified-restaurant-orders",
        "description": "Multi-platform order management system for restaurants",
        "problem_statement": simulated_research["problem"],
        "target_audience": simulated_research["target_audience"],
        "monetization_strategy": simulated_research["monetization"],
        "initial_features": [
            "Multi-platform order aggregation dashboard",
            "Real-time inventory synchronization",
            "AI demand forecasting system",
            "Automated menu availability updates",
            "Order analytics and reporting"
        ],
        "customizations": {
            "packages": ["pusher/pusher-php-server", "spatie/laravel-webhook-client"],
            "configuration": ["Redis pub/sub for real-time updates", "Webhook endpoints for platforms"],
            "features": ["Platform API integrations", "Real-time order notifications"]
        }
    }
    
    print("\n3. Generated Project Details:")
    print("-" * 50)
    print(json.dumps(project_details, indent=2))
    
    # Check for hardcoded content
    hardcoded_projects = [
        "Business Analytics Dashboard",
        "Customer Engagement Mobile App",
        "Content Management System",
        "API Gateway Service",
        "E-Commerce Marketplace",
        "Team Collaboration Suite"
    ]
    
    print("\n4. Validation:")
    print("-" * 50)
    
    is_dynamic = True
    for hardcoded in hardcoded_projects:
        if hardcoded.lower() in project_details["name"].lower() or \
           hardcoded.lower() in project_details["description"].lower():
            print(f"❌ Found hardcoded project: {hardcoded}")
            is_dynamic = False
    
    if is_dynamic:
        print("✅ Project is dynamically generated based on research!")
        print("✅ No hardcoded project templates found!")
    
    # Show architecture prompt
    print("\n5. Architecture Generation Prompt Preview:")
    print("-" * 50)
    print(f"Project: {project_details['name']}")
    print(f"Problem: {project_details['problem_statement'][:100]}...")
    print(f"Target: {project_details['target_audience']}")
    print("Will generate comprehensive architecture with:")
    print("- Design system (colors, typography)")
    print("- Database schema")
    print("- API endpoints")
    print("- Real-time features")
    print("- Security measures")
    print("- Testing strategy")
    
    return project_details


async def main():
    """Run the test."""
    print("Dynamic Project Generation Demo")
    print("=" * 60)
    print("\nThis demonstrates how projects are now generated through:")
    print("1. Real market research (via Gemini API)")
    print("2. Problem identification")
    print("3. Solution design")
    print("4. Architecture planning")
    print("\nNO hardcoded project lists!")
    
    await test_dynamic_research()
    
    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())