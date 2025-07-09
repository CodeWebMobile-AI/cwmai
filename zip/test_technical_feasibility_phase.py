#!/usr/bin/env python3
"""
Test script to reproduce the technical feasibility phase issue where Gemini Pro
generates extremely long (20-50K chars) responses with repetitive content.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.gemini_deep_research import GeminiDeepResearch

async def test_technical_feasibility():
    """Test just the technical feasibility phase with a sample opportunity."""
    
    # Initialize the research module
    research = GeminiDeepResearch()
    
    # Sample opportunity data (similar to what would be passed in real scenario)
    test_opportunity = {
        "solution_name": "HealthNavigator AI",
        "tagline": "AI-powered healthcare cost prediction and navigation",
        "core_innovation": "Personalized AI predictions for healthcare costs with community support",
        "key_features": [
            "AI-powered cost predictions",
            "Insurance plan comparisons",
            "Provider recommendations",
            "Community support forums",
            "Financial assistance matching"
        ],
        "technical_approach": "AI/ML models for cost prediction, API integrations with insurance providers",
        "implementation_phases": [
            "Phase 1: Core prediction engine",
            "Phase 2: Insurance integrations",
            "Phase 3: Community features"
        ],
        "revenue_model": "Freemium with premium subscriptions",
        "pricing_strategy": "$0-19/month tiered pricing",
        "go_to_market": "Content marketing and healthcare partnerships",
        "risk_mitigation": "HIPAA compliance and data security focus",
        "success_metrics": "User engagement and cost savings achieved",
        "competitive_moat": "AI accuracy and community network effects",
        "year_one_projection": "10,000 users, $500K ARR",
        "automation_level": "semi-automated",
        "revenue_frequency": "monthly",
        "passive_income_score": 7
    }
    
    # The exact prompt used in the technical feasibility phase
    prompt = f"""
    You have access to Google Search. Conduct deep technical feasibility assessment.
    
    Proposed solution:
    {json.dumps(test_opportunity, indent=2)}
    
    Technical validation tasks:
    
    1. Required technologies and APIs:
       - What specific APIs, services, and technologies are needed?
       - Pricing models for each external service
       - Rate limits and usage restrictions
       - Alternative options if primary choices fail
    
    2. Development complexity assessment:
       - Similar successful implementations
       - Required team composition and skills
       - Realistic development timeline
       - Major technical challenges
    
    3. Infrastructure and scaling:
       - Initial infrastructure costs (1k, 10k, 100k users)
       - Scaling challenges and solutions
       - Performance optimization strategies
       - Security and compliance requirements
    
    4. Technical risks and mitigation:
       - Single points of failure
       - Vendor lock-in risks
       - Data privacy concerns
       - Technical debt accumulation
    
    Search for:
    - "[similar solution] technical architecture"
    - "[technology] implementation challenges"
    - "[API name] pricing calculator"
    - "[technology] vs [alternative] comparison"
    - "scaling [solution type] to 100k users"
    
    Return assessment:
    {{
        "technical_score": 1-10,
        "required_apis": [
            {{
                "name": "API name",
                "cost": "pricing structure",
                "limits": "rate limits",
                "alternatives": ["backup options"]
            }}
        ],
        "technology_stack": {{
            "frontend": "frameworks and tools",
            "backend": "languages and frameworks",
            "database": "database choices",
            "infrastructure": "hosting and deployment",
            "third_party": "external services"
        }},
        "required_team_size": "team composition",
        "time_to_mvp": "realistic timeline",
        "development_complexity": "low/medium/high",
        "infrastructure_costs": {{
            "1k_users": "monthly cost",
            "10k_users": "monthly cost",
            "100k_users": "monthly cost"
        }},
        "technical_risks": ["identified risks"],
        "similar_implementations": ["examples with outcomes"],
        "unique_technical_challenges": "specific challenges",
        "technical_moat": "defensible technical advantages"
    }}
    
    Only validate if technical_score >= 7 and time_to_mvp <= "3 months".
    """
    
    # The exact schema used
    technical_schema = {
        "type": "object",
        "properties": {
            "technical_score": {"type": "number"},
            "required_apis": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cost": {"type": "string"},
                        "limits": {"type": "string"},
                        "alternatives": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "cost", "limits", "alternatives"]
                }
            },
            "technology_stack": {
                "type": "object",
                "properties": {
                    "frontend": {"type": "string"},
                    "backend": {"type": "string"},
                    "database": {"type": "string"},
                    "infrastructure": {"type": "string"},
                    "third_party": {"type": "string"}
                },
                "required": ["frontend", "backend", "database", "infrastructure", "third_party"]
            },
            "required_team_size": {"type": "string"},
            "time_to_mvp": {"type": "string"},
            "development_complexity": {"type": "string"},
            "infrastructure_costs": {
                "type": "object",
                "properties": {
                    "1k_users": {"type": "string"},
                    "10k_users": {"type": "string"},
                    "100k_users": {"type": "string"}
                },
                "required": ["1k_users", "10k_users", "100k_users"]
            },
            "technical_risks": {
                "type": "array",
                "items": {"type": "string"}
            },
            "similar_implementations": {
                "type": "array",
                "items": {"type": "string"}
            },
            "unique_technical_challenges": {"type": "string"},
            "technical_moat": {"type": "string"}
        },
        "required": ["technical_score", "required_apis", "technology_stack",
                    "required_team_size", "time_to_mvp", "development_complexity",
                    "infrastructure_costs", "technical_risks", "similar_implementations",
                    "unique_technical_challenges", "technical_moat"]
    }
    
    print("=" * 80)
    print("TESTING TECHNICAL FEASIBILITY PHASE")
    print("=" * 80)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Schema complexity: {len(json.dumps(technical_schema))} chars")
    print("=" * 80)
    
    # Make the API call
    try:
        result = await research._call_gemini_structured(prompt, technical_schema, max_retries=3)
        
        if result:
            print(f"\n✅ SUCCESS - Response received")
            print(f"Response type: {type(result)}")
            print(f"Response length: {len(json.dumps(result))} chars")
            print(f"\nResponse preview:")
            print(json.dumps(result, indent=2)[:500] + "...")
            
            # Save full response for analysis
            with open("technical_feasibility_response.json", "w") as f:
                json.dump({
                    "prompt_length": len(prompt),
                    "schema": technical_schema,
                    "response": result,
                    "response_length": len(json.dumps(result))
                }, f, indent=2)
            print(f"\n📝 Full response saved to technical_feasibility_response.json")
        else:
            print("\n❌ FAILED - No response received")
            
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we have the Gemini API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Run: export $(cat ../.env.local | grep -v '^#' | xargs)")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_technical_feasibility())