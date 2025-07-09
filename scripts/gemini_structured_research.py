"""
Gemini Structured Research Engine with Proper JSON Output

This module implements Google's recommended approach for structured output
using response_mime_type and response_schema for guaranteed valid JSON.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import os

from scripts.ai_brain import IntelligentAIBrain


# Pydantic models for structured output
class SectorInsight(BaseModel):
    """Insight for a single market sector"""
    sector_name: str
    biggest_problem: str = Field(description="One sentence describing main pain point")
    market_size: str = Field(description="TAM in one line, e.g. '$50B TAM'")
    opportunity_score: int = Field(ge=1, le=10, description="Score from 1-10")
    growth_rate: Optional[str] = Field(default=None, description="YoY growth percentage")
    key_trend: Optional[str] = Field(default=None, description="Main trend in sector")


class MarketExploration(BaseModel):
    """Complete market exploration results"""
    explored_sectors: List[SectorInsight]
    emerging_opportunities: List[str] = Field(description="New areas discovered")
    cross_sector_synergies: List[str] = Field(description="Opportunities at intersection")
    technology_enablers: List[str] = Field(description="Tech creating opportunities")
    priority_areas: List[str] = Field(description="Top 10 most promising areas")


class Problem(BaseModel):
    """A specific problem to solve"""
    problem_id: str = Field(pattern=r"^P\d+$", description="ID like P1, P2, etc")
    title: str = Field(description="One line problem statement")
    description: str = Field(max_length=300, description="2-3 sentences max")
    affected_users: str = Field(description="Number and segment")
    pain_cost: str = Field(description="Dollar amount or time lost")
    opportunity_score: int = Field(ge=1, le=10)
    revenue_potential: str = Field(description="Revenue range like '$10K-50K MRR'")


class CompetitorAnalysis(BaseModel):
    """Analysis of a single competitor"""
    name: str
    pricing: str = Field(description="Pricing model and tiers")
    strengths: List[str] = Field(max_items=3)
    weaknesses: List[str] = Field(max_items=3)
    market_share: Optional[str] = None


class OpportunityValidation(BaseModel):
    """Validation details for an opportunity"""
    viable: bool
    confidence: int = Field(ge=0, le=100, description="Confidence percentage")
    reason: str = Field(max_length=200, description="One line reason")
    revenue_potential: str = Field(description="Like '$10K-50K MRR'")
    break_even_months: Optional[int] = Field(ge=1, le=36, default=None)


class MarketOpportunity(BaseModel):
    """A complete market opportunity"""
    solution_name: str = Field(description="Memorable, brandable name")
    tagline: str = Field(max_length=100, description="Clear value proposition")
    problem_description: str
    affected_audience: str
    monetization_model: str
    market_size: str = Field(description="TAM, SAM, SOM with numbers")
    competition_level: str = Field(pattern=r"^(Low|Medium|High)$")
    implementation_difficulty: str = Field(pattern=r"^(Low|Medium|High)$")
    potential_revenue: str
    mvp_features: List[str] = Field(max_items=5)
    go_to_market: str = Field(max_length=300)
    unique_advantages: List[str] = Field(max_items=3)
    validation_confidence: int = Field(ge=0, le=100)


class GeminiStructuredResearch:
    """Market research using Gemini's structured output capabilities"""
    
    DEFAULT_FOCUS_AREAS = [
        "Small business operations",
        "Personal productivity", 
        "Health and wellness",
        "Education and learning",
        "Financial management",
        "Community and social needs",
        "Environmental solutions",
        "Remote work challenges",
        "Agriculture and Food Security",
        "Transportation and Logistics",
        "Creative Industries and Entertainment",
        "Real Estate and Housing",
        "Cybersecurity and Digital Privacy",
        "Travel and Hospitality",
        "Retail and E-commerce",
        "Parenting and Family Management",
        "Renewable Energy and Power Solutions",
        "Government Services and Civic Technology (GovTech)",
        "Manufacturing and Local Production",
        "Legal Technology and Access to Justice",
        "Water, Sanitation, and Hygiene (WASH)",
        "Mining and Solid Minerals Development",
        "Waste Management and The Circular Economy",
        "Human Capital and Workforce Development",
        "Insurance and Risk Management (InsurTech)"
    ]
    
    def __init__(self, chunk_size: int = 5):
        """Initialize with chunk size for processing sectors"""
        self.logger = logging.getLogger(__name__)
        self.ai_brain = IntelligentAIBrain()
        self.chunk_size = chunk_size
        self.results = {
            "sectors_analyzed": [],
            "problems_discovered": [],
            "opportunities_validated": []
        }
    
    async def conduct_structured_research(self, 
                                        focus_areas: Optional[List[str]] = None) -> List[MarketOpportunity]:
        """
        Conduct research using Gemini's structured output.
        
        Args:
            focus_areas: Specific areas to research (uses defaults if not provided)
            
        Returns:
            List of validated market opportunities
        """
        print("🔬 Starting Gemini Structured Research with response_schema")
        
        areas = focus_areas or self.DEFAULT_FOCUS_AREAS
        print(f"📊 Analyzing {len(areas)} market sectors in chunks of {self.chunk_size}")
        
        # Process in chunks to avoid context limits
        all_problems = []
        
        for i in range(0, len(areas), self.chunk_size):
            chunk = areas[i:i + self.chunk_size]
            print(f"\n🔄 Processing chunk {i//self.chunk_size + 1}/{(len(areas)-1)//self.chunk_size + 1}")
            
            # Phase 1: Explore sectors
            insights = await self._explore_sectors_structured(chunk)
            
            # Phase 2: Discover problems
            problems = await self._discover_problems_structured(insights)
            all_problems.extend(problems)
            
            # Small delay between chunks
            await asyncio.sleep(1)
        
        # Phase 3: Validate top problems
        print(f"\n🎯 Validating top {min(10, len(all_problems))} problems...")
        top_problems = sorted(all_problems, key=lambda x: x.opportunity_score, reverse=True)[:10]
        
        opportunities = []
        for problem in top_problems:
            opp = await self._create_opportunity_structured(problem)
            if opp and opp.validation_confidence > 70:
                opportunities.append(opp)
        
        print(f"\n✅ Found {len(opportunities)} validated opportunities")
        return opportunities
    
    async def _explore_sectors_structured(self, sectors: List[str]) -> List[SectorInsight]:
        """Explore sectors using structured output"""
        prompt = f"""
        Analyze these market sectors and identify the biggest opportunity in each:
        {', '.join(sectors)}
        
        For each sector, identify:
        - The single biggest unsolved problem
        - Market size (TAM)
        - Opportunity score (1-10)
        - Growth rate if available
        - Key trend driving the opportunity
        
        Focus on problems that can be solved with software/digital solutions.
        """
        
        try:
            # Use structured output with response schema
            response = await self.ai_brain.generate_enhanced_response(
                prompt,
                model='gemini-pro',
                response_mime_type="application/json",
                response_schema=List[SectorInsight]
            )
            
            # Parse the structured response
            if isinstance(response, dict) and 'content' in response:
                insights = json.loads(response['content'])
                return [SectorInsight(**item) for item in insights]
            return []
            
        except Exception as e:
            self.logger.error(f"Error in sector exploration: {e}")
            return []
    
    async def _discover_problems_structured(self, insights: List[SectorInsight]) -> List[Problem]:
        """Discover specific problems using structured output"""
        if not insights:
            return []
        
        # Focus on top sectors by opportunity score
        top_sectors = sorted(insights, key=lambda x: x.opportunity_score, reverse=True)[:3]
        
        prompt = f"""
        Based on these sector insights, identify 2-3 specific, high-impact problems:
        
        {chr(10).join([f"- {s.sector_name}: {s.biggest_problem} (Score: {s.opportunity_score}/10)" for s in top_sectors])}
        
        For each problem:
        - Be specific about who is affected and how
        - Quantify the pain (time/money lost)
        - Focus on problems that can generate recurring revenue
        - Ensure technical feasibility with current technology
        
        Prioritize problems with clear monetization paths.
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(
                prompt,
                model='gemini-pro',
                response_mime_type="application/json",
                response_schema=List[Problem]
            )
            
            if isinstance(response, dict) and 'content' in response:
                problems = json.loads(response['content'])
                return [Problem(**item) for item in problems]
            return []
            
        except Exception as e:
            self.logger.error(f"Error discovering problems: {e}")
            return []
    
    async def _create_opportunity_structured(self, problem: Problem) -> Optional[MarketOpportunity]:
        """Create and validate a market opportunity"""
        prompt = f"""
        Create a software solution for this validated problem:
        
        Problem: {problem.title}
        Description: {problem.description}
        Affected Users: {problem.affected_users}
        Pain Cost: {problem.pain_cost}
        Revenue Potential: {problem.revenue_potential}
        
        Design a solution that:
        1. Directly addresses the core problem
        2. Can be built by a small team in 3-6 months
        3. Has clear differentiation from existing solutions
        4. Can generate {problem.revenue_potential} within 12 months
        5. Uses modern technology stack (AI, cloud, mobile)
        
        Include:
        - Memorable product name and tagline
        - 3-5 MVP features that solve the core problem
        - Specific go-to-market strategy
        - Realistic monetization model with pricing
        - 3 unique advantages over competition
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(
                prompt,
                model='gemini-pro',
                response_mime_type="application/json",
                response_schema=MarketOpportunity
            )
            
            if isinstance(response, dict) and 'content' in response:
                opportunity = MarketOpportunity(**json.loads(response['content']))
                return opportunity
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating opportunity: {e}")
            return None
    
    def save_results(self, opportunities: List[MarketOpportunity], filename: str = None):
        """Save research results to file"""
        if not filename:
            filename = f"structured_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "opportunities": [opp.dict() for opp in opportunities],
            "summary": {
                "total_opportunities": len(opportunities),
                "avg_confidence": sum(o.validation_confidence for o in opportunities) / len(opportunities) if opportunities else 0,
                "sectors_covered": list(set(self.results["sectors_analyzed"]))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return filename


async def test_structured_research():
    """Test the structured research engine"""
    print("🚀 Testing Gemini Structured Research Engine")
    print("=" * 60)
    
    # Create engine with smaller chunk size for testing
    engine = GeminiStructuredResearch(chunk_size=3)
    
    # Test with a subset of sectors
    test_sectors = [
        "Small business operations",
        "Personal productivity",
        "Health and wellness",
        "Remote work challenges",
        "Financial management",
        "Education and learning"
    ]
    
    # Run research
    opportunities = await engine.conduct_structured_research(test_sectors)
    
    # Display results
    print(f"\n📊 Research Complete - Found {len(opportunities)} opportunities\n")
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp.solution_name} - {opp.tagline}")
        print(f"   💰 Revenue: {opp.potential_revenue}")
        print(f"   🎯 Market: {opp.market_size}")
        print(f"   🏆 Competition: {opp.competition_level}")
        print(f"   ✅ Confidence: {opp.validation_confidence}%")
        print(f"   🚀 MVP Features: {', '.join(opp.mvp_features[:3])}")
        print()
    
    # Save results
    filename = engine.save_results(opportunities)
    print(f"\n✅ Full results saved to {filename}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    asyncio.run(test_structured_research())