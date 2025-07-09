"""
Unified Project Discovery System

A streamlined approach to discovering real-world problems and converting them
into actionable Laravel React projects with a consistent tech stack.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from scripts.ai_brain import IntelligentAIBrain
from scripts.gemini_deep_research import GeminiDeepResearch


# System-wide configuration
SYSTEM_CONFIG = {
    "tech_stack": {
        "backend": ["laravel", "php"],
        "frontend": ["react", "typescript"],
        "database": ["mysql"],
        "cache": ["redis"],
        "realtime": ["laravel-echo-server"]
    },
    "project_defaults": {
        "type": "Web Application",
        "architecture": "laravel-react",
        "starter_kit": "https://github.com/laravel/react-starter-kit"
    }
}


@dataclass
class MarketValidation:
    """Market validation data."""
    validated: bool
    confidence_score: float
    market_size: str
    competition_level: str
    demand_evidence: List[str]
    unique_advantages: List[str]


@dataclass
class MonetizationModel:
    """Monetization strategy."""
    model_type: str  # 'saas', 'marketplace', 'freemium'
    pricing_tiers: List[Dict[str, Any]]
    revenue_streams: List[str]
    estimated_revenue: str


@dataclass
class UnifiedProjectOpportunity:
    """Single data model that progressively builds from problem to project."""
    
    # Phase 1: Problem Discovery (from Gemini research)
    problem_description: str
    affected_audience: str
    current_pain_points: List[str]
    severity_score: float
    
    # Phase 2: Market Validation (from Gemini research)
    market_validation: MarketValidation
    
    # Phase 3: Solution Design (generated)
    title: str = ""
    tagline: str = ""
    solution_approach: str = ""
    unique_value_proposition: str = ""
    
    # Phase 4: Business Model (generated)
    monetization_model: MonetizationModel = None
    target_market_size: str = ""
    
    # Phase 5: Technical Implementation (standardized)
    tech_stack: List[str] = field(default_factory=list)
    initial_features: List[str] = field(default_factory=list)
    complexity_estimate: str = "medium"
    
    # Phase 6: Project Metadata
    repository_name: str = ""
    created_at: str = ""
    opportunity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem_description": self.problem_description,
            "affected_audience": self.affected_audience,
            "current_pain_points": self.current_pain_points,
            "severity_score": self.severity_score,
            "market_validation": {
                "validated": self.market_validation.validated,
                "confidence_score": self.market_validation.confidence_score,
                "market_size": self.market_validation.market_size,
                "competition_level": self.market_validation.competition_level,
                "demand_evidence": self.market_validation.demand_evidence,
                "unique_advantages": self.market_validation.unique_advantages
            } if self.market_validation else None,
            "title": self.title,
            "tagline": self.tagline,
            "solution_approach": self.solution_approach,
            "unique_value_proposition": self.unique_value_proposition,
            "monetization_model": {
                "model_type": self.monetization_model.model_type,
                "pricing_tiers": self.monetization_model.pricing_tiers,
                "revenue_streams": self.monetization_model.revenue_streams,
                "estimated_revenue": self.monetization_model.estimated_revenue
            } if self.monetization_model else None,
            "target_market_size": self.target_market_size,
            "tech_stack": self.tech_stack,
            "initial_features": self.initial_features,
            "complexity_estimate": self.complexity_estimate,
            "repository_name": self.repository_name,
            "created_at": self.created_at,
            "opportunity_score": self.opportunity_score
        }


class UnifiedProjectDiscovery:
    """Unified system for discovering and creating projects."""
    
    def __init__(self, ai_brain: IntelligentAIBrain = None):
        """Initialize the unified discovery system."""
        self.ai_brain = ai_brain or IntelligentAIBrain()
        self.gemini_research = GeminiDeepResearch(self.ai_brain)
        self.logger = logging.getLogger(__name__)
        
    async def discover_and_create_projects(self, 
                                         count: int = 10,
                                         existing_portfolio: Dict[str, Any] = None) -> List[UnifiedProjectOpportunity]:
        """Main pipeline: Discover problems → Design solutions → Create projects.
        
        Args:
            count: Number of projects to generate
            existing_portfolio: Existing projects to avoid duplicates
            
        Returns:
            List of fully-formed project opportunities
        """
        print("🚀 Unified Project Discovery Pipeline")
        print("=" * 60)
        
        # Phase 1: Discover real-world problems
        print("\n📍 Phase 1: Discovering Real-World Problems...")
        market_opportunities = await self.gemini_research.conduct_deep_market_research(existing_portfolio)
        
        if not market_opportunities:
            print("❌ No market opportunities found")
            return []
            
        print(f"✅ Found {len(market_opportunities)} validated market opportunities")
        
        # Phase 2: Convert to unified opportunities
        print("\n📍 Phase 2: Converting to Unified Opportunities...")
        unified_opportunities = self._convert_market_opportunities(market_opportunities)
        
        # Phase 3: Design solutions for each opportunity
        print("\n📍 Phase 3: Designing Solutions...")
        await self._design_solutions(unified_opportunities)
        
        # Phase 4: Create business models
        print("\n📍 Phase 4: Creating Business Models...")
        await self._create_business_models(unified_opportunities)
        
        # Phase 5: Apply standard tech stack
        print("\n📍 Phase 5: Applying Standard Tech Stack...")
        self._apply_tech_stack(unified_opportunities)
        
        # Phase 6: Generate initial features
        print("\n📍 Phase 6: Generating Initial Features...")
        await self._generate_initial_features(unified_opportunities)
        
        # Phase 7: Filter and rank
        print("\n📍 Phase 7: Filtering and Ranking...")
        final_projects = self._filter_and_rank(unified_opportunities, count)
        
        print(f"\n✅ Generated {len(final_projects)} project opportunities")
        return final_projects
        
    def _convert_market_opportunities(self, market_opps: List[Any]) -> List[UnifiedProjectOpportunity]:
        """Convert Gemini market opportunities to unified format."""
        unified = []
        
        for opp in market_opps:
            # Extract pain points from problem description
            pain_points = [
                opp.problem_description,
                f"Current solutions: {opp.current_solutions}",
                f"Opportunity: {opp.opportunity}"
            ]
            
            # Create market validation
            validation = MarketValidation(
                validated=True,
                confidence_score=0.85,  # High confidence since Gemini validated
                market_size=opp.market_size,
                competition_level=opp.competition_level,
                demand_evidence=opp.validation_sources,
                unique_advantages=opp.unique_advantages
            )
            
            # Create unified opportunity
            unified_opp = UnifiedProjectOpportunity(
                problem_description=opp.problem_description,
                affected_audience=opp.affected_audience,
                current_pain_points=pain_points,
                severity_score=8.0,  # High severity since validated
                market_validation=validation,
                created_at=datetime.now().isoformat()
            )
            
            unified.append(unified_opp)
            
        return unified
        
    async def _design_solutions(self, opportunities: List[UnifiedProjectOpportunity]) -> None:
        """Design solutions for each opportunity."""
        
        for i, opp in enumerate(opportunities):
            print(f"  Designing solution {i+1}/{len(opportunities)}...")
            
            prompt = f"""
            Design a web-based solution for this validated problem.
            
            Problem: {opp.problem_description}
            Audience: {opp.affected_audience}
            Market Size: {opp.market_validation.market_size}
            Unique Advantages: {', '.join(opp.market_validation.unique_advantages[:3])}
            
            Create:
            1. A memorable, brandable project name (one or two words)
            2. A clear tagline (max 10 words)
            3. A solution approach (2-3 sentences)
            4. A unique value proposition (1 sentence)
            5. A repository name (lowercase, hyphens)
            
            Focus on practical, implementable solutions that can be built
            with a small team and generate recurring revenue.
            
            Return as JSON:
            {{
                "title": "ProjectName",
                "tagline": "Clear value in few words",
                "solution_approach": "How it solves the problem",
                "unique_value_proposition": "Why it's 10x better",
                "repository_name": "project-name"
            }}
            """
            
            try:
                response = await self.ai_brain.generate_enhanced_response(prompt)
                if response and response.get('content'):
                    # Parse response
                    solution_data = self._parse_json_response(response['content'])
                    
                    if solution_data:
                        opp.title = solution_data.get('title', f'Solution{i+1}')
                        opp.tagline = solution_data.get('tagline', 'Innovative solution')
                        opp.solution_approach = solution_data.get('solution_approach', '')
                        opp.unique_value_proposition = solution_data.get('unique_value_proposition', '')
                        opp.repository_name = solution_data.get('repository_name', f'project-{i+1}')
            except Exception as e:
                self.logger.error(f"Error designing solution: {e}")
                # Set defaults
                opp.title = f"Solution{i+1}"
                opp.repository_name = f"project-{i+1}"
                
    async def _create_business_models(self, opportunities: List[UnifiedProjectOpportunity]) -> None:
        """Create business models for each opportunity."""
        
        for i, opp in enumerate(opportunities):
            print(f"  Creating business model {i+1}/{len(opportunities)}...")
            
            prompt = f"""
            Create a monetization model for this solution.
            
            Solution: {opp.title}
            Problem: {opp.problem_description}
            Target: {opp.affected_audience}
            Market Size: {opp.market_validation.market_size}
            
            Design a 24/7 revenue model with:
            1. Model type (saas/marketplace/freemium)
            2. 3 pricing tiers with specific prices
            3. Multiple revenue streams
            4. Realistic revenue estimate for year 1
            
            Return as JSON:
            {{
                "model_type": "saas",
                "pricing_tiers": [
                    {{"name": "Free", "price": 0, "features": ["basic"]}},
                    {{"name": "Pro", "price": 29, "features": ["advanced"]}},
                    {{"name": "Enterprise", "price": 99, "features": ["all"]}}
                ],
                "revenue_streams": ["subscriptions", "api_usage", "premium_features"],
                "estimated_revenue": "$50K-200K ARR"
            }}
            """
            
            try:
                response = await self.ai_brain.generate_enhanced_response(prompt)
                if response and response.get('content'):
                    model_data = self._parse_json_response(response['content'])
                    
                    if model_data:
                        opp.monetization_model = MonetizationModel(
                            model_type=model_data.get('model_type', 'saas'),
                            pricing_tiers=model_data.get('pricing_tiers', []),
                            revenue_streams=model_data.get('revenue_streams', ['subscriptions']),
                            estimated_revenue=model_data.get('estimated_revenue', '$10K-50K ARR')
                        )
            except Exception as e:
                self.logger.error(f"Error creating business model: {e}")
                # Set default
                opp.monetization_model = MonetizationModel(
                    model_type='saas',
                    pricing_tiers=[
                        {"name": "Free", "price": 0},
                        {"name": "Pro", "price": 29},
                        {"name": "Enterprise", "price": 99}
                    ],
                    revenue_streams=['subscriptions'],
                    estimated_revenue='$10K-50K ARR'
                )
                
    def _apply_tech_stack(self, opportunities: List[UnifiedProjectOpportunity]) -> None:
        """Apply standardized tech stack to all opportunities."""
        
        # Build full tech stack from config
        tech_stack = []
        for category, techs in SYSTEM_CONFIG["tech_stack"].items():
            tech_stack.extend(techs)
            
        # Apply to all opportunities
        for opp in opportunities:
            opp.tech_stack = tech_stack.copy()
            
    async def _generate_initial_features(self, opportunities: List[UnifiedProjectOpportunity]) -> None:
        """Generate initial features for each project."""
        
        for i, opp in enumerate(opportunities):
            print(f"  Generating features {i+1}/{len(opportunities)}...")
            
            prompt = f"""
            Generate 5 specific, implementable features for the MVP.
            
            Project: {opp.title}
            Solution: {opp.solution_approach}
            Monetization: {opp.monetization_model.model_type if opp.monetization_model else 'saas'}
            Target Users: {opp.affected_audience}
            
            Features should be:
            1. Core to solving the main problem
            2. Implementable in 1-2 weeks each
            3. Directly tied to revenue generation
            4. Specific and measurable
            
            Return as JSON array of strings:
            ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
            """
            
            try:
                response = await self.ai_brain.generate_enhanced_response(prompt)
                if response and response.get('content'):
                    features = self._parse_json_array_response(response['content'])
                    
                    if features:
                        opp.initial_features = features[:5]
                    else:
                        opp.initial_features = self._get_default_features()
            except Exception as e:
                self.logger.error(f"Error generating features: {e}")
                opp.initial_features = self._get_default_features()
                
    def _filter_and_rank(self, opportunities: List[UnifiedProjectOpportunity], 
                        count: int) -> List[UnifiedProjectOpportunity]:
        """Filter and rank opportunities by score."""
        
        # Calculate opportunity scores
        for opp in opportunities:
            score = 0.0
            
            # Market validation score (40%)
            if opp.market_validation:
                score += opp.market_validation.confidence_score * 0.4
                
            # Solution completeness (30%)
            if opp.title and opp.solution_approach:
                score += 0.3
                
            # Business model clarity (20%)
            if opp.monetization_model:
                score += 0.2
                
            # Technical readiness (10%)
            if opp.initial_features:
                score += 0.1
                
            opp.opportunity_score = score
            
        # Sort by score and return top N
        opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
        return opportunities[:count]
        
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response."""
        try:
            # Remove markdown code blocks
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Try direct parse
            try:
                return json.loads(content)
            except:
                # Find JSON object
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON: {e}")
        return None
        
    def _parse_json_array_response(self, content: str) -> Optional[List[Any]]:
        """Parse JSON array from AI response."""
        try:
            # Remove markdown code blocks
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Try direct parse
            try:
                return json.loads(content)
            except:
                # Find JSON array
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON array: {e}")
        return None
        
    def _get_default_features(self) -> List[str]:
        """Get default features for any project."""
        return [
            "User authentication and profile management",
            "Core problem-solving functionality",
            "Dashboard with key metrics",
            "Payment integration for subscriptions",
            "API for third-party integrations"
        ]
        

async def demonstrate_unified_discovery():
    """Demonstrate the unified discovery system."""
    print("🚀 Demonstrating Unified Project Discovery")
    print("=" * 60)
    
    # Initialize system
    discovery = UnifiedProjectDiscovery()
    
    # Discover and create projects
    projects = await discovery.discover_and_create_projects(count=5)
    
    # Display results
    print("\n📊 Generated Projects:")
    print("=" * 60)
    
    for i, project in enumerate(projects, 1):
        print(f"\n{i}. {project.title}")
        print(f"   📝 {project.tagline}")
        print(f"   🎯 Target: {project.affected_audience}")
        print(f"   💡 Solution: {project.solution_approach}")
        print(f"   💰 Model: {project.monetization_model.model_type if project.monetization_model else 'N/A'}")
        print(f"   🛠️ Tech: {', '.join(project.tech_stack[:5])}...")
        print(f"   ⭐ Score: {project.opportunity_score:.2f}")
        print(f"   📁 Repo: {project.repository_name}")
        
    # Save results
    results = [p.to_dict() for p in projects]
    with open('unified_discovery_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to unified_discovery_results.json")


if __name__ == "__main__":
    asyncio.run(demonstrate_unified_discovery())