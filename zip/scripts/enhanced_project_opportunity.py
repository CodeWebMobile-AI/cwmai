"""
Enhanced Project Opportunity with Detailed Monetization

This module extends the ProjectOpportunity class to include comprehensive
monetization details and integrates with the business market research engine.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from scripts.business_market_research_engine import (
    MonetizationModel, 
    MarketValidation,
    BusinessOpportunity
)


@dataclass
class EnhancedProjectOpportunity:
    """Enhanced project opportunity with full business and monetization details."""
    
    # Basic Information
    id: str
    title: str
    tagline: str
    description: str
    
    # Problem & Solution
    problem_statement: str
    solution_approach: str
    unique_value_proposition: str
    
    # Technical Details
    tech_stack: List[str]
    architecture_type: str  # Always 'laravel-react' per system requirement
    estimated_complexity: str  # 'low', 'medium', 'high'
    
    # Market Information
    target_market: Dict[str, Any]  # Detailed target market info
    market_validation: MarketValidation
    competitor_advantages: List[str]
    
    # Monetization Details
    monetization_model: MonetizationModel
    revenue_projections: Dict[int, float]  # month: MRR
    monetization_strategy_content: Optional[str] = None  # Full markdown content
    
    # Scoring & Priority
    market_demand_score: float  # 0.0 to 1.0
    innovation_score: float  # 0.0 to 1.0
    monetization_clarity_score: float  # 0.0 to 1.0
    portfolio_fit_score: float  # 0.0 to 1.0
    total_opportunity_score: float  # 0.0 to 1.0
    
    # Implementation Details
    time_to_market_days: int
    time_to_revenue_days: int
    initial_features: List[str]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "market_research"  # 'market_research', 'portfolio_gap', 'ai_generated'
    
    def to_task_metadata(self) -> Dict[str, Any]:
        """Convert to metadata format for task creation."""
        return {
            'opportunity_id': self.id,
            'project_type': self.title,
            'tagline': self.tagline,
            'description': self.description,
            'problem_statement': self.problem_statement,
            'solution_approach': self.solution_approach,
            'unique_value_proposition': self.unique_value_proposition,
            'tech_stack': self.tech_stack,
            'architecture_type': self.architecture_type,
            'target_market': self.target_market,
            'monetization_model': {
                'type': self.monetization_model.model_type,
                'target_arpu': self.monetization_model.target_arpu,
                'pricing_tiers': self.monetization_model.pricing_tiers,
                'revenue_streams': self.monetization_model.revenue_streams
            },
            'scores': {
                'market_demand': self.market_demand_score,
                'innovation': self.innovation_score,
                'monetization_clarity': self.monetization_clarity_score,
                'portfolio_fit': self.portfolio_fit_score,
                'total': self.total_opportunity_score
            },
            'implementation': {
                'complexity': self.estimated_complexity,
                'time_to_market': self.time_to_market_days,
                'time_to_revenue': self.time_to_revenue_days,
                'initial_features': self.initial_features
            }
        }
    
    def to_business_opportunity(self) -> BusinessOpportunity:
        """Convert to BusinessOpportunity format."""
        return BusinessOpportunity(
            id=self.id,
            name=self.title,
            tagline=self.tagline,
            problem_statement=self.problem_statement,
            solution_description=self.solution_approach,
            unique_value_proposition=self.unique_value_proposition,
            target_audience=self.target_market,
            market_validation=self.market_validation,
            monetization_model=self.monetization_model,
            implementation_complexity=self.estimated_complexity,
            time_to_market=self.time_to_market_days,
            time_to_revenue=self.time_to_revenue_days,
            total_score=self.total_opportunity_score,
            revenue_projections=self.revenue_projections
        )
    
    def get_priority(self) -> str:
        """Determine task priority based on scores."""
        if self.total_opportunity_score > 0.8 and self.market_demand_score > 0.8:
            return "CRITICAL"
        elif self.total_opportunity_score > 0.7:
            return "HIGH"
        elif self.total_opportunity_score > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_estimated_cycles(self) -> int:
        """Get estimated work cycles based on complexity."""
        complexity_cycles = {
            'low': 3,
            'medium': 5,
            'high': 8
        }
        return complexity_cycles.get(self.estimated_complexity, 5)
    
    def should_proceed(self) -> bool:
        """Determine if this opportunity should be pursued."""
        return (
            self.total_opportunity_score > 0.7 and
            self.monetization_clarity_score > 0.6 and
            self.market_demand_score > 0.6
        )
    
    def get_monetization_summary(self) -> str:
        """Get a brief monetization summary."""
        model = self.monetization_model
        return (
            f"{model.model_type.upper()} model targeting ${model.target_arpu} ARPU. "
            f"LTV:CAC ratio {model.ltv_cac_ratio:.1f}:1. "
            f"Primary revenue: {list(model.revenue_streams.keys())[0]}. "
            f"Payback: {model.payback_months} months."
        )
    
    def get_first_year_revenue_projection(self) -> float:
        """Get total projected revenue for first year."""
        return sum(self.revenue_projections.values())
    
    def to_work_item_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for work item creation."""
        return {
            'type': 'NEW_PROJECT',
            'title': f"Create {self.title} - {self.tagline}",
            'description': (
                f"{self.description}\n\n"
                f"**Problem:** {self.problem_statement}\n"
                f"**Solution:** {self.solution_approach}\n"
                f"**Monetization:** {self.get_monetization_summary()}\n"
                f"**First Year Revenue Potential:** ${self.get_first_year_revenue_projection():,.0f}"
            ),
            'priority': self.get_priority(),
            'metadata': self.to_task_metadata(),
            'estimated_cycles': self.get_estimated_cycles()
        }


class ProjectOpportunityFactory:
    """Factory for creating enhanced project opportunities."""
    
    @staticmethod
    def from_business_opportunity(
        opp: BusinessOpportunity,
        tech_stack: List[str] = None,
        architecture_type: str = "laravel-react"
    ) -> EnhancedProjectOpportunity:
        """Create enhanced opportunity from business opportunity."""
        
        # Always use Laravel React stack - enforcing system requirement
        tech_stack = ["php", "laravel", "react", "typescript", "mysql", "redis", "laravel-echo-server"]
        architecture_type = "laravel-react"  # Force Laravel React architecture
        
        # Calculate additional scores
        monetization_clarity = 0.9 if opp.monetization_model.pricing_tiers else 0.5
        portfolio_fit = 0.8  # Default high fit for new opportunities
        
        # Extract initial features from solution
        initial_features = [
            "User authentication and onboarding",
            "Core value delivery feature",
            "Basic analytics dashboard",
            "Billing integration",
            "API foundation"
        ]
        
        return EnhancedProjectOpportunity(
            id=opp.id,
            title=opp.name,
            tagline=opp.tagline,
            description=opp.solution_description,
            problem_statement=opp.problem_statement,
            solution_approach=opp.solution_description,
            unique_value_proposition=opp.unique_value_proposition,
            tech_stack=tech_stack,
            architecture_type="laravel-react",  # Always Laravel React
            estimated_complexity=opp.implementation_complexity,
            target_market=opp.target_audience,
            market_validation=opp.market_validation,
            competitor_advantages=[
                "First-mover in specific niche",
                "Superior user experience",
                "AI-powered automation"
            ],
            monetization_model=opp.monetization_model,
            revenue_projections=opp.revenue_projections,
            market_demand_score=opp.market_validation.validation_score,
            innovation_score=0.8,  # Default high for new projects
            monetization_clarity_score=monetization_clarity,
            portfolio_fit_score=portfolio_fit,
            total_opportunity_score=opp.total_score,
            time_to_market_days=opp.time_to_market,
            time_to_revenue_days=opp.time_to_revenue,
            initial_features=initial_features
        )
    
    @staticmethod
    def from_ai_idea(
        idea: Dict[str, Any],
        market_validation: MarketValidation = None,
        monetization_model: MonetizationModel = None
    ) -> EnhancedProjectOpportunity:
        """Create enhanced opportunity from AI-generated idea."""
        
        # Generate ID
        import hashlib
        opp_id = hashlib.md5(
            f"{idea['title']}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:8]
        
        # Create default monetization if not provided
        if not monetization_model:
            monetization_model = MonetizationModel(
                model_type='saas',
                pricing_tiers=[
                    {'name': 'Free', 'price': 0},
                    {'name': 'Pro', 'price': 29},
                    {'name': 'Enterprise', 'price': 'custom'}
                ],
                revenue_streams={'subscriptions': 0.8, 'add_ons': 0.2},
                target_arpu=50,
                target_cac=100,
                target_ltv=1200,
                ltv_cac_ratio=12.0,
                payback_months=3,
                gross_margin_target=0.8,
                twenty_four_seven_mechanisms=['self-service', 'automated billing']
            )
        
        # Create default market validation if not provided
        if not market_validation:
            market_validation = MarketValidation(
                problem_validation={'validated': True, 'severity': 0.8},
                market_size={'tam': 1000000000, 'sam': 100000000, 'som': 10000000},
                competitor_analysis=[],
                target_segments=[{'name': 'small businesses', 'size': 1000000}],
                willingness_to_pay={'average': 50, 'range_low': 20, 'range_high': 100},
                acquisition_channels=[{'channel': 'content', 'cac': 50}],
                validation_score=0.8
            )
        
        # Default revenue projections
        revenue_projections = {}
        base_customers = 10
        for month in range(1, 13):
            customers = int(base_customers * (1.4 ** (month - 1)))
            revenue_projections[month] = customers * monetization_model.target_arpu
        
        return EnhancedProjectOpportunity(
            id=opp_id,
            title=idea.get('title', 'Untitled'),
            tagline=idea.get('tagline', idea.get('description', '')[:50]),
            description=idea.get('description', ''),
            problem_statement=idea.get('problem', idea.get('description', '')),
            solution_approach=idea.get('solution', idea.get('description', '')),
            unique_value_proposition=idea.get('unique_value', '10x better solution'),
            tech_stack=['laravel', 'react', 'mysql', 'redis', 'laravel-echo-server'],  # Always use full stack
            architecture_type='laravel-react',
            estimated_complexity=idea.get('complexity', 'medium'),
            target_market={'primary': idea.get('target', 'businesses')},
            market_validation=market_validation,
            competitor_advantages=['innovative approach'],
            monetization_model=monetization_model,
            revenue_projections=revenue_projections,
            market_demand_score=0.7,
            innovation_score=0.8,
            monetization_clarity_score=0.7,
            portfolio_fit_score=0.8,
            total_opportunity_score=0.75,
            time_to_market_days=30,
            time_to_revenue_days=45,
            initial_features=['core feature', 'user management', 'billing'],
            source='ai_generated'
        )