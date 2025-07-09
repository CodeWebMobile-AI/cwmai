"""
Business-Focused Market Research Engine

This module provides market analysis focused on business opportunities,
revenue validation, and monetization potential rather than technical trends.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import hashlib

from scripts.ai_brain import IntelligentAIBrain


@dataclass
class MonetizationModel:
    """Detailed monetization model for a project."""
    model_type: str  # 'saas', 'marketplace', 'usage_based', 'transaction', 'advertising'
    pricing_tiers: List[Dict[str, Any]]
    revenue_streams: Dict[str, float]  # stream_name: percentage_of_revenue
    target_arpu: float
    target_cac: float
    target_ltv: float
    ltv_cac_ratio: float
    payback_months: int
    gross_margin_target: float
    twenty_four_seven_mechanisms: List[str]  # How it makes money while you sleep


@dataclass
class MarketValidation:
    """Market validation data for a business opportunity."""
    problem_validation: Dict[str, Any]
    market_size: Dict[str, float]  # TAM, SAM, SOM
    competitor_analysis: List[Dict[str, Any]]
    target_segments: List[Dict[str, Any]]
    willingness_to_pay: Dict[str, float]
    acquisition_channels: List[Dict[str, Any]]
    validation_score: float  # 0.0 to 1.0


@dataclass
class BusinessOpportunity:
    """A validated business opportunity with full monetization details."""
    id: str
    name: str
    tagline: str
    problem_statement: str
    solution_description: str
    unique_value_proposition: str
    target_audience: Dict[str, Any]
    market_validation: MarketValidation
    monetization_model: MonetizationModel
    implementation_complexity: str  # 'low', 'medium', 'high'
    time_to_market: int  # days
    time_to_revenue: int  # days
    total_score: float  # 0.0 to 1.0
    revenue_projections: Dict[int, float] = field(default_factory=dict)  # month: MRR


class BusinessMarketResearchEngine:
    """Engine for conducting business-focused market research."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize the business market research engine."""
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Cache for research results
        self.opportunity_cache: Dict[str, BusinessOpportunity] = {}
        self.validation_cache: Dict[str, MarketValidation] = {}
        self.last_research_time: Optional[datetime] = None
        self.research_interval_hours = 12  # More frequent for business opportunities
        
    async def discover_business_opportunities(
        self, 
        existing_portfolio: Dict[str, Any],
        max_opportunities: int = 5
    ) -> List[BusinessOpportunity]:
        """Discover and validate business opportunities.
        
        Args:
            existing_portfolio: Current portfolio to avoid duplication
            max_opportunities: Maximum opportunities to return
            
        Returns:
            List of validated business opportunities
        """
        self.logger.info("🔍 Discovering business opportunities...")
        
        opportunities = []
        
        # 1. Identify underserved markets
        market_gaps = await self._identify_market_gaps(existing_portfolio)
        
        # 2. Validate problems worth solving
        validated_problems = await self._validate_problems(market_gaps)
        
        # 3. Design monetization models
        for problem in validated_problems[:max_opportunities]:
            opportunity = await self._create_business_opportunity(problem)
            if opportunity and opportunity.total_score > 0.7:
                opportunities.append(opportunity)
        
        # 4. Sort by total score
        opportunities.sort(key=lambda x: x.total_score, reverse=True)
        
        return opportunities[:max_opportunities]
    
    async def _identify_market_gaps(self, existing_portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify market gaps and underserved segments."""
        prompt = """
        Analyze current market gaps and underserved business segments.
        Focus on B2B and B2C opportunities with clear monetization potential.
        
        Consider:
        1. Problems businesses pay to solve ($1000+/month potential)
        2. Consumer pain points with proven willingness to pay
        3. Inefficient markets ready for disruption
        4. Emerging needs from remote work, AI adoption, sustainability
        
        Avoid:
        - Developer tools and technical infrastructure
        - Saturated markets (basic CRM, blog platforms)
        - Low-value consumer apps (<$10/month potential)
        
        Return 5-7 market gaps as JSON:
        [{
            "market": "specific market segment",
            "problem": "clear problem description",
            "current_solutions": ["existing solution 1", "solution 2"],
            "why_underserved": "reason current solutions fail",
            "market_size_estimate": "rough TAM estimate",
            "urgency": "high/medium/low"
        }]
        """
        
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        if response and response.get('status') == 'success':
            try:
                result_text = response.get('result', '')
                # Extract JSON from response
                import re
                json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if json_match:
                    gaps = json.loads(json_match.group())
                    return gaps
            except Exception as e:
                self.logger.error(f"Error parsing market gaps: {e}")
        
        return []
    
    async def _validate_problems(self, market_gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that problems are worth solving."""
        validated = []
        
        for gap in market_gaps:
            validation_prompt = f"""
            Validate this business problem:
            Market: {gap['market']}
            Problem: {gap['problem']}
            
            Analyze:
            1. How much does this problem cost the target audience? (time/money)
            2. How frequently do they experience this problem?
            3. What are they currently paying to solve it?
            4. Would they switch to a 10x better solution?
            5. Can this support a subscription model?
            
            Return validation as JSON:
            {{
                "problem_cost_per_month": dollar amount,
                "frequency": "daily/weekly/monthly",
                "current_spend": dollar amount,
                "switching_likelihood": 0.0-1.0,
                "subscription_viable": true/false,
                "validation_score": 0.0-1.0,
                "key_insights": ["insight 1", "insight 2"]
            }}
            """
            
            response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': validation_prompt})
            
            if response and response.get('status') == 'success':
                try:
                    result_text = response.get('result', '')
                    # Extract JSON
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        validation = json.loads(json_match.group())
                        if validation.get('validation_score', 0) > 0.7:
                            gap.update(validation)
                            validated.append(gap)
                except Exception as e:
                    self.logger.error(f"Error validating problem: {e}")
        
        return validated
    
    async def _create_business_opportunity(self, validated_problem: Dict[str, Any]) -> Optional[BusinessOpportunity]:
        """Create a full business opportunity from a validated problem."""
        try:
            # Generate business details
            opportunity_prompt = f"""
            Create a complete business opportunity for:
            Problem: {validated_problem['problem']}
            Market: {validated_problem['market']}
            Current spend: ${validated_problem.get('current_spend', 0)}
            
            Design a modern, scalable solution with:
            1. Catchy brand name (like Stripe, Notion, Slack)
            2. Clear value proposition
            3. Specific target audience with buyer personas
            4. Unique approach that's 10x better
            5. Multiple revenue streams
            6. 24/7 automated revenue mechanisms
            
            Return as JSON:
            {{
                "name": "BrandName",
                "tagline": "Short memorable tagline",
                "solution": "How it solves the problem",
                "unique_value": "What makes it 10x better",
                "target_audience": {{
                    "primary": "Specific segment",
                    "size": number,
                    "characteristics": ["char1", "char2"]
                }},
                "core_features": ["feature1", "feature2"],
                "implementation_complexity": "low/medium/high",
                "time_to_market_days": number,
                "time_to_revenue_days": number
            }}
            """
            
            response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': opportunity_prompt})
            
            if not response or response.get('status') != 'success':
                return None
            
            # Parse response
            result_text = response.get('result', '')
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                return None
            
            opp_data = json.loads(json_match.group())
            
            # Create monetization model
            monetization = await self._design_monetization_model(opp_data, validated_problem)
            
            # Validate market
            market_validation = await self._validate_market(opp_data, validated_problem)
            
            # Calculate total score
            total_score = self._calculate_opportunity_score(
                validated_problem, 
                monetization, 
                market_validation
            )
            
            # Generate revenue projections
            revenue_projections = self._project_revenue(monetization, market_validation)
            
            # Create opportunity ID
            opp_id = hashlib.md5(
                f"{opp_data['name']}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]
            
            return BusinessOpportunity(
                id=opp_id,
                name=opp_data['name'],
                tagline=opp_data['tagline'],
                problem_statement=validated_problem['problem'],
                solution_description=opp_data['solution'],
                unique_value_proposition=opp_data['unique_value'],
                target_audience=opp_data['target_audience'],
                market_validation=market_validation,
                monetization_model=monetization,
                implementation_complexity=opp_data['implementation_complexity'],
                time_to_market=opp_data['time_to_market_days'],
                time_to_revenue=opp_data['time_to_revenue_days'],
                total_score=total_score,
                revenue_projections=revenue_projections
            )
            
        except Exception as e:
            self.logger.error(f"Error creating business opportunity: {e}")
            return None
    
    async def _design_monetization_model(
        self, 
        opportunity: Dict[str, Any], 
        problem: Dict[str, Any]
    ) -> MonetizationModel:
        """Design a detailed monetization model."""
        prompt = f"""
        Design a monetization model for {opportunity['name']}:
        Solution: {opportunity['solution']}
        Target: {opportunity['target_audience']['primary']}
        Current spend: ${problem.get('current_spend', 0)}
        
        Create a compelling pricing strategy:
        1. Model type (SaaS, marketplace, usage-based, etc.)
        2. Pricing tiers with clear value at each level
        3. Multiple revenue streams (primary + secondary)
        4. Price anchoring and psychology
        5. 24/7 revenue automation
        
        Return as JSON:
        {{
            "model_type": "saas/marketplace/usage_based/transaction",
            "pricing_tiers": [
                {{
                    "name": "Free/Starter/Pro/Enterprise",
                    "price": number,
                    "features": ["feature1", "feature2"],
                    "limits": {{}},
                    "target_user": "description"
                }}
            ],
            "revenue_streams": {{
                "subscriptions": 0.7,
                "usage_overages": 0.15,
                "add_ons": 0.1,
                "marketplace": 0.05
            }},
            "pricing_psychology": "explanation",
            "automation_mechanisms": [
                "Self-service signup",
                "Usage-based upgrades",
                "Failed payment recovery"
            ],
            "target_metrics": {{
                "arpu": number,
                "cac": number,
                "ltv": number,
                "payback_months": number,
                "gross_margin": 0.7
            }}
        }}
        """
        
        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        # Parse response with defaults
        try:
            result_text = response.get('result', '') if response else ''
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                model_data = json.loads(json_match.group())
                
                metrics = model_data.get('target_metrics', {})
                ltv = metrics.get('ltv', 1000)
                cac = metrics.get('cac', 100)
                
                return MonetizationModel(
                    model_type=model_data.get('model_type', 'saas'),
                    pricing_tiers=model_data.get('pricing_tiers', []),
                    revenue_streams=model_data.get('revenue_streams', {'subscriptions': 1.0}),
                    target_arpu=metrics.get('arpu', 50),
                    target_cac=cac,
                    target_ltv=ltv,
                    ltv_cac_ratio=ltv / max(cac, 1),
                    payback_months=metrics.get('payback_months', 6),
                    gross_margin_target=metrics.get('gross_margin', 0.7),
                    twenty_four_seven_mechanisms=model_data.get('automation_mechanisms', [])
                )
        except Exception as e:
            self.logger.error(f"Error parsing monetization model: {e}")
        
        # Return default model
        return MonetizationModel(
            model_type='saas',
            pricing_tiers=[],
            revenue_streams={'subscriptions': 1.0},
            target_arpu=50,
            target_cac=100,
            target_ltv=1000,
            ltv_cac_ratio=10.0,
            payback_months=6,
            gross_margin_target=0.7,
            twenty_four_seven_mechanisms=['Self-service']
        )
    
    async def _validate_market(
        self, 
        opportunity: Dict[str, Any], 
        problem: Dict[str, Any]
    ) -> MarketValidation:
        """Validate the market opportunity."""
        # This would ideally use real data sources
        # For now, we'll use AI to estimate based on available information
        
        validation_data = {
            'problem_validation': {
                'validated': True,
                'problem_severity': problem.get('validation_score', 0.8),
                'frequency': problem.get('frequency', 'weekly')
            },
            'market_size': {
                'tam': 1000000000,  # $1B default TAM
                'sam': 100000000,   # $100M SAM
                'som': 10000000     # $10M SOM
            },
            'competitor_analysis': [],
            'target_segments': [opportunity['target_audience']],
            'willingness_to_pay': {
                'average': problem.get('current_spend', 50),
                'range_low': problem.get('current_spend', 50) * 0.5,
                'range_high': problem.get('current_spend', 50) * 2
            },
            'acquisition_channels': [
                {'channel': 'content_seo', 'cac': 25},
                {'channel': 'product_led', 'cac': 10},
                {'channel': 'paid_search', 'cac': 75}
            ],
            'validation_score': problem.get('validation_score', 0.8)
        }
        
        return MarketValidation(**validation_data)
    
    def _calculate_opportunity_score(
        self, 
        problem: Dict[str, Any],
        monetization: MonetizationModel,
        validation: MarketValidation
    ) -> float:
        """Calculate total opportunity score."""
        scores = {
            'problem_severity': problem.get('validation_score', 0.7) * 0.2,
            'market_size': min(validation.market_size['som'] / 10000000, 1.0) * 0.15,
            'monetization_clarity': (1.0 if monetization.pricing_tiers else 0.5) * 0.2,
            'ltv_cac_ratio': min(monetization.ltv_cac_ratio / 3, 1.0) * 0.15,
            'automation': len(monetization.twenty_four_seven_mechanisms) / 5 * 0.15,
            'time_to_revenue': (1.0 if monetization.payback_months < 6 else 0.5) * 0.15
        }
        
        return sum(scores.values())
    
    def _project_revenue(
        self, 
        monetization: MonetizationModel,
        validation: MarketValidation
    ) -> Dict[int, float]:
        """Project revenue for first 12 months."""
        projections = {}
        
        # Simple exponential growth model
        starting_customers = 10
        growth_rate = 1.5  # 50% monthly growth
        
        for month in range(1, 13):
            customers = int(starting_customers * (growth_rate ** (month - 1)))
            mrr = customers * monetization.target_arpu
            projections[month] = mrr
        
        return projections
    
    def evaluate_portfolio_fit(
        self, 
        opportunity: BusinessOpportunity,
        existing_portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate how well opportunity fits with existing portfolio."""
        fit_scores = {
            'market_overlap': 0.0,
            'technology_synergy': 0.0,
            'customer_synergy': 0.0,
            'resource_efficiency': 0.0,
            'strategic_value': 0.0
        }
        
        # Check for market overlap (avoid too much)
        existing_markets = set()
        for project in existing_portfolio.values():
            if 'market' in project:
                existing_markets.add(project['market'])
        
        # Some overlap is good, too much is bad
        if opportunity.target_audience['primary'] not in existing_markets:
            fit_scores['market_overlap'] = 0.8  # New market
        else:
            fit_scores['market_overlap'] = 0.4  # Existing market
        
        # Strategic value based on monetization potential
        if opportunity.monetization_model.ltv_cac_ratio > 5:
            fit_scores['strategic_value'] = 1.0
        elif opportunity.monetization_model.ltv_cac_ratio > 3:
            fit_scores['strategic_value'] = 0.7
        else:
            fit_scores['strategic_value'] = 0.4
        
        total_fit = sum(fit_scores.values()) / len(fit_scores)
        
        return {
            'total_fit_score': total_fit,
            'fit_details': fit_scores,
            'recommendation': 'pursue' if total_fit > 0.6 else 'reconsider'
        }