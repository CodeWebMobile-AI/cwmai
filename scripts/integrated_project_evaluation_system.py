"""
Integrated Project Evaluation System

This module provides a unified interface for evaluating all project opportunities,
whether they are new projects or existing ones that need monetization. It combines
all decision systems to provide comprehensive recommendations.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path

from scripts.ai_brain import IntelligentAIBrain
from scripts.business_market_research_engine import (
    BusinessMarketResearchEngine,
    BusinessOpportunity,
    MarketValidation,
    MonetizationModel
)
from scripts.enhanced_project_opportunity import EnhancedProjectOpportunity
from scripts.existing_project_monetization_analyzer import ExistingProjectMonetizationAnalyzer
from scripts.portfolio_intelligence import PortfolioIntelligence, PortfolioInsights
from scripts.dynamic_charter import DynamicCharter
from scripts.market_research_engine import MarketResearchEngine, ProjectOpportunity


@dataclass
class EvaluationCriteria:
    """Unified evaluation criteria for all opportunities."""
    market_validation_score: float  # 0.0 to 1.0
    monetization_clarity: float  # 0.0 to 1.0
    implementation_effort: float  # 0.0 to 1.0 (lower is better)
    portfolio_fit: float  # 0.0 to 1.0
    strategic_alignment: float  # 0.0 to 1.0
    time_to_revenue: int  # days
    revenue_potential: float  # estimated MRR at 6 months
    risk_level: str  # 'low', 'medium', 'high'
    recommendation_score: float  # 0.0 to 1.0 (weighted composite)


@dataclass
class UnifiedOpportunity:
    """Unified representation of any opportunity (new or existing project)."""
    id: str
    type: str  # 'new_project' or 'existing_project'
    name: str
    description: str
    current_state: str  # 'concept', 'development', 'production', 'mature'
    evaluation: EvaluationCriteria
    monetization_model: Optional[MonetizationModel]
    market_validation: Optional[MarketValidation]
    implementation_plan: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report with recommendations."""
    timestamp: datetime
    portfolio_summary: Dict[str, Any]
    all_opportunities: List[UnifiedOpportunity]
    recommendations: List[Dict[str, Any]]
    strategic_insights: Dict[str, Any]
    action_plan: List[Dict[str, Any]]
    executive_summary: str


class IntegratedProjectEvaluationSystem:
    """Unified system for evaluating all project opportunities."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize the integrated evaluation system."""
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Initialize all subsystems
        self.business_research = BusinessMarketResearchEngine(ai_brain)
        self.market_research = MarketResearchEngine(ai_brain)
        self.monetization_analyzer = ExistingProjectMonetizationAnalyzer(ai_brain)
        self.portfolio_intelligence = PortfolioIntelligence(ai_brain)
        self.dynamic_charter = DynamicCharter(ai_brain)
        
        # Evaluation weights
        self.criteria_weights = {
            'market_validation': 0.25,
            'monetization_clarity': 0.20,
            'implementation_effort': 0.15,
            'portfolio_fit': 0.15,
            'strategic_alignment': 0.15,
            'time_to_revenue': 0.10
        }
    
    async def evaluate_all_opportunities(
        self,
        existing_portfolio: Dict[str, Any],
        max_new_opportunities: int = 5,
        include_existing: bool = True
    ) -> EvaluationReport:
        """Main entry point - evaluate all opportunities comprehensively.
        
        Args:
            existing_portfolio: Current portfolio data
            max_new_opportunities: Maximum new opportunities to evaluate
            include_existing: Whether to analyze existing projects for monetization
            
        Returns:
            Comprehensive evaluation report
        """
        self.logger.info("🔍 Starting comprehensive opportunity evaluation...")
        
        # 1. Generate dynamic charter for strategic alignment
        charter_context = {
            'projects': list(existing_portfolio.keys()),
            'capabilities': ['market_research', 'monetization_analysis', 'portfolio_intelligence'],
            'recent_outcomes': [],  # Could be loaded from history
            'market_trends': []
        }
        charter = await self.dynamic_charter.generate_charter(charter_context)
        
        # 2. Analyze current portfolio
        portfolio_insights = await self.portfolio_intelligence.analyze_portfolio(
            existing_portfolio, 
            force_refresh=True
        )
        
        # 3. Discover new opportunities
        new_opportunities = await self._discover_new_opportunities(
            existing_portfolio,
            portfolio_insights,
            max_new_opportunities
        )
        
        # 4. Analyze existing projects for monetization
        existing_opportunities = []
        if include_existing:
            existing_opportunities = await self._analyze_existing_projects(
                existing_portfolio,
                portfolio_insights
            )
        
        # 5. Unify and evaluate all opportunities
        all_opportunities = await self._unify_opportunities(
            new_opportunities,
            existing_opportunities,
            charter,
            portfolio_insights
        )
        
        # 6. Score and rank opportunities
        scored_opportunities = await self._score_opportunities(
            all_opportunities,
            charter,
            portfolio_insights
        )
        
        # 7. Generate recommendations
        recommendations = await self._generate_recommendations(
            scored_opportunities,
            portfolio_insights,
            charter
        )
        
        # 8. Create action plan
        action_plan = await self._create_action_plan(
            recommendations,
            scored_opportunities
        )
        
        # 9. Generate executive summary
        executive_summary = await self._generate_executive_summary(
            scored_opportunities,
            recommendations,
            portfolio_insights
        )
        
        # Create final report
        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc),
            portfolio_summary={
                'total_projects': portfolio_insights.total_projects,
                'technology_coverage': dict(portfolio_insights.technology_coverage),
                'domain_coverage': dict(portfolio_insights.domain_coverage),
                'maturity_distribution': dict(portfolio_insights.maturity_distribution),
                'innovation_leaders': portfolio_insights.innovation_leaders,
                'maintenance_concerns': portfolio_insights.maintenance_concerns
            },
            all_opportunities=scored_opportunities,
            recommendations=recommendations,
            strategic_insights={
                'gaps': portfolio_insights.strategic_gaps,
                'growth_areas': portfolio_insights.growth_recommendations,
                'synergies': portfolio_insights.integration_opportunities
            },
            action_plan=action_plan,
            executive_summary=executive_summary
        )
        
        return report
    
    async def _discover_new_opportunities(
        self,
        existing_portfolio: Dict[str, Any],
        portfolio_insights: PortfolioInsights,
        max_opportunities: int
    ) -> List[BusinessOpportunity]:
        """Discover new project opportunities."""
        opportunities = []
        
        # Get opportunities from business research
        business_opps = await self.business_research.discover_business_opportunities(
            existing_portfolio,
            max_opportunities=max_opportunities
        )
        opportunities.extend(business_opps)
        
        # Get opportunities from technical market research
        tech_opps = await self.market_research.generate_project_opportunities(
            existing_portfolio,
            max_opportunities=max_opportunities
        )
        
        # Convert technical opportunities to business opportunities
        for tech_opp in tech_opps:
            business_opp = await self._convert_to_business_opportunity(tech_opp)
            if business_opp:
                opportunities.append(business_opp)
        
        return opportunities[:max_opportunities]
    
    async def _analyze_existing_projects(
        self,
        existing_portfolio: Dict[str, Any],
        portfolio_insights: PortfolioInsights
    ) -> List[Dict[str, Any]]:
        """Analyze existing projects for monetization opportunities."""
        existing_opportunities = []
        
        # Get all projects lacking monetization
        all_analyses = await self.monetization_analyzer.analyze_all_projects()
        
        for project_name, analysis in all_analyses.items():
            if not analysis.get('exists'):  # No existing monetization strategy
                # Get project profile from portfolio intelligence
                profile = self.portfolio_intelligence.project_profiles.get(project_name)
                
                if profile and profile.growth_potential > 0.5:  # Focus on high-potential projects
                    existing_opportunities.append({
                        'project_name': project_name,
                        'analysis': analysis,
                        'profile': profile,
                        'type': 'existing_project'
                    })
        
        return existing_opportunities
    
    async def _unify_opportunities(
        self,
        new_opportunities: List[BusinessOpportunity],
        existing_opportunities: List[Dict[str, Any]],
        charter: Dict[str, Any],
        portfolio_insights: PortfolioInsights
    ) -> List[UnifiedOpportunity]:
        """Convert all opportunities to unified format."""
        unified = []
        
        # Convert new opportunities
        for opp in new_opportunities:
            unified_opp = UnifiedOpportunity(
                id=opp.id,
                type='new_project',
                name=opp.name,
                description=opp.solution_description,
                current_state='concept',
                evaluation=EvaluationCriteria(
                    market_validation_score=opp.market_validation.validation_score,
                    monetization_clarity=self._calculate_monetization_clarity(opp.monetization_model),
                    implementation_effort=self._effort_score(opp.implementation_complexity),
                    portfolio_fit=0.0,  # To be calculated
                    strategic_alignment=0.0,  # To be calculated
                    time_to_revenue=opp.time_to_revenue,
                    revenue_potential=opp.revenue_projections.get(6, 0),
                    risk_level=self._assess_risk(opp),
                    recommendation_score=0.0  # To be calculated
                ),
                monetization_model=opp.monetization_model,
                market_validation=opp.market_validation,
                implementation_plan={
                    'complexity': opp.implementation_complexity,
                    'time_to_market': opp.time_to_market,
                    'initial_investment': self._estimate_investment(opp)
                },
                metadata={'original': opp}
            )
            unified.append(unified_opp)
        
        # Convert existing opportunities
        for existing in existing_opportunities:
            project_name = existing['project_name']
            analysis = existing['analysis']
            profile = existing['profile']
            
            # Create monetization model from analysis
            monetization = self._extract_monetization_model(analysis)
            
            unified_opp = UnifiedOpportunity(
                id=f"existing_{project_name}",
                type='existing_project',
                name=project_name,
                description=profile.primary_purpose,
                current_state=profile.maturity_level,
                evaluation=EvaluationCriteria(
                    market_validation_score=profile.market_fit_score,
                    monetization_clarity=0.8 if monetization else 0.2,
                    implementation_effort=0.3,  # Lower effort for existing projects
                    portfolio_fit=1.0,  # Already in portfolio
                    strategic_alignment=0.0,  # To be calculated
                    time_to_revenue=30,  # Faster for existing projects
                    revenue_potential=self._estimate_existing_revenue(analysis),
                    risk_level='low',  # Lower risk for existing projects
                    recommendation_score=0.0  # To be calculated
                ),
                monetization_model=monetization,
                market_validation=None,  # Could be generated if needed
                implementation_plan={
                    'complexity': 'low',
                    'time_to_market': 30,
                    'initial_investment': 5000  # Lower for adding monetization
                },
                metadata={'profile': profile, 'analysis': analysis}
            )
            unified.append(unified_opp)
        
        return unified
    
    async def _score_opportunities(
        self,
        opportunities: List[UnifiedOpportunity],
        charter: Dict[str, Any],
        portfolio_insights: PortfolioInsights
    ) -> List[UnifiedOpportunity]:
        """Score all opportunities based on unified criteria."""
        for opp in opportunities:
            # Calculate portfolio fit
            opp.evaluation.portfolio_fit = await self._calculate_portfolio_fit(
                opp, 
                portfolio_insights
            )
            
            # Calculate strategic alignment
            opp.evaluation.strategic_alignment = await self._calculate_strategic_alignment(
                opp,
                charter
            )
            
            # Calculate weighted recommendation score
            opp.evaluation.recommendation_score = self._calculate_recommendation_score(
                opp.evaluation
            )
        
        # Sort by recommendation score
        opportunities.sort(key=lambda x: x.evaluation.recommendation_score, reverse=True)
        
        return opportunities
    
    async def _calculate_portfolio_fit(
        self,
        opportunity: UnifiedOpportunity,
        portfolio_insights: PortfolioInsights
    ) -> float:
        """Calculate how well opportunity fits with existing portfolio."""
        if opportunity.type == 'existing_project':
            return 1.0  # Already in portfolio
        
        fit_score = 0.0
        
        # Check if it fills a strategic gap
        for gap in portfolio_insights.strategic_gaps:
            if gap.get('gap_type') in opportunity.description.lower():
                fit_score += 0.3
        
        # Check for synergies with existing projects
        if portfolio_insights.integration_opportunities:
            fit_score += 0.2
        
        # Check technology alignment
        if opportunity.metadata.get('original'):
            tech_overlap = set(opportunity.metadata['original'].target_audience.get('technologies', []))
            existing_tech = set()
            for tech_list in portfolio_insights.technology_coverage.keys():
                existing_tech.add(tech_list.lower())
            
            if tech_overlap & existing_tech:
                fit_score += 0.2
        
        # Diversity bonus - different domain than majority
        domain_diversity = len(portfolio_insights.domain_coverage) > 3
        if not domain_diversity:
            fit_score += 0.3
        
        return min(fit_score, 1.0)
    
    async def _calculate_strategic_alignment(
        self,
        opportunity: UnifiedOpportunity,
        charter: Dict[str, Any]
    ) -> float:
        """Calculate alignment with dynamic charter."""
        # Validate against charter
        validation = await self.dynamic_charter.validate_against_charter({
            'type': 'project_opportunity',
            'name': opportunity.name,
            'description': opportunity.description,
            'purpose': opportunity.metadata.get('original', {}).problem_statement if opportunity.metadata.get('original') else ''
        })
        
        if validation.get('aligned'):
            base_score = 0.8
        else:
            base_score = 0.3
        
        # Bonus for matching core objectives
        objectives = charter.get('CORE_OBJECTIVES', [])
        for obj in objectives:
            if any(keyword in opportunity.description.lower() for keyword in obj.lower().split()):
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _calculate_recommendation_score(self, evaluation: EvaluationCriteria) -> float:
        """Calculate weighted recommendation score."""
        score = 0.0
        
        # Apply weights to each criterion
        score += evaluation.market_validation_score * self.criteria_weights['market_validation']
        score += evaluation.monetization_clarity * self.criteria_weights['monetization_clarity']
        score += (1.0 - evaluation.implementation_effort) * self.criteria_weights['implementation_effort']
        score += evaluation.portfolio_fit * self.criteria_weights['portfolio_fit']
        score += evaluation.strategic_alignment * self.criteria_weights['strategic_alignment']
        
        # Time to revenue factor (normalize to 0-1, faster is better)
        time_factor = max(0, 1.0 - (evaluation.time_to_revenue / 365))  # Within a year
        score += time_factor * self.criteria_weights['time_to_revenue']
        
        # Risk adjustment
        risk_multiplier = {
            'low': 1.0,
            'medium': 0.85,
            'high': 0.7
        }
        score *= risk_multiplier.get(evaluation.risk_level, 0.85)
        
        return round(score, 3)
    
    async def _generate_recommendations(
        self,
        opportunities: List[UnifiedOpportunity],
        portfolio_insights: PortfolioInsights,
        charter: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Top opportunities by score
        top_new = [o for o in opportunities if o.type == 'new_project'][:3]
        top_existing = [o for o in opportunities if o.type == 'existing_project'][:3]
        
        # Immediate action items
        if top_existing and top_existing[0].evaluation.recommendation_score > 0.7:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'monetize_existing',
                'title': f"Monetize {top_existing[0].name}",
                'rationale': f"High-potential existing project with clear path to revenue. "
                           f"Score: {top_existing[0].evaluation.recommendation_score}",
                'expected_revenue': top_existing[0].evaluation.revenue_potential,
                'time_to_revenue': top_existing[0].evaluation.time_to_revenue,
                'action_items': [
                    "Implement monetization strategy",
                    "Set up payment processing",
                    "Create pricing page",
                    "Launch to existing users"
                ]
            })
        
        if top_new and top_new[0].evaluation.recommendation_score > 0.75:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'create_new',
                'title': f"Create {top_new[0].name}",
                'rationale': f"Strong market validation and clear monetization path. "
                           f"Score: {top_new[0].evaluation.recommendation_score}",
                'expected_revenue': top_new[0].evaluation.revenue_potential,
                'time_to_market': top_new[0].implementation_plan['time_to_market'],
                'action_items': [
                    "Fork Laravel React starter kit",
                    "Implement core features",
                    "Set up monetization from day 1",
                    "Launch MVP within 30 days"
                ]
            })
        
        # Strategic recommendations
        if len(portfolio_insights.maintenance_concerns) > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'portfolio_health',
                'title': "Address Technical Debt",
                'rationale': f"{len(portfolio_insights.maintenance_concerns)} projects need attention",
                'projects': portfolio_insights.maintenance_concerns,
                'action_items': [
                    "Prioritize refactoring high-burden projects",
                    "Consider sunsetting low-value projects",
                    "Automate maintenance where possible"
                ]
            })
        
        # Portfolio balance recommendations
        gap_domains = [g.get('description', '') for g in portfolio_insights.strategic_gaps[:2]]
        if gap_domains:
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'portfolio_balance',
                'title': "Fill Strategic Gaps",
                'rationale': "Portfolio missing key market segments",
                'gaps': gap_domains,
                'suggested_projects': [o.name for o in top_new if o.evaluation.portfolio_fit > 0.6]
            })
        
        return recommendations
    
    async def _create_action_plan(
        self,
        recommendations: List[Dict[str, Any]],
        opportunities: List[UnifiedOpportunity]
    ) -> List[Dict[str, Any]]:
        """Create detailed action plan."""
        action_plan = []
        
        # Week 1-2: Quick wins
        quick_wins = [r for r in recommendations if r.get('type') == 'monetize_existing']
        for win in quick_wins[:1]:  # Focus on one at a time
            action_plan.append({
                'phase': 'Week 1-2',
                'action': win['title'],
                'type': 'quick_win',
                'tasks': win['action_items'],
                'expected_outcome': f"${win['expected_revenue']}/month MRR",
                'resources_needed': ['Developer time', 'Stripe account', 'Marketing copy']
            })
        
        # Week 3-4: New project initiation
        new_projects = [r for r in recommendations if r.get('type') == 'create_new']
        for proj in new_projects[:1]:
            action_plan.append({
                'phase': 'Week 3-4',
                'action': proj['title'],
                'type': 'new_initiative',
                'tasks': proj['action_items'],
                'expected_outcome': f"MVP launched, targeting ${proj['expected_revenue']}/month",
                'resources_needed': ['Full development focus', 'Domain & hosting', 'Initial marketing budget']
            })
        
        # Month 2: Scale and optimize
        action_plan.append({
            'phase': 'Month 2',
            'action': 'Scale and Optimize',
            'type': 'growth',
            'tasks': [
                "Analyze user feedback and metrics",
                "Optimize conversion funnels",
                "Implement referral programs",
                "Begin content marketing"
            ],
            'expected_outcome': "2x growth in MRR",
            'resources_needed': ['Analytics tools', 'Marketing automation', 'Content creation']
        })
        
        # Month 3: Portfolio optimization
        if any(r.get('type') == 'portfolio_health' for r in recommendations):
            action_plan.append({
                'phase': 'Month 3',
                'action': 'Portfolio Optimization',
                'type': 'maintenance',
                'tasks': [
                    "Refactor high-maintenance projects",
                    "Implement monitoring and alerting",
                    "Document all monetization flows",
                    "Create playbooks for common issues"
                ],
                'expected_outcome': "Reduced maintenance burden by 50%",
                'resources_needed': ['DevOps tools', 'Documentation time']
            })
        
        return action_plan
    
    async def _generate_executive_summary(
        self,
        opportunities: List[UnifiedOpportunity],
        recommendations: List[Dict[str, Any]],
        portfolio_insights: PortfolioInsights
    ) -> str:
        """Generate executive summary of the evaluation."""
        top_opp = opportunities[0] if opportunities else None
        total_revenue_potential = sum(o.evaluation.revenue_potential for o in opportunities[:5])
        
        summary = f"""# Executive Summary

## Current State
- Portfolio Size: {portfolio_insights.total_projects} projects
- Innovation Leaders: {', '.join(portfolio_insights.innovation_leaders[:3])}
- Projects Needing Monetization: {len([o for o in opportunities if o.type == 'existing_project'])}

## Key Findings
1. **Immediate Revenue Opportunity**: {top_opp.name if top_opp else 'None identified'}
   - Type: {top_opp.type.replace('_', ' ').title() if top_opp else 'N/A'}
   - Revenue Potential: ${top_opp.evaluation.revenue_potential:,.0f}/month if top_opp else 'N/A'
   - Time to Revenue: {top_opp.evaluation.time_to_revenue} days if top_opp else 'N/A'
   - Recommendation Score: {top_opp.evaluation.recommendation_score:.2f} if top_opp else 'N/A'

2. **Total Revenue Potential** (Top 5 Opportunities): ${total_revenue_potential:,.0f}/month

3. **Strategic Gaps Identified**: {len(portfolio_insights.strategic_gaps)}

## Recommended Actions
"""
        
        for i, rec in enumerate(recommendations[:3], 1):
            summary += f"\n{i}. **{rec['title']}** (Priority: {rec['priority']})\n"
            summary += f"   - Rationale: {rec['rationale']}\n"
            if 'expected_revenue' in rec:
                summary += f"   - Expected Revenue: ${rec['expected_revenue']:,.0f}/month\n"
        
        summary += f"""
## Success Metrics
- 30-day target: Launch monetization for 1 existing project
- 60-day target: ${total_revenue_potential * 0.3:,.0f}/month MRR
- 90-day target: ${total_revenue_potential * 0.5:,.0f}/month MRR

## Risk Mitigation
- Focus on proven monetization models
- Start with existing high-potential projects
- Validate with small user segments first
- Maintain portfolio balance
"""
        
        return summary
    
    def _calculate_monetization_clarity(self, model: MonetizationModel) -> float:
        """Calculate how clear and actionable the monetization model is."""
        clarity_score = 0.0
        
        # Check for complete pricing tiers
        if len(model.pricing_tiers) >= 3:
            clarity_score += 0.3
        
        # Check for multiple revenue streams
        if len(model.revenue_streams) >= 2:
            clarity_score += 0.2
        
        # Check for reasonable unit economics
        if model.ltv_cac_ratio > 3:
            clarity_score += 0.2
        
        # Check for 24/7 mechanisms
        if len(model.twenty_four_seven_mechanisms) >= 2:
            clarity_score += 0.3
        
        return min(clarity_score, 1.0)
    
    def _effort_score(self, complexity: str) -> float:
        """Convert complexity to effort score (0-1, lower is better)."""
        return {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8
        }.get(complexity, 0.5)
    
    def _assess_risk(self, opportunity: BusinessOpportunity) -> str:
        """Assess risk level of opportunity."""
        risk_factors = 0
        
        if opportunity.implementation_complexity == 'high':
            risk_factors += 2
        elif opportunity.implementation_complexity == 'medium':
            risk_factors += 1
        
        if opportunity.market_validation.validation_score < 0.5:
            risk_factors += 2
        
        if opportunity.time_to_market > 90:
            risk_factors += 1
        
        if risk_factors >= 3:
            return 'high'
        elif risk_factors >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_investment(self, opportunity: BusinessOpportunity) -> float:
        """Estimate initial investment needed."""
        base_investment = {
            'low': 5000,
            'medium': 15000,
            'high': 50000
        }
        
        return base_investment.get(opportunity.implementation_complexity, 15000)
    
    def _extract_monetization_model(self, analysis: Dict[str, Any]) -> Optional[MonetizationModel]:
        """Extract monetization model from analysis."""
        if not analysis or analysis.get('exists'):
            return None
        
        exec_summary = analysis.get('executive_summary', {})
        pricing = analysis.get('pricing_tiers', [])
        financial = analysis.get('financial_projections', {})
        unit_econ = financial.get('unit_economics', {})
        
        return MonetizationModel(
            model_type=exec_summary.get('revenue_model', 'subscription'),
            pricing_tiers=pricing,
            revenue_streams=analysis.get('revenue_streams', {}),
            target_arpu=29,  # Default
            target_cac=unit_econ.get('cac', 50),
            target_ltv=unit_econ.get('ltv', 1000),
            ltv_cac_ratio=unit_econ.get('ltv_cac_ratio', 20),
            payback_months=unit_econ.get('payback_months', 2),
            gross_margin_target=0.8,
            twenty_four_seven_mechanisms=['Automated billing', 'Self-service platform']
        )
    
    def _estimate_existing_revenue(self, analysis: Dict[str, Any]) -> float:
        """Estimate revenue potential for existing project."""
        if analysis.get('executive_summary'):
            return analysis['executive_summary'].get('target_mrr', 5000)
        return 5000  # Conservative default
    
    async def generate_decision_report(
        self,
        evaluation_report: EvaluationReport,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a comprehensive decision report."""
        report_content = f"""# Project Evaluation Report
Generated: {evaluation_report.timestamp.isoformat()}

{evaluation_report.executive_summary}

## Detailed Opportunity Analysis

### Top Opportunities by Score
"""
        
        for i, opp in enumerate(evaluation_report.all_opportunities[:10], 1):
            report_content += f"""
#### {i}. {opp.name} ({opp.type.replace('_', ' ').title()})
- **Score**: {opp.evaluation.recommendation_score:.3f}
- **Market Validation**: {opp.evaluation.market_validation_score:.2f}
- **Monetization Clarity**: {opp.evaluation.monetization_clarity:.2f}
- **Time to Revenue**: {opp.evaluation.time_to_revenue} days
- **Revenue Potential**: ${opp.evaluation.revenue_potential:,.0f}/month
- **Risk Level**: {opp.evaluation.risk_level}
- **Description**: {opp.description}
"""
        
        report_content += """
## Portfolio Analysis
"""
        
        for key, value in evaluation_report.portfolio_summary.items():
            if isinstance(value, dict):
                report_content += f"\n### {key.replace('_', ' ').title()}\n"
                for k, v in value.items():
                    report_content += f"- {k}: {v}\n"
            elif isinstance(value, list):
                report_content += f"\n### {key.replace('_', ' ').title()}\n"
                for item in value:
                    report_content += f"- {item}\n"
            else:
                report_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        report_content += """
## Strategic Insights
"""
        
        if evaluation_report.strategic_insights.get('gaps'):
            report_content += "\n### Strategic Gaps\n"
            for gap in evaluation_report.strategic_insights['gaps']:
                report_content += f"- **{gap.get('gap_type', 'Unknown')}**: {gap.get('description', 'N/A')}\n"
        
        report_content += """
## Action Plan
"""
        
        for phase in evaluation_report.action_plan:
            report_content += f"\n### {phase['phase']}: {phase['action']}\n"
            report_content += f"**Type**: {phase['type']}\n\n"
            report_content += "**Tasks**:\n"
            for task in phase['tasks']:
                report_content += f"- {task}\n"
            report_content += f"\n**Expected Outcome**: {phase['expected_outcome']}\n"
            report_content += f"**Resources Needed**: {', '.join(phase['resources_needed'])}\n"
        
        # Save if path provided
        if output_path:
            output_path.write_text(report_content)
            self.logger.info(f"Report saved to {output_path}")
        
        return report_content
    
    def score_new_vs_existing(
        self,
        opportunities: List[UnifiedOpportunity]
    ) -> Dict[str, List[UnifiedOpportunity]]:
        """Compare and categorize new vs existing opportunities."""
        categorized = {
            'immediate_action': [],  # Score > 0.8
            'high_priority': [],     # Score 0.6-0.8
            'medium_priority': [],   # Score 0.4-0.6
            'low_priority': [],      # Score < 0.4
            'new_projects': [],
            'existing_projects': []
        }
        
        for opp in opportunities:
            # Categorize by type
            if opp.type == 'new_project':
                categorized['new_projects'].append(opp)
            else:
                categorized['existing_projects'].append(opp)
            
            # Categorize by priority
            score = opp.evaluation.recommendation_score
            if score > 0.8:
                categorized['immediate_action'].append(opp)
            elif score > 0.6:
                categorized['high_priority'].append(opp)
            elif score > 0.4:
                categorized['medium_priority'].append(opp)
            else:
                categorized['low_priority'].append(opp)
        
        return categorized
    
    async def create_decision_report(
        self,
        evaluation_report: EvaluationReport
    ) -> Dict[str, Any]:
        """Create executive decision summary."""
        categorized = self.score_new_vs_existing(evaluation_report.all_opportunities)
        
        decision_summary = {
            'decision': 'CREATE_NEW' if len(categorized['immediate_action']) > 0 and categorized['immediate_action'][0].type == 'new_project' else 'MONETIZE_EXISTING',
            'primary_opportunity': categorized['immediate_action'][0].name if categorized['immediate_action'] else 'None',
            'rationale': evaluation_report.recommendations[0]['rationale'] if evaluation_report.recommendations else 'No clear opportunity',
            'expected_outcome': {
                '30_days': f"${sum(o.evaluation.revenue_potential * 0.1 for o in categorized['immediate_action']):,.0f}/month",
                '60_days': f"${sum(o.evaluation.revenue_potential * 0.3 for o in categorized['immediate_action']):,.0f}/month",
                '90_days': f"${sum(o.evaluation.revenue_potential * 0.5 for o in categorized['immediate_action']):,.0f}/month"
            },
            'action_items': evaluation_report.action_plan[0]['tasks'] if evaluation_report.action_plan else [],
            'investment_required': sum(o.implementation_plan.get('initial_investment', 0) for o in categorized['immediate_action']),
            'risk_assessment': categorized['immediate_action'][0].evaluation.risk_level if categorized['immediate_action'] else 'unknown'
        }
        
        return decision_summary