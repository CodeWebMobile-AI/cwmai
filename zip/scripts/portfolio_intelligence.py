"""
Portfolio Intelligence System

Provides deep analysis of existing projects to understand capabilities,
gaps, and opportunities for strategic growth.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json

from scripts.ai_brain import IntelligentAIBrain
from scripts.repository_exclusion import RepositoryExclusion


@dataclass
class ProjectProfile:
    """Detailed profile of a project in the portfolio."""
    repository_name: str
    primary_purpose: str
    problem_domain: str
    target_audience: str
    technology_stack: List[str]
    architecture_pattern: str
    maturity_level: str  # 'prototype', 'mvp', 'production', 'mature'
    innovation_score: float  # 0.0 to 1.0
    market_fit_score: float  # 0.0 to 1.0
    maintenance_burden: float  # 0.0 to 1.0 (higher = more burden)
    growth_potential: float  # 0.0 to 1.0
    key_features: List[str]
    integration_points: List[str]
    improvement_areas: List[str]


@dataclass
class PortfolioInsights:
    """Insights from portfolio analysis."""
    total_projects: int
    technology_coverage: Dict[str, int]  # tech -> count
    domain_coverage: Dict[str, int]  # domain -> count
    market_coverage: Dict[str, int]  # market -> count
    maturity_distribution: Dict[str, int]  # maturity level -> count
    innovation_leaders: List[str]  # top innovative projects
    maintenance_concerns: List[str]  # projects needing attention
    integration_opportunities: List[Tuple[str, str]]  # project pairs
    strategic_gaps: List[Dict[str, Any]]  # identified gaps
    growth_recommendations: List[Dict[str, Any]]  # growth opportunities


class PortfolioIntelligence:
    """Intelligent portfolio analysis system."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize the portfolio intelligence system.
        
        Args:
            ai_brain: AI brain for intelligent analysis
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Cache for project profiles
        self.project_profiles: Dict[str, ProjectProfile] = {}
        self.last_analysis_time: Optional[datetime] = None
        
    async def analyze_portfolio(
        self,
        portfolio: Dict[str, Any],
        force_refresh: bool = False
    ) -> PortfolioInsights:
        """Perform comprehensive portfolio analysis.
        
        Args:
            portfolio: Dictionary of repository data
            force_refresh: Force re-analysis even if cached
            
        Returns:
            Portfolio insights
        """
        self.logger.info("🔬 Analyzing portfolio intelligence...")
        
        # Filter excluded repositories
        filtered_portfolio = RepositoryExclusion.filter_excluded_repos_dict(portfolio)
        
        # Analyze each project
        if force_refresh or not self.project_profiles:
            await self._analyze_all_projects(filtered_portfolio)
        
        # Generate insights
        insights = self._generate_insights(filtered_portfolio)
        
        # Identify strategic opportunities
        insights.strategic_gaps = await self._identify_strategic_gaps(insights)
        insights.growth_recommendations = await self._generate_growth_recommendations(insights)
        
        self.last_analysis_time = datetime.now(timezone.utc)
        
        return insights
    
    async def _analyze_all_projects(self, portfolio: Dict[str, Any]) -> None:
        """Analyze all projects in the portfolio."""
        self.project_profiles.clear()
        
        # Batch analyze projects for efficiency
        batch_size = 5
        project_items = list(portfolio.items())
        
        for i in range(0, len(project_items), batch_size):
            batch = project_items[i:i + batch_size]
            
            # Analyze batch concurrently
            tasks = [
                self._analyze_single_project(repo_name, repo_data)
                for repo_name, repo_data in batch
            ]
            
            profiles = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store successful profiles
            for j, profile in enumerate(profiles):
                if isinstance(profile, ProjectProfile):
                    self.project_profiles[batch[j][0]] = profile
                else:
                    self.logger.error(f"Failed to analyze {batch[j][0]}: {profile}")
    
    async def _analyze_single_project(
        self,
        repo_name: str,
        repo_data: Dict[str, Any]
    ) -> ProjectProfile:
        """Analyze a single project in depth."""
        try:
            # Gather basic information
            description = repo_data.get('description', '')
            language = repo_data.get('language', '')
            topics = repo_data.get('topics', [])
            readme_content = repo_data.get('readme_content', '')
            recent_activity = repo_data.get('recent_activity', {})
            
            # Use AI for deep analysis
            prompt = f"""
            Analyze this software project in detail:
            
            Project: {repo_name}
            Description: {description}
            Primary Language: {language}
            Topics/Tags: {', '.join(topics)}
            Recent Commits: {recent_activity.get('recent_commits', 0)}
            README Preview: {readme_content[:500] if readme_content else 'No README'}
            
            Provide a comprehensive analysis:
            
            1. primary_purpose: Main purpose in one sentence
            2. problem_domain: Specific domain (e.g., "e-commerce analytics", "developer productivity")
            3. target_audience: Primary users (e.g., "SaaS developers", "small business owners")
            4. technology_stack: All key technologies used (not just main language)
            5. architecture_pattern: Pattern used (e.g., "MVC", "microservices", "event-driven")
            6. maturity_level: Current state - prototype/mvp/production/mature
            7. innovation_score: 0.0-1.0 (how innovative/unique is this?)
            8. market_fit_score: 0.0-1.0 (how well does it meet market needs?)
            9. maintenance_burden: 0.0-1.0 (technical debt, complexity)
            10. growth_potential: 0.0-1.0 (room for expansion/improvement)
            11. key_features: List 3-5 main features
            12. integration_points: What it could integrate with
            13. improvement_areas: Top 3 areas for improvement
            
            Return as JSON object with these exact fields.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                analysis = self._parse_json(response.get('result', ''))
                
                if analysis:
                    # Ensure all fields have proper types
                    return ProjectProfile(
                        repository_name=repo_name,
                        primary_purpose=analysis.get('primary_purpose', 'Unknown purpose'),
                        problem_domain=analysis.get('problem_domain', 'General'),
                        target_audience=analysis.get('target_audience', 'General users'),
                        technology_stack=analysis.get('technology_stack', [language] if language else []),
                        architecture_pattern=analysis.get('architecture_pattern', 'Unknown'),
                        maturity_level=analysis.get('maturity_level', 'prototype'),
                        innovation_score=float(analysis.get('innovation_score', 0.5)),
                        market_fit_score=float(analysis.get('market_fit_score', 0.5)),
                        maintenance_burden=float(analysis.get('maintenance_burden', 0.5)),
                        growth_potential=float(analysis.get('growth_potential', 0.5)),
                        key_features=analysis.get('key_features', []),
                        integration_points=analysis.get('integration_points', []),
                        improvement_areas=analysis.get('improvement_areas', [])
                    )
            
            # Fallback to basic analysis
            return self._create_basic_profile(repo_name, repo_data)
            
        except Exception as e:
            self.logger.error(f"Error analyzing project {repo_name}: {e}")
            return self._create_basic_profile(repo_name, repo_data)
    
    def _create_basic_profile(self, repo_name: str, repo_data: Dict[str, Any]) -> ProjectProfile:
        """Create a basic profile when AI analysis fails."""
        return ProjectProfile(
            repository_name=repo_name,
            primary_purpose=repo_data.get('description', 'No description')[:100],
            problem_domain='Unknown',
            target_audience='General users',
            technology_stack=[repo_data.get('language', 'Unknown')],
            architecture_pattern='Unknown',
            maturity_level='prototype',
            innovation_score=0.5,
            market_fit_score=0.5,
            maintenance_burden=0.5,
            growth_potential=0.5,
            key_features=[],
            integration_points=[],
            improvement_areas=['Needs analysis']
        )
    
    def _generate_insights(self, portfolio: Dict[str, Any]) -> PortfolioInsights:
        """Generate insights from analyzed projects."""
        insights = PortfolioInsights(
            total_projects=len(self.project_profiles),
            technology_coverage={},
            domain_coverage={},
            market_coverage={},
            maturity_distribution={},
            innovation_leaders=[],
            maintenance_concerns=[],
            integration_opportunities=[],
            strategic_gaps=[],
            growth_recommendations=[]
        )
        
        # Aggregate data from profiles
        innovation_scores = []
        maintenance_scores = []
        
        for profile in self.project_profiles.values():
            # Technology coverage
            for tech in profile.technology_stack:
                insights.technology_coverage[tech] = insights.technology_coverage.get(tech, 0) + 1
            
            # Domain coverage
            insights.domain_coverage[profile.problem_domain] = \
                insights.domain_coverage.get(profile.problem_domain, 0) + 1
            
            # Market coverage
            insights.market_coverage[profile.target_audience] = \
                insights.market_coverage.get(profile.target_audience, 0) + 1
            
            # Maturity distribution
            insights.maturity_distribution[profile.maturity_level] = \
                insights.maturity_distribution.get(profile.maturity_level, 0) + 1
            
            # Track scores
            innovation_scores.append((profile.repository_name, profile.innovation_score))
            maintenance_scores.append((profile.repository_name, profile.maintenance_burden))
        
        # Identify innovation leaders (top 3)
        innovation_scores.sort(key=lambda x: x[1], reverse=True)
        insights.innovation_leaders = [name for name, _ in innovation_scores[:3]]
        
        # Identify maintenance concerns (burden > 0.7)
        insights.maintenance_concerns = [
            name for name, burden in maintenance_scores if burden > 0.7
        ]
        
        # Find integration opportunities
        insights.integration_opportunities = self._find_integration_opportunities()
        
        return insights
    
    def _find_integration_opportunities(self) -> List[Tuple[str, str]]:
        """Find potential integration opportunities between projects."""
        opportunities = []
        profiles = list(self.project_profiles.values())
        
        for i, profile1 in enumerate(profiles):
            for profile2 in profiles[i+1:]:
                # Check for complementary domains
                if (profile1.problem_domain != profile2.problem_domain and
                    any(point in profile2.integration_points for point in profile1.key_features)):
                    opportunities.append((profile1.repository_name, profile2.repository_name))
                
                # Check for shared technology stack
                shared_tech = set(profile1.technology_stack) & set(profile2.technology_stack)
                if len(shared_tech) >= 2 and profile1.problem_domain != profile2.problem_domain:
                    opportunities.append((profile1.repository_name, profile2.repository_name))
        
        # Remove duplicates and limit to top 5
        unique_opportunities = list(set(opportunities))[:5]
        return unique_opportunities
    
    async def _identify_strategic_gaps(self, insights: PortfolioInsights) -> List[Dict[str, Any]]:
        """Identify strategic gaps in the portfolio."""
        try:
            prompt = f"""
            Analyze this portfolio coverage to identify strategic gaps:
            
            Technology Coverage: {dict(list(insights.technology_coverage.items())[:10])}
            Domain Coverage: {dict(list(insights.domain_coverage.items())[:10])}
            Market Coverage: {dict(list(insights.market_coverage.items())[:10])}
            Maturity Distribution: {insights.maturity_distribution}
            
            Identify 3-5 strategic gaps considering:
            1. Missing technology areas that are in high demand
            2. Underserved market segments
            3. Domain areas with high growth potential
            4. Technology combinations that could create value
            5. Market trends not represented in the portfolio
            
            For each gap provide:
            - gap_type: "technology", "market", "domain", or "integration"
            - description: What's missing and why it matters
            - opportunity_size: "small", "medium", or "large"
            - urgency: "low", "medium", or "high"
            - recommended_action: Specific suggestion
            
            Return as JSON array.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                gaps = self._parse_json_array(response.get('result', ''))
                return gaps
                
        except Exception as e:
            self.logger.error(f"Error identifying strategic gaps: {e}")
        
        return []
    
    async def _generate_growth_recommendations(
        self,
        insights: PortfolioInsights
    ) -> List[Dict[str, Any]]:
        """Generate growth recommendations based on portfolio analysis."""
        try:
            # Get high-potential projects
            high_growth_projects = [
                name for name, profile in self.project_profiles.items()
                if profile.growth_potential > 0.7
            ]
            
            prompt = f"""
            Generate growth recommendations based on this portfolio analysis:
            
            High Growth Potential Projects: {high_growth_projects[:5]}
            Innovation Leaders: {insights.innovation_leaders}
            Integration Opportunities: {insights.integration_opportunities[:3]}
            Maintenance Concerns: {insights.maintenance_concerns}
            
            Provide 3-5 actionable growth recommendations:
            1. How to leverage high-potential projects
            2. Ways to expand successful innovations
            3. Integration strategies for synergy
            4. Addressing maintenance debt
            
            For each recommendation provide:
            - recommendation_type: "expansion", "integration", "optimization", or "innovation"
            - title: Clear action title
            - description: Detailed explanation
            - affected_projects: List of involved projects
            - expected_impact: "low", "medium", or "high"
            - implementation_effort: "low", "medium", or "high"
            
            Return as JSON array.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                recommendations = self._parse_json_array(response.get('result', ''))
                return recommendations
                
        except Exception as e:
            self.logger.error(f"Error generating growth recommendations: {e}")
        
        return []
    
    def _parse_json(self, result: str) -> Optional[Dict[str, Any]]:
        """Parse JSON object from AI response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON: {e}")
        return None
    
    def _parse_json_array(self, result: str) -> List[Dict[str, Any]]:
        """Parse JSON array from AI response."""
        try:
            import re
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON array: {e}")
        return []
    
    def get_project_synergies(self, project_name: str) -> List[Dict[str, Any]]:
        """Get potential synergies for a specific project."""
        synergies = []
        
        if project_name not in self.project_profiles:
            return synergies
        
        profile = self.project_profiles[project_name]
        
        for other_name, other_profile in self.project_profiles.items():
            if other_name == project_name:
                continue
            
            # Calculate synergy score
            synergy_score = 0.0
            reasons = []
            
            # Technology overlap
            tech_overlap = set(profile.technology_stack) & set(other_profile.technology_stack)
            if tech_overlap:
                synergy_score += 0.3
                reasons.append(f"Shared technologies: {', '.join(tech_overlap)}")
            
            # Complementary domains
            if profile.problem_domain != other_profile.problem_domain:
                if profile.target_audience == other_profile.target_audience:
                    synergy_score += 0.4
                    reasons.append("Same target audience, different solutions")
            
            # Integration potential
            if any(feature in other_profile.integration_points for feature in profile.key_features):
                synergy_score += 0.3
                reasons.append("Natural integration points")
            
            if synergy_score > 0.5:
                synergies.append({
                    'project': other_name,
                    'synergy_score': synergy_score,
                    'reasons': reasons,
                    'potential_integration': profile.problem_domain + ' + ' + other_profile.problem_domain
                })
        
        # Sort by synergy score
        synergies.sort(key=lambda x: x['synergy_score'], reverse=True)
        
        return synergies[:5]  # Top 5 synergies