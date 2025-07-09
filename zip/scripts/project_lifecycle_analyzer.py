"""
Project Lifecycle Analyzer

Analyzes projects to determine their current lifecycle stage and recommends
appropriate tasks to progress to the next stage.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field


class ProjectStage(Enum):
    """Project lifecycle stages."""
    INCEPTION = "inception"          # New project, basic setup needed
    EARLY_DEVELOPMENT = "early_development"  # Core features being built
    ACTIVE_DEVELOPMENT = "active_development"  # Rapid feature addition
    GROWTH = "growth"               # Scaling and optimization focus
    MATURE = "mature"               # Stable, incremental improvements
    MAINTENANCE = "maintenance"     # Bug fixes and updates only
    DECLINING = "declining"         # Low activity, may need revival
    ARCHIVED = "archived"           # No longer actively developed


@dataclass
class StageIndicators:
    """Indicators for determining project stage."""
    repository_age_days: int
    commit_frequency: float  # commits per week
    issue_velocity: float    # issues closed per week
    feature_vs_bug_ratio: float
    documentation_score: float  # 0-1
    test_coverage_score: float  # 0-1
    ci_cd_maturity: float      # 0-1
    code_complexity_trend: str  # increasing/stable/decreasing
    contributor_count: int
    last_major_release_days: int
    dependency_freshness: float  # 0-1
    security_score: float       # 0-1


@dataclass
class StageTransition:
    """Requirements for transitioning to the next stage."""
    target_stage: ProjectStage
    required_tasks: List[str]
    key_milestones: List[str]
    estimated_duration_days: int
    priority_focus: List[str]
    success_criteria: Dict[str, Any]


class ProjectLifecycleAnalyzer:
    """Analyzes project lifecycle and recommends stage-appropriate tasks."""
    
    def __init__(self, ai_brain=None):
        """Initialize the lifecycle analyzer.
        
        Args:
            ai_brain: AI brain for intelligent analysis
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Stage characteristics
        self.stage_profiles = self._define_stage_profiles()
        
        # Transition requirements
        self.stage_transitions = self._define_stage_transitions()
        
    def _define_stage_profiles(self) -> Dict[ProjectStage, Dict[str, Any]]:
        """Define characteristics of each lifecycle stage."""
        return {
            ProjectStage.INCEPTION: {
                "age_range": (0, 30),
                "commit_frequency": (0, 5),
                "typical_tasks": ["setup", "architecture", "initial_features"],
                "focus": "foundation",
                "indicators": {
                    "has_readme": False,
                    "has_ci": False,
                    "has_tests": False,
                    "core_features": False
                }
            },
            ProjectStage.EARLY_DEVELOPMENT: {
                "age_range": (15, 90),
                "commit_frequency": (5, 20),
                "typical_tasks": ["core_features", "basic_testing", "documentation"],
                "focus": "functionality",
                "indicators": {
                    "has_readme": True,
                    "has_ci": False,
                    "has_tests": True,
                    "core_features": "partial"
                }
            },
            ProjectStage.ACTIVE_DEVELOPMENT: {
                "age_range": (30, 180),
                "commit_frequency": (10, 50),
                "typical_tasks": ["features", "testing", "optimization"],
                "focus": "growth",
                "indicators": {
                    "has_readme": True,
                    "has_ci": True,
                    "has_tests": True,
                    "core_features": True,
                    "active_issues": True
                }
            },
            ProjectStage.GROWTH: {
                "age_range": (90, 365),
                "commit_frequency": (5, 30),
                "typical_tasks": ["scaling", "performance", "advanced_features"],
                "focus": "optimization",
                "indicators": {
                    "performance_focus": True,
                    "scaling_needs": True,
                    "user_growth": True
                }
            },
            ProjectStage.MATURE: {
                "age_range": (180, None),
                "commit_frequency": (2, 15),
                "typical_tasks": ["maintenance", "refactoring", "stability"],
                "focus": "stability",
                "indicators": {
                    "stable_api": True,
                    "comprehensive_docs": True,
                    "high_test_coverage": True
                }
            },
            ProjectStage.MAINTENANCE: {
                "age_range": (365, None),
                "commit_frequency": (0, 5),
                "typical_tasks": ["bug_fixes", "security_updates", "dependency_updates"],
                "focus": "upkeep",
                "indicators": {
                    "mostly_fixes": True,
                    "few_features": True
                }
            }
        }
    
    def _define_stage_transitions(self) -> Dict[Tuple[ProjectStage, ProjectStage], StageTransition]:
        """Define requirements for stage transitions."""
        return {
            (ProjectStage.INCEPTION, ProjectStage.EARLY_DEVELOPMENT): StageTransition(
                target_stage=ProjectStage.EARLY_DEVELOPMENT,
                required_tasks=[
                    "Complete initial setup",
                    "Define project architecture",
                    "Implement core data models",
                    "Set up development environment",
                    "Create basic documentation"
                ],
                key_milestones=[
                    "First working prototype",
                    "Basic CI/CD pipeline",
                    "Initial test suite"
                ],
                estimated_duration_days=30,
                priority_focus=["setup", "architecture", "core_features"],
                success_criteria={
                    "has_working_code": True,
                    "has_basic_docs": True,
                    "can_run_locally": True
                }
            ),
            (ProjectStage.EARLY_DEVELOPMENT, ProjectStage.ACTIVE_DEVELOPMENT): StageTransition(
                target_stage=ProjectStage.ACTIVE_DEVELOPMENT,
                required_tasks=[
                    "Complete core feature set",
                    "Achieve 60% test coverage",
                    "Set up automated deployment",
                    "Create user documentation",
                    "Implement error handling"
                ],
                key_milestones=[
                    "MVP release",
                    "First external users",
                    "Automated testing"
                ],
                estimated_duration_days=60,
                priority_focus=["features", "testing", "deployment"],
                success_criteria={
                    "core_features_complete": True,
                    "test_coverage": 0.6,
                    "ci_cd_active": True
                }
            ),
            (ProjectStage.ACTIVE_DEVELOPMENT, ProjectStage.GROWTH): StageTransition(
                target_stage=ProjectStage.GROWTH,
                required_tasks=[
                    "Optimize performance",
                    "Implement caching strategy",
                    "Add monitoring and analytics",
                    "Create scaling plan",
                    "Enhance security measures"
                ],
                key_milestones=[
                    "Performance benchmarks met",
                    "Scaling infrastructure ready",
                    "Security audit passed"
                ],
                estimated_duration_days=90,
                priority_focus=["performance", "scaling", "security"],
                success_criteria={
                    "performance_optimized": True,
                    "can_handle_load": True,
                    "security_hardened": True
                }
            )
        }
    
    async def analyze_project_stage(self, repository_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project to determine its lifecycle stage.
        
        Args:
            repository_analysis: Repository analysis data
            
        Returns:
            Lifecycle analysis including stage and recommendations
        """
        self.logger.info(f"Analyzing lifecycle stage for {repository_analysis.get('repository', 'unknown')}")
        
        # Calculate stage indicators
        indicators = self._calculate_indicators(repository_analysis)
        
        # Determine current stage
        current_stage = self._determine_stage(indicators, repository_analysis)
        
        # Get stage-appropriate tasks
        appropriate_tasks = await self._get_stage_appropriate_tasks(
            current_stage, repository_analysis
        )
        
        # Determine next stage transition
        transition_plan = self._get_transition_plan(current_stage, indicators)
        
        # Generate lifecycle insights
        insights = await self._generate_lifecycle_insights(
            current_stage, indicators, repository_analysis
        )
        
        return {
            "current_stage": current_stage.value,
            "stage_indicators": self._indicators_to_dict(indicators),
            "stage_confidence": self._calculate_stage_confidence(indicators, current_stage),
            "appropriate_task_types": appropriate_tasks,
            "transition_plan": transition_plan,
            "lifecycle_insights": insights,
            "stage_characteristics": self.stage_profiles.get(current_stage, {}),
            "recommended_focus": self._get_recommended_focus(current_stage, indicators)
        }
    
    def _calculate_indicators(self, repo_analysis: Dict[str, Any]) -> StageIndicators:
        """Calculate lifecycle indicators from repository analysis."""
        basic_info = repo_analysis.get('basic_info', {})
        health_metrics = repo_analysis.get('health_metrics', {})
        code_analysis = repo_analysis.get('code_analysis', {})
        issues_analysis = repo_analysis.get('issues_analysis', {})
        recent_activity = repo_analysis.get('recent_activity', {})
        
        # Calculate repository age
        created_at = basic_info.get('created_at', '')
        if created_at:
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age_days = (datetime.now(timezone.utc) - created_date).days
        else:
            age_days = 0
        
        # Calculate commit frequency (commits per week)
        recent_commits = health_metrics.get('recent_commits', 0)
        commit_frequency = (recent_commits / 30) * 7  # Convert monthly to weekly
        
        # Calculate issue velocity
        open_issues = issues_analysis.get('total_open', 0)
        closed_issues = basic_info.get('closed_issues_count', 0)
        issue_velocity = closed_issues / max(age_days / 7, 1) if age_days > 0 else 0
        
        # Calculate feature vs bug ratio
        bugs = issues_analysis.get('bug_count', 0)
        features = issues_analysis.get('feature_requests', 0)
        total_typed_issues = bugs + features
        feature_ratio = features / total_typed_issues if total_typed_issues > 0 else 0.5
        
        # Enhanced score calculations based on actual code analysis
        doc_score = self._calculate_documentation_score(code_analysis, repo_analysis)
        test_score = self._calculate_test_coverage_score(code_analysis, repo_analysis)
        ci_score = self._calculate_ci_maturity_score(code_analysis, repo_analysis)
        
        # Estimate other metrics
        contributors = recent_activity.get('active_contributors', 1)
        
        # Calculate code complexity trend
        complexity_trend = self._analyze_complexity_trend(repo_analysis)
        
        # Calculate dependency freshness
        dep_freshness = self._calculate_dependency_freshness(code_analysis)
        
        # Calculate security score
        security_score = self._calculate_security_score(repo_analysis)
        
        # Calculate last major release
        last_release_days = self._calculate_last_major_release(repo_analysis)
        
        return StageIndicators(
            repository_age_days=age_days,
            commit_frequency=commit_frequency,
            issue_velocity=issue_velocity,
            feature_vs_bug_ratio=feature_ratio,
            documentation_score=doc_score,
            test_coverage_score=test_score,
            ci_cd_maturity=ci_score,
            code_complexity_trend=complexity_trend,
            contributor_count=contributors,
            last_major_release_days=last_release_days,
            dependency_freshness=dep_freshness,
            security_score=security_score
        )
    
    def _determine_stage(self, indicators: StageIndicators, 
                        repo_analysis: Dict[str, Any]) -> ProjectStage:
        """Determine project stage based on indicators and code maturity."""
        # First, check code maturity indicators
        code_maturity = self._assess_code_maturity(repo_analysis)
        
        # Special case: Very new repos with no structure
        if indicators.repository_age_days < 7 and code_maturity['completeness'] < 0.2:
            return ProjectStage.INCEPTION
        
        # Check for production indicators
        is_production = self._check_production_indicators(repo_analysis)
        
        # If in production but low activity, it's maintenance
        if is_production and indicators.commit_frequency < 2:
            return ProjectStage.MAINTENANCE
        
        # Score each stage based on comprehensive analysis
        stage_scores = {}
        
        # INCEPTION: Project just starting
        inception_score = 0.0
        if code_maturity['has_basic_structure'] and not code_maturity['has_core_features']:
            inception_score += 0.4
        if indicators.documentation_score < 0.3:
            inception_score += 0.2
        if indicators.test_coverage_score < 0.2:
            inception_score += 0.2
        if not code_maturity['has_authentication']:
            inception_score += 0.2
        stage_scores[ProjectStage.INCEPTION] = inception_score
        
        # EARLY_DEVELOPMENT: Building core features
        early_dev_score = 0.0
        if code_maturity['has_basic_structure'] and code_maturity['completeness'] < 0.5:
            early_dev_score += 0.3
        if indicators.feature_vs_bug_ratio > 0.7:  # Mostly building features
            early_dev_score += 0.3
        if 0.2 <= indicators.test_coverage_score <= 0.5:
            early_dev_score += 0.2
        if code_maturity['has_authentication'] and not code_maturity['has_api_docs']:
            early_dev_score += 0.2
        stage_scores[ProjectStage.EARLY_DEVELOPMENT] = early_dev_score
        
        # ACTIVE_DEVELOPMENT: Rapid feature addition
        active_dev_score = 0.0
        if code_maturity['completeness'] >= 0.5 and indicators.commit_frequency > 5:
            active_dev_score += 0.4
        if indicators.issue_velocity > 1:
            active_dev_score += 0.2
        if 0.5 <= indicators.feature_vs_bug_ratio <= 0.8:
            active_dev_score += 0.2
        if code_maturity['has_ci_cd'] and indicators.test_coverage_score > 0.5:
            active_dev_score += 0.2
        stage_scores[ProjectStage.ACTIVE_DEVELOPMENT] = active_dev_score
        
        # GROWTH: Scaling and optimization
        growth_score = 0.0
        if code_maturity['completeness'] > 0.7 and code_maturity['has_monitoring']:
            growth_score += 0.4
        if indicators.commit_frequency > 3 and indicators.feature_vs_bug_ratio < 0.5:
            growth_score += 0.3
        if code_maturity['has_caching'] or code_maturity['has_optimization']:
            growth_score += 0.3
        stage_scores[ProjectStage.GROWTH] = growth_score
        
        # MATURE: Stable and refined
        mature_score = 0.0
        if code_maturity['completeness'] > 0.8 and indicators.documentation_score > 0.7:
            mature_score += 0.4
        if indicators.feature_vs_bug_ratio < 0.3:  # Mostly maintenance
            mature_score += 0.3
        if indicators.test_coverage_score > 0.7 and code_maturity['has_api_docs']:
            mature_score += 0.3
        stage_scores[ProjectStage.MATURE] = mature_score
        
        # MAINTENANCE: Minimal changes
        maintenance_score = 0.0
        if indicators.commit_frequency < 3 and indicators.repository_age_days > 365:
            maintenance_score += 0.4
        if indicators.feature_vs_bug_ratio < 0.2:
            maintenance_score += 0.3
        if is_production and code_maturity['completeness'] > 0.8:
            maintenance_score += 0.3
        stage_scores[ProjectStage.MAINTENANCE] = maintenance_score
        
        # DECLINING: Very low activity
        if indicators.commit_frequency < 0.5 and indicators.issue_velocity < 0.1:
            if indicators.repository_age_days > 180:
                return ProjectStage.DECLINING
        
        # Return stage with highest score
        best_stage = max(stage_scores.items(), key=lambda x: x[1])
        
        # Log the analysis for debugging
        self.logger.debug(f"Stage scores: {stage_scores}")
        self.logger.debug(f"Selected stage: {best_stage[0].value} (score: {best_stage[1]})")
        
        return best_stage[0]
    
    def _assess_code_maturity(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess code maturity based on actual implementation."""
        code_analysis = repo_analysis.get('code_analysis', {})
        architecture = repo_analysis.get('architecture', {})
        
        maturity = {
            'has_basic_structure': False,
            'has_core_features': False,
            'has_authentication': False,
            'has_api_docs': False,
            'has_ci_cd': False,
            'has_monitoring': False,
            'has_caching': False,
            'has_optimization': False,
            'completeness': 0.0
        }
        
        # Check basic structure
        config_files = code_analysis.get('config_files', [])
        if any('package.json' in f or 'composer.json' in f for f in config_files):
            maturity['has_basic_structure'] = True
        
        # Check for authentication
        if architecture.get('has_authentication', False):
            maturity['has_authentication'] = True
        elif any('auth' in f.lower() for f in code_analysis.get('directories', [])):
            maturity['has_authentication'] = True
            
        # Check for API documentation
        docs = code_analysis.get('documentation', [])
        if any('api' in d.lower() or 'swagger' in d.lower() for d in docs):
            maturity['has_api_docs'] = True
            
        # Check CI/CD
        if any('.github/workflows' in f or '.gitlab-ci' in f for f in config_files):
            maturity['has_ci_cd'] = True
            
        # Check for monitoring/logging
        if any('monitor' in f.lower() or 'log' in f.lower() for f in config_files):
            maturity['has_monitoring'] = True
            
        # Check for caching
        if any('cache' in f.lower() or 'redis' in f.lower() for f in config_files):
            maturity['has_caching'] = True
            
        # Check core features based on architecture
        core_entities = architecture.get('core_entities', [])
        if len(core_entities) > 3:
            maturity['has_core_features'] = True
            
        # Calculate overall completeness
        score = 0.0
        if maturity['has_basic_structure']: score += 0.15
        if maturity['has_core_features']: score += 0.25
        if maturity['has_authentication']: score += 0.15
        if maturity['has_api_docs']: score += 0.1
        if maturity['has_ci_cd']: score += 0.15
        if maturity['has_monitoring']: score += 0.1
        if maturity['has_caching']: score += 0.1
        
        maturity['completeness'] = score
        
        return maturity
    
    def _check_production_indicators(self, repo_analysis: Dict[str, Any]) -> bool:
        """Check if project shows signs of being in production."""
        indicators = []
        
        # Check for production config files
        config_files = repo_analysis.get('code_analysis', {}).get('config_files', [])
        indicators.append(any('.env.production' in f for f in config_files))
        indicators.append(any('deploy' in f.lower() for f in config_files))
        
        # Check for monitoring/logging setup
        indicators.append(repo_analysis.get('code_maturity', {}).get('has_monitoring', False))
        
        # Check for release tags
        basic_info = repo_analysis.get('basic_info', {})
        indicators.append(basic_info.get('has_releases', False))
        
        # If 2+ indicators present, likely in production
        return sum(indicators) >= 2
    
    async def _get_stage_appropriate_tasks(self, stage: ProjectStage, 
                                         repo_analysis: Dict[str, Any]) -> List[str]:
        """Get task types appropriate for the current stage."""
        base_tasks = self.stage_profiles[stage]["typical_tasks"]
        
        if not self.ai_brain:
            return base_tasks
        
        # Use AI to determine specific appropriate tasks
        prompt = f"""
        Based on this project's lifecycle stage and analysis, determine appropriate task types.
        
        Current Stage: {stage.value}
        Stage Characteristics: {json.dumps(self.stage_profiles[stage], indent=2)}
        
        Repository Analysis Summary:
        - Health Score: {repo_analysis.get('health_metrics', {}).get('health_score', 0)}
        - Open Issues: {repo_analysis.get('issues_analysis', {}).get('total_open', 0)}
        - Has Tests: {repo_analysis.get('code_analysis', {}).get('test_coverage', 'unknown')}
        - Documentation: {len(repo_analysis.get('code_analysis', {}).get('documentation', []))} files
        
        Specific Needs:
        {json.dumps(repo_analysis.get('specific_needs', []), indent=2)}
        
        Return a list of 5-8 specific task types that would be most valuable for this project
        at its current stage. Consider what would help it progress to the next stage.
        
        Format as JSON: {{"task_types": ["type1", "type2", ...], "rationale": "explanation"}}
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        result = self._parse_json_response(response)
        
        return result.get("task_types", base_tasks)
    
    def _get_transition_plan(self, current_stage: ProjectStage, 
                           indicators: StageIndicators) -> Dict[str, Any]:
        """Get plan for transitioning to the next stage."""
        # Determine logical next stage
        next_stage_map = {
            ProjectStage.INCEPTION: ProjectStage.EARLY_DEVELOPMENT,
            ProjectStage.EARLY_DEVELOPMENT: ProjectStage.ACTIVE_DEVELOPMENT,
            ProjectStage.ACTIVE_DEVELOPMENT: ProjectStage.GROWTH,
            ProjectStage.GROWTH: ProjectStage.MATURE,
            ProjectStage.MATURE: ProjectStage.MAINTENANCE
        }
        
        next_stage = next_stage_map.get(current_stage)
        if not next_stage:
            return {
                "next_stage": None,
                "reason": "Project is in final lifecycle stage"
            }
        
        # Get transition requirements
        transition_key = (current_stage, next_stage)
        transition = self.stage_transitions.get(transition_key)
        
        if not transition:
            # Create generic transition plan
            return {
                "next_stage": next_stage.value,
                "required_improvements": self._identify_improvements_needed(
                    current_stage, next_stage, indicators
                ),
                "estimated_duration": "2-3 months",
                "focus_areas": self.stage_profiles[next_stage]["typical_tasks"]
            }
        
        return {
            "next_stage": transition.target_stage.value,
            "required_tasks": transition.required_tasks,
            "key_milestones": transition.key_milestones,
            "estimated_duration_days": transition.estimated_duration_days,
            "priority_focus": transition.priority_focus,
            "success_criteria": transition.success_criteria,
            "current_readiness": self._calculate_transition_readiness(indicators, transition)
        }
    
    def _identify_improvements_needed(self, current: ProjectStage, 
                                    target: ProjectStage,
                                    indicators: StageIndicators) -> List[str]:
        """Identify improvements needed for stage transition."""
        improvements = []
        
        if indicators.documentation_score < 0.7:
            improvements.append("Improve documentation coverage")
        
        if indicators.test_coverage_score < 0.6:
            improvements.append("Increase test coverage to 60%+")
        
        if indicators.ci_cd_maturity < 0.5:
            improvements.append("Implement CI/CD pipeline")
        
        if current == ProjectStage.INCEPTION:
            improvements.extend([
                "Complete initial setup",
                "Define architecture",
                "Implement core features"
            ])
        elif current == ProjectStage.EARLY_DEVELOPMENT:
            improvements.extend([
                "Stabilize core features",
                "Add comprehensive testing",
                "Set up deployment"
            ])
        
        return improvements[:5]  # Top 5 improvements
    
    def _calculate_transition_readiness(self, indicators: StageIndicators,
                                      transition: StageTransition) -> float:
        """Calculate readiness for stage transition (0-1)."""
        readiness_score = 0.0
        factors = 0
        
        # Check success criteria
        for criterion, required in transition.success_criteria.items():
            factors += 1
            if criterion == "test_coverage" and indicators.test_coverage_score >= required:
                readiness_score += 1
            elif criterion == "ci_cd_active" and indicators.ci_cd_maturity > 0.7:
                readiness_score += 1
            # Add more criteria checks as needed
        
        return readiness_score / factors if factors > 0 else 0.0
    
    async def _generate_lifecycle_insights(self, stage: ProjectStage,
                                         indicators: StageIndicators,
                                         repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights about project lifecycle."""
        insights = {
            "health_assessment": self._assess_stage_health(stage, indicators),
            "growth_trajectory": self._analyze_growth_trajectory(indicators),
            "risk_factors": self._identify_lifecycle_risks(stage, indicators),
            "opportunities": self._identify_lifecycle_opportunities(stage, repo_analysis)
        }
        
        if self.ai_brain:
            # Get AI-powered insights
            prompt = f"""
            Provide lifecycle insights for this project:
            
            Stage: {stage.value}
            Age: {indicators.repository_age_days} days
            Commit Frequency: {indicators.commit_frequency:.1f}/week
            Issue Velocity: {indicators.issue_velocity:.1f}/week
            
            Key Observations:
            - Documentation Score: {indicators.documentation_score:.1%}
            - Test Coverage: {indicators.test_coverage_score:.1%}
            - CI/CD Maturity: {indicators.ci_cd_maturity:.1%}
            
            Provide insights on:
            1. Is the project progressing normally for its stage?
            2. What are the key blockers to advancement?
            3. What quick wins could accelerate progress?
            4. Any concerning patterns to address?
            
            Format as JSON with: assessment, blockers, quick_wins, concerns
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            ai_insights = self._parse_json_response(response)
            insights.update(ai_insights)
        
        return insights
    
    def _assess_stage_health(self, stage: ProjectStage, 
                           indicators: StageIndicators) -> str:
        """Assess health relative to lifecycle stage."""
        expected_profile = self.stage_profiles[stage]
        
        # Compare actual vs expected commit frequency
        expected_freq = expected_profile["commit_frequency"]
        if indicators.commit_frequency < expected_freq[0]:
            return "below_expected_activity"
        elif indicators.commit_frequency > expected_freq[1]:
            return "above_expected_activity"
        else:
            return "healthy_activity_level"
    
    def _analyze_growth_trajectory(self, indicators: StageIndicators) -> str:
        """Analyze project growth trajectory."""
        if indicators.commit_frequency > 20 and indicators.issue_velocity > 2:
            return "rapid_growth"
        elif indicators.commit_frequency > 10:
            return "steady_growth"
        elif indicators.commit_frequency > 5:
            return "moderate_growth"
        elif indicators.commit_frequency > 1:
            return "slow_growth"
        else:
            return "stagnant"
    
    def _identify_lifecycle_risks(self, stage: ProjectStage,
                                indicators: StageIndicators) -> List[str]:
        """Identify risks based on lifecycle stage."""
        risks = []
        
        if stage == ProjectStage.INCEPTION and indicators.repository_age_days > 60:
            risks.append("Project inception taking too long")
        
        if stage == ProjectStage.ACTIVE_DEVELOPMENT and indicators.commit_frequency < 5:
            risks.append("Development velocity too low for active stage")
        
        if indicators.documentation_score < 0.3:
            risks.append("Insufficient documentation may hinder growth")
        
        if indicators.test_coverage_score < 0.4 and stage != ProjectStage.INCEPTION:
            risks.append("Low test coverage increases technical debt")
        
        if indicators.contributor_count == 1:
            risks.append("Single contributor creates bottleneck risk")
        
        return risks
    
    def _identify_lifecycle_opportunities(self, stage: ProjectStage,
                                        repo_analysis: Dict[str, Any]) -> List[str]:
        """Identify opportunities based on lifecycle stage."""
        opportunities = []
        
        if stage == ProjectStage.EARLY_DEVELOPMENT:
            opportunities.append("Perfect time to establish testing practices")
            opportunities.append("Set up CI/CD before codebase grows")
        
        if stage == ProjectStage.ACTIVE_DEVELOPMENT:
            opportunities.append("High activity period ideal for major features")
            opportunities.append("Good time to expand contributor base")
        
        if stage == ProjectStage.GROWTH:
            opportunities.append("Focus on performance optimization")
            opportunities.append("Implement advanced monitoring")
        
        return opportunities
    
    def _get_recommended_focus(self, stage: ProjectStage,
                             indicators: StageIndicators) -> List[str]:
        """Get recommended focus areas for current stage."""
        base_focus = self.stage_profiles[stage]["typical_tasks"]
        
        # Adjust based on specific weaknesses
        if indicators.documentation_score < 0.5:
            base_focus.insert(0, "documentation")
        
        if indicators.test_coverage_score < 0.5:
            base_focus.insert(0, "testing")
        
        if indicators.ci_cd_maturity < 0.5 and stage != ProjectStage.INCEPTION:
            base_focus.insert(0, "automation")
        
        return base_focus[:5]
    
    def _calculate_stage_confidence(self, indicators: StageIndicators,
                                  stage: ProjectStage) -> float:
        """Calculate confidence in stage determination."""
        confidence = 0.5  # Base confidence
        
        profile = self.stage_profiles[stage]
        
        # Check age match
        age_range = profile.get("age_range", (0, None))
        if age_range[1] is None:
            if indicators.repository_age_days >= age_range[0]:
                confidence += 0.2
        else:
            if age_range[0] <= indicators.repository_age_days <= age_range[1]:
                confidence += 0.3
        
        # Check activity match
        freq_range = profile.get("commit_frequency", (0, 100))
        if freq_range[0] <= indicators.commit_frequency <= freq_range[1]:
            confidence += 0.2
        
        return min(confidence, 0.95)
    
    def _indicators_to_dict(self, indicators: StageIndicators) -> Dict[str, Any]:
        """Convert indicators to dictionary."""
        return {
            "repository_age_days": indicators.repository_age_days,
            "commit_frequency_per_week": round(indicators.commit_frequency, 1),
            "issue_velocity_per_week": round(indicators.issue_velocity, 1),
            "feature_vs_bug_ratio": round(indicators.feature_vs_bug_ratio, 2),
            "documentation_score": round(indicators.documentation_score, 2),
            "test_coverage_score": round(indicators.test_coverage_score, 2),
            "ci_cd_maturity": round(indicators.ci_cd_maturity, 2),
            "code_complexity_trend": indicators.code_complexity_trend,
            "contributor_count": indicators.contributor_count,
            "last_major_release_days": indicators.last_major_release_days,
            "dependency_freshness": round(indicators.dependency_freshness, 2),
            "security_score": round(indicators.security_score, 2)
        }
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from AI response."""
        content = response.get('content', '')
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            
        return {}
    
    def _calculate_documentation_score(self, code_analysis: Dict[str, Any], 
                                     repo_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive documentation score."""
        score = 0.0
        
        # Check for various documentation indicators
        docs = code_analysis.get('documentation', [])
        has_readme = any('README' in doc.upper() for doc in docs)
        has_api_docs = any('api' in doc.lower() or 'swagger' in doc.lower() for doc in docs)
        has_architecture = repo_analysis.get('architecture', {}).get('document_exists', False)
        
        # Calculate score
        if has_readme:
            score += 0.3
        if has_api_docs:
            score += 0.3
        if has_architecture:
            score += 0.2
        if len(docs) > 3:
            score += 0.2
            
        return min(score, 1.0)
    
    def _calculate_test_coverage_score(self, code_analysis: Dict[str, Any],
                                     repo_analysis: Dict[str, Any]) -> float:
        """Calculate test coverage score based on actual test presence."""
        test_coverage = code_analysis.get('test_coverage', 'unknown')
        
        if test_coverage == 'comprehensive':
            return 0.9
        elif test_coverage == 'has_tests':
            # Check for different test types
            test_dirs = code_analysis.get('test_directories', [])
            if len(test_dirs) > 2:
                return 0.8
            elif len(test_dirs) > 0:
                return 0.6
            else:
                return 0.4
        else:
            return 0.1
    
    def _calculate_ci_maturity_score(self, code_analysis: Dict[str, Any],
                                   repo_analysis: Dict[str, Any]) -> float:
        """Calculate CI/CD maturity score."""
        score = 0.0
        config_files = code_analysis.get('config_files', [])
        
        # Check for CI/CD indicators
        if any('.github/workflows' in cf for cf in config_files):
            score += 0.4
        if any('.gitlab-ci' in cf for cf in config_files):
            score += 0.4
        if any('Dockerfile' in cf for cf in config_files):
            score += 0.2
        if any('docker-compose' in cf for cf in config_files):
            score += 0.1
        if any('.env.example' in cf for cf in config_files):
            score += 0.1
            
        return min(score, 1.0)
    
    def _analyze_complexity_trend(self, repo_analysis: Dict[str, Any]) -> str:
        """Analyze code complexity trend."""
        # This would need more sophisticated analysis in a real implementation
        recent_activity = repo_analysis.get('recent_activity', {})
        
        if recent_activity.get('refactoring_commits', 0) > 2:
            return "decreasing"
        elif repo_analysis.get('health_metrics', {}).get('code_duplication', 0) > 20:
            return "increasing"
        else:
            return "stable"
    
    def _calculate_dependency_freshness(self, code_analysis: Dict[str, Any]) -> float:
        """Calculate dependency freshness score."""
        # Check for dependency files
        config_files = code_analysis.get('config_files', [])
        
        has_lockfile = any('lock' in cf.lower() for cf in config_files)
        has_deps = any(cf.endswith(('.json', '.toml', '.yaml', '.yml')) 
                      for cf in config_files if 'package' in cf or 'requirements' in cf)
        
        if has_lockfile and has_deps:
            return 0.8  # Assume relatively fresh if lockfiles exist
        elif has_deps:
            return 0.6
        else:
            return 0.4
    
    def _calculate_security_score(self, repo_analysis: Dict[str, Any]) -> float:
        """Calculate security score."""
        score = 0.7  # Base score
        
        # Check for security indicators
        code_analysis = repo_analysis.get('code_analysis', {})
        config_files = code_analysis.get('config_files', [])
        
        if any('security' in cf.lower() for cf in config_files):
            score += 0.1
        if repo_analysis.get('health_metrics', {}).get('vulnerability_alerts', 0) == 0:
            score += 0.1
        if any('.env.example' in cf for cf in config_files):
            score += 0.1  # Proper env handling
            
        return min(score, 1.0)
    
    def _calculate_last_major_release(self, repo_analysis: Dict[str, Any]) -> int:
        """Calculate days since last major release."""
        # This would need release API analysis in real implementation
        recent_activity = repo_analysis.get('recent_activity', {})
        
        # Estimate based on activity patterns
        if recent_activity.get('recent_commits', 0) > 50:
            return 15  # Likely recent release
        else:
            return 60  # Conservative estimate