"""
Project Outcome Tracker for Adaptive Learning

Tracks the success and outcomes of generated projects to improve
future recommendations through learning from past decisions.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio

from scripts.ai_brain import IntelligentAIBrain


@dataclass
class ProjectOutcome:
    """Represents the outcome of a generated project."""
    project_id: str
    project_name: str
    creation_date: datetime
    project_type: str
    tech_stack: List[str]
    problem_statement: str
    target_market: str
    initial_scores: Dict[str, float]  # market_demand, innovation_score, etc.
    
    # Outcome metrics
    completion_status: str  # 'active', 'completed', 'abandoned', 'pivoted'
    health_score: float  # 0.0 to 1.0
    activity_level: str  # 'high', 'medium', 'low', 'none'
    commits_count: int
    contributors_count: int
    issues_closed: int
    features_implemented: int
    user_engagement: float  # 0.0 to 1.0
    
    # Learning insights
    success_factors: List[str]
    failure_factors: List[str]
    lessons_learned: List[str]
    recommendation_accuracy: float  # How accurate was our initial assessment
    
    # Timestamps
    last_updated: datetime
    evaluation_date: Optional[datetime]


@dataclass
class LearningInsight:
    """Insights derived from project outcomes."""
    insight_type: str  # 'pattern', 'correlation', 'trend'
    description: str
    confidence: float  # 0.0 to 1.0
    affected_categories: List[str]
    recommendations: List[str]
    supporting_evidence: List[str]


class ProjectOutcomeTracker:
    """Tracks and learns from project outcomes."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, storage_path: str = "project_outcomes.json"):
        """Initialize the outcome tracker.
        
        Args:
            ai_brain: AI brain for analysis
            storage_path: Path to store outcome data
        """
        self.ai_brain = ai_brain
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)
        
        # Load existing outcomes
        self.outcomes: Dict[str, ProjectOutcome] = self._load_outcomes()
        self.learning_insights: List[LearningInsight] = []
        
    def _load_outcomes(self) -> Dict[str, ProjectOutcome]:
        """Load existing project outcomes from storage."""
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
            outcomes = {}
            for project_id, outcome_data in data.items():
                # Convert string dates back to datetime
                outcome_data['creation_date'] = datetime.fromisoformat(outcome_data['creation_date'])
                outcome_data['last_updated'] = datetime.fromisoformat(outcome_data['last_updated'])
                if outcome_data.get('evaluation_date'):
                    outcome_data['evaluation_date'] = datetime.fromisoformat(outcome_data['evaluation_date'])
                
                outcomes[project_id] = ProjectOutcome(**outcome_data)
            
            return outcomes
            
        except Exception as e:
            self.logger.error(f"Error loading outcomes: {e}")
            return {}
    
    def _save_outcomes(self) -> None:
        """Save outcomes to storage."""
        try:
            data = {}
            for project_id, outcome in self.outcomes.items():
                outcome_dict = asdict(outcome)
                # Convert datetime to ISO format for JSON
                outcome_dict['creation_date'] = outcome.creation_date.isoformat()
                outcome_dict['last_updated'] = outcome.last_updated.isoformat()
                if outcome.evaluation_date:
                    outcome_dict['evaluation_date'] = outcome.evaluation_date.isoformat()
                
                data[project_id] = outcome_dict
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving outcomes: {e}")
    
    async def track_new_project(
        self,
        project_id: str,
        project_name: str,
        project_metadata: Dict[str, Any]
    ) -> None:
        """Track a newly created project.
        
        Args:
            project_id: Unique project identifier
            project_name: Project name
            project_metadata: Metadata from project creation
        """
        outcome = ProjectOutcome(
            project_id=project_id,
            project_name=project_name,
            creation_date=datetime.now(timezone.utc),
            project_type=project_metadata.get('project_type', 'unknown'),
            tech_stack=project_metadata.get('tech_stack', []),
            problem_statement=project_metadata.get('problem_statement', ''),
            target_market=project_metadata.get('target_market', ''),
            initial_scores={
                'market_demand': project_metadata.get('market_demand', 0.5),
                'innovation_score': project_metadata.get('innovation_score', 0.5),
                'complexity': project_metadata.get('complexity', 0.5)
            },
            completion_status='active',
            health_score=1.0,
            activity_level='high',
            commits_count=0,
            contributors_count=1,
            issues_closed=0,
            features_implemented=0,
            user_engagement=0.0,
            success_factors=[],
            failure_factors=[],
            lessons_learned=[],
            recommendation_accuracy=0.0,
            last_updated=datetime.now(timezone.utc),
            evaluation_date=None
        )
        
        self.outcomes[project_id] = outcome
        self._save_outcomes()
        
        self.logger.info(f"📊 Tracking new project: {project_name} ({project_id})")
    
    async def update_project_metrics(
        self,
        project_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Update project metrics.
        
        Args:
            project_id: Project identifier
            metrics: Updated metrics
        """
        if project_id not in self.outcomes:
            self.logger.warning(f"Unknown project: {project_id}")
            return
        
        outcome = self.outcomes[project_id]
        
        # Update metrics
        outcome.health_score = metrics.get('health_score', outcome.health_score)
        outcome.commits_count = metrics.get('commits_count', outcome.commits_count)
        outcome.contributors_count = metrics.get('contributors_count', outcome.contributors_count)
        outcome.issues_closed = metrics.get('issues_closed', outcome.issues_closed)
        outcome.features_implemented = metrics.get('features_implemented', outcome.features_implemented)
        
        # Determine activity level
        if outcome.commits_count > 50:
            outcome.activity_level = 'high'
        elif outcome.commits_count > 20:
            outcome.activity_level = 'medium'
        elif outcome.commits_count > 5:
            outcome.activity_level = 'low'
        else:
            outcome.activity_level = 'none'
        
        outcome.last_updated = datetime.now(timezone.utc)
        
        self._save_outcomes()
    
    async def evaluate_project_outcome(self, project_id: str) -> Dict[str, Any]:
        """Evaluate a project's outcome and learn from it.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Evaluation results
        """
        if project_id not in self.outcomes:
            return {'error': 'Unknown project'}
        
        outcome = self.outcomes[project_id]
        
        try:
            # Use AI to analyze the project outcome
            prompt = f"""
            Evaluate this project's outcome:
            
            Project: {outcome.project_name}
            Type: {outcome.project_type}
            Problem: {outcome.problem_statement}
            Target Market: {outcome.target_market}
            Tech Stack: {', '.join(outcome.tech_stack)}
            
            Initial Predictions:
            - Market Demand: {outcome.initial_scores.get('market_demand', 0)}
            - Innovation Score: {outcome.initial_scores.get('innovation_score', 0)}
            
            Actual Outcomes:
            - Health Score: {outcome.health_score}
            - Activity Level: {outcome.activity_level}
            - Commits: {outcome.commits_count}
            - Contributors: {outcome.contributors_count}
            - Features Implemented: {outcome.features_implemented}
            
            Analyze:
            1. Was this project successful? Why or why not?
            2. What factors contributed to its success/failure?
            3. How accurate were our initial predictions?
            4. What lessons can we learn for future projects?
            5. What types of projects should we prioritize/avoid?
            
            Provide:
            - success_score: 0.0-1.0
            - success_factors: List of factors that helped
            - failure_factors: List of factors that hindered
            - lessons_learned: Key takeaways
            - recommendation_accuracy: 0.0-1.0 (how accurate were predictions)
            - future_recommendations: Specific recommendations
            
            Return as JSON object.
            """
            
            response = await self.ai_brain.execute_capability(
                'problem_analysis',
                {'prompt': prompt}
            )
            
            if response and response.get('status') == 'success':
                evaluation = self._parse_json(response.get('result', ''))
                
                if evaluation:
                    # Update outcome with evaluation
                    outcome.success_factors = evaluation.get('success_factors', [])
                    outcome.failure_factors = evaluation.get('failure_factors', [])
                    outcome.lessons_learned = evaluation.get('lessons_learned', [])
                    outcome.recommendation_accuracy = float(evaluation.get('recommendation_accuracy', 0.5))
                    outcome.evaluation_date = datetime.now(timezone.utc)
                    
                    # Update completion status based on success
                    success_score = float(evaluation.get('success_score', 0.5))
                    if success_score > 0.7:
                        outcome.completion_status = 'completed'
                    elif success_score < 0.3:
                        outcome.completion_status = 'abandoned'
                    
                    self._save_outcomes()
                    
                    return {
                        'project_id': project_id,
                        'success_score': success_score,
                        'evaluation': evaluation
                    }
                    
        except Exception as e:
            self.logger.error(f"Error evaluating project: {e}")
        
        return {'error': 'Evaluation failed'}
    
    async def generate_learning_insights(self) -> List[LearningInsight]:
        """Generate learning insights from all project outcomes."""
        self.logger.info("🧠 Generating learning insights from project outcomes...")
        
        # Group projects by various dimensions
        by_type = {}
        by_tech = {}
        by_market = {}
        successful_projects = []
        failed_projects = []
        
        for outcome in self.outcomes.values():
            # Skip unevaluated projects
            if not outcome.evaluation_date:
                continue
            
            # Group by type
            if outcome.project_type not in by_type:
                by_type[outcome.project_type] = []
            by_type[outcome.project_type].append(outcome)
            
            # Group by tech stack
            for tech in outcome.tech_stack:
                if tech not in by_tech:
                    by_tech[tech] = []
                by_tech[tech].append(outcome)
            
            # Group by market
            if outcome.target_market not in by_market:
                by_market[outcome.target_market] = []
            by_market[outcome.target_market].append(outcome)
            
            # Categorize by success
            if outcome.completion_status == 'completed':
                successful_projects.append(outcome)
            elif outcome.completion_status == 'abandoned':
                failed_projects.append(outcome)
        
        insights = []
        
        # Analyze patterns
        try:
            # Technology success patterns
            for tech, projects in by_tech.items():
                if len(projects) >= 3:  # Need enough data
                    success_rate = sum(1 for p in projects if p.completion_status == 'completed') / len(projects)
                    
                    if success_rate > 0.7:
                        insights.append(LearningInsight(
                            insight_type='pattern',
                            description=f"Projects using {tech} have {success_rate:.0%} success rate",
                            confidence=min(len(projects) / 10, 1.0),
                            affected_categories=['technology'],
                            recommendations=[f"Prioritize {tech} for new projects"],
                            supporting_evidence=[p.project_name for p in projects[:3]]
                        ))
                    elif success_rate < 0.3:
                        insights.append(LearningInsight(
                            insight_type='pattern',
                            description=f"Projects using {tech} struggle with {success_rate:.0%} success rate",
                            confidence=min(len(projects) / 10, 1.0),
                            affected_categories=['technology'],
                            recommendations=[f"Avoid {tech} or provide better support"],
                            supporting_evidence=[p.project_name for p in projects[:3]]
                        ))
            
            # Market fit patterns
            for market, projects in by_market.items():
                if len(projects) >= 2:
                    avg_engagement = sum(p.user_engagement for p in projects) / len(projects)
                    
                    if avg_engagement > 0.7:
                        insights.append(LearningInsight(
                            insight_type='correlation',
                            description=f"{market} market shows high engagement ({avg_engagement:.0%})",
                            confidence=min(len(projects) / 5, 1.0),
                            affected_categories=['market'],
                            recommendations=[f"Focus on {market} for high-impact projects"],
                            supporting_evidence=[p.project_name for p in projects]
                        ))
            
            # Common success factors
            all_success_factors = []
            for project in successful_projects:
                all_success_factors.extend(project.success_factors)
            
            # Count factor frequency
            factor_counts = {}
            for factor in all_success_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            # Identify top factors
            top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if top_factors:
                insights.append(LearningInsight(
                    insight_type='pattern',
                    description="Key success factors identified across projects",
                    confidence=0.8,
                    affected_categories=['strategy'],
                    recommendations=[f"Ensure new projects have: {', '.join(f[0] for f in top_factors[:3])}"],
                    supporting_evidence=[f"{factor}: {count} projects" for factor, count in top_factors]
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
        
        self.learning_insights = insights
        return insights
    
    def get_recommendations_for_new_project(
        self,
        project_type: str,
        tech_stack: List[str],
        target_market: str
    ) -> Dict[str, Any]:
        """Get recommendations based on learned patterns.
        
        Args:
            project_type: Type of project
            tech_stack: Planned technologies
            target_market: Target market
            
        Returns:
            Recommendations and predictions
        """
        recommendations = {
            'predicted_success_rate': 0.5,
            'risk_factors': [],
            'success_factors': [],
            'similar_projects': [],
            'recommendations': []
        }
        
        # Find similar successful projects
        for outcome in self.outcomes.values():
            similarity_score = 0.0
            
            if outcome.project_type == project_type:
                similarity_score += 0.4
            
            tech_overlap = set(outcome.tech_stack) & set(tech_stack)
            if tech_overlap:
                similarity_score += 0.3 * (len(tech_overlap) / len(tech_stack))
            
            if outcome.target_market == target_market:
                similarity_score += 0.3
            
            if similarity_score > 0.6:
                recommendations['similar_projects'].append({
                    'name': outcome.project_name,
                    'success': outcome.completion_status == 'completed',
                    'similarity': similarity_score,
                    'lessons': outcome.lessons_learned[:2]
                })
        
        # Calculate predicted success rate
        if recommendations['similar_projects']:
            success_count = sum(1 for p in recommendations['similar_projects'] if p['success'])
            recommendations['predicted_success_rate'] = success_count / len(recommendations['similar_projects'])
        
        # Apply learning insights
        for insight in self.learning_insights:
            if project_type in str(insight.affected_categories):
                recommendations['recommendations'].extend(insight.recommendations)
            
            for tech in tech_stack:
                if tech in insight.description:
                    if 'struggle' in insight.description or 'avoid' in insight.description:
                        recommendations['risk_factors'].append(insight.description)
                    else:
                        recommendations['success_factors'].append(insight.description)
        
        return recommendations
    
    def _parse_json(self, result: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing JSON: {e}")
        return None
    
    def get_outcome_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked outcomes."""
        total_projects = len(self.outcomes)
        evaluated_projects = sum(1 for o in self.outcomes.values() if o.evaluation_date)
        
        success_rate = 0.0
        avg_accuracy = 0.0
        
        if evaluated_projects > 0:
            successful = sum(1 for o in self.outcomes.values() 
                           if o.completion_status == 'completed')
            success_rate = successful / evaluated_projects
            
            total_accuracy = sum(o.recommendation_accuracy for o in self.outcomes.values() 
                               if o.evaluation_date)
            avg_accuracy = total_accuracy / evaluated_projects
        
        return {
            'total_tracked': total_projects,
            'evaluated': evaluated_projects,
            'success_rate': success_rate,
            'prediction_accuracy': avg_accuracy,
            'insights_generated': len(self.learning_insights),
            'active_projects': sum(1 for o in self.outcomes.values() 
                                 if o.completion_status == 'active')
        }