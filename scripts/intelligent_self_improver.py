"""
Intelligent Self-Improver

Integrates AI-powered analysis with staged improvements for intelligent,
context-aware code improvements.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from ai_brain import IntelligentAIBrain
from ai_code_analyzer import AICodeAnalyzer, CodeImprovement
from context_aware_improver import ContextAwareImprover
from improvement_learning_system import ImprovementLearningSystem
from staged_self_improver import StagedSelfImprover
from progressive_confidence import ProgressiveConfidence
from safe_self_improver import ModificationType, Modification
from research_knowledge_store import ResearchKnowledgeStore


class IntelligentSelfImprover:
    """Orchestrates intelligent self-improvement using AI and context awareness."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, repo_path: str,
                 staging_enabled: bool = True):
        """Initialize the intelligent self-improver.
        
        Args:
            ai_brain: The AI brain for analysis
            repo_path: Path to the repository
            staging_enabled: Whether to use staging
        """
        self.ai_brain = ai_brain
        self.repo_path = repo_path
        self.staging_enabled = staging_enabled
        
        # Initialize components
        self.code_analyzer = AICodeAnalyzer(ai_brain)
        self.context_improver = ContextAwareImprover(ai_brain, repo_path)
        self.learning_system = ImprovementLearningSystem()
        self.confidence_system = ProgressiveConfidence(repo_path)
        self.knowledge_store = ResearchKnowledgeStore()
        
        if staging_enabled:
            self.staged_improver = StagedSelfImprover(repo_path)
        else:
            self.staged_improver = None
        
        # Configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for intelligent improvements."""
        config_path = os.path.join(self.repo_path, '.self_improver', 'intelligent_config.json')
        
        default_config = {
            'min_confidence': 0.7,
            'max_improvements_per_file': 5,
            'max_daily_improvements': 20,
            'auto_apply_threshold': 0.85,
            'context_awareness_enabled': True,
            'learning_enabled': True,
            'research_integration_enabled': True,
            'improvement_priorities': {
                'security': 1.5,
                'optimization': 1.2,
                'quality': 1.0,
                'documentation': 0.8
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    async def find_intelligent_improvements(self, target_files: Optional[List[str]] = None,
                                          max_improvements: int = 10) -> List[Dict[str, Any]]:
        """Find improvements using AI-powered analysis.
        
        Args:
            target_files: Specific files to analyze (None for automatic selection)
            max_improvements: Maximum improvements to return
            
        Returns:
            List of improvement opportunities
        """
        improvements = []
        
        # Select files to analyze
        if not target_files:
            target_files = await self._select_files_for_improvement()
        
        # Analyze each file
        for file_path in target_files:
            if not os.path.exists(file_path):
                continue
            
            print(f"Analyzing {file_path} with AI...")
            
            try:
                # Get improvements with context
                if self.config['context_awareness_enabled']:
                    file_improvements = await self.context_improver.find_improvements_with_context(
                        file_path,
                        max_improvements=self.config['max_improvements_per_file']
                    )
                else:
                    # Simple analysis without context
                    file_improvements = await self.code_analyzer.analyze_file_with_context(
                        file_path
                    )
                
                # Convert to opportunity format
                for imp in file_improvements:
                    opportunity = self._improvement_to_opportunity(imp, file_path)
                    
                    # Score with learning system
                    if self.config['learning_enabled']:
                        learning_score = self.learning_system.score_improvement(
                            imp,
                            {'file_path': file_path}
                        )
                        opportunity['score'] = 0.7 * opportunity['score'] + 0.3 * learning_score
                    
                    improvements.append(opportunity)
                    
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        # Sort by score and priority
        improvements.sort(key=lambda x: x['score'] * x['priority'], reverse=True)
        
        return improvements[:max_improvements]
    
    async def _select_files_for_improvement(self) -> List[str]:
        """Intelligently select files for improvement."""
        candidates = []
        
        # Get research insights if available
        if self.config['research_integration_enabled']:
            insights = self.knowledge_store.get_insights('continuous_improvement')
            if insights:
                # Extract files mentioned in research
                for insight in insights:
                    if 'target_files' in insight.get('data', {}):
                        candidates.extend(insight['data']['target_files'])
        
        # Ask AI for suggestions
        prompt = """
Suggest Python files in this codebase that would benefit most from improvement.
Consider:
1. Files with high complexity
2. Files that are frequently modified
3. Core functionality files
4. Files with known issues

Base your suggestions on the codebase structure and return a list of file paths.
"""
        
        response_dict = await self.ai_brain.generate_enhanced_response(prompt)
        ai_response = response_dict.get('content', '')
        
        # Parse AI suggestions (simplified)
        lines = ai_response.strip().split('\n')
        for line in lines:
            if '.py' in line and '/' in line:
                # Extract file path from line
                parts = line.split()
                for part in parts:
                    if part.endswith('.py'):
                        candidates.append(part.strip('"\''))
        
        # Add files from context analysis
        context_report = self.context_improver.generate_context_report()
        for item in context_report.get('most_connected', [])[:5]:
            file_path = os.path.join(self.repo_path, item['file'])
            if os.path.exists(file_path):
                candidates.append(file_path)
        
        # Remove duplicates and validate
        valid_files = []
        seen = set()
        
        for file_path in candidates:
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.repo_path, file_path)
            
            if file_path not in seen and os.path.exists(file_path):
                valid_files.append(file_path)
                seen.add(file_path)
        
        return valid_files[:10]  # Limit to 10 files
    
    def _improvement_to_opportunity(self, improvement: CodeImprovement,
                                   file_path: str) -> Dict[str, Any]:
        """Convert CodeImprovement to opportunity format."""
        # Calculate priority based on type
        type_priority = self.config['improvement_priorities'].get(
            improvement.type.value.lower(),
            1.0
        )
        
        return {
            'file': file_path,
            'type': improvement.type,
            'description': improvement.description,
            'explanation': improvement.explanation,
            'original_code': improvement.original_code,
            'improved_code': improvement.improved_code,
            'line_start': improvement.line_start,
            'line_end': improvement.line_end,
            'score': improvement.confidence,
            'priority': type_priority,
            'impact_analysis': improvement.impact_analysis,
            'test_suggestions': improvement.test_suggestions
        }
    
    async def apply_intelligent_improvements(self, opportunities: List[Dict[str, Any]],
                                           auto_apply: bool = False) -> Dict[str, Any]:
        """Apply improvements intelligently with staging if enabled.
        
        Args:
            opportunities: List of improvement opportunities
            auto_apply: Whether to auto-apply high-confidence improvements
            
        Returns:
            Results of applying improvements
        """
        results = {
            'total': len(opportunities),
            'staged': 0,
            'validated': 0,
            'applied': 0,
            'auto_applied': 0,
            'failed': 0,
            'details': []
        }
        
        for opportunity in opportunities:
            try:
                # Create modification object
                modification = Modification(
                    file_path=opportunity['file'],
                    original_code=opportunity['original_code'],
                    modified_code=opportunity['improved_code'],
                    modification_type=opportunity['type'],
                    description=opportunity['description'],
                    safety_score=opportunity['score'],
                    line_number=opportunity['line_start']
                )
                
                if self.staging_enabled and self.staged_improver:
                    # Stage the improvement
                    staged = self.staged_improver.stage_improvement(modification)
                    if staged:
                        results['staged'] += 1
                        
                        # Validate
                        validation = await self.staged_improver.validate_staged_improvement(
                            staged.metadata['staging_id']
                        )
                        
                        if validation.get('ready_to_apply'):
                            results['validated'] += 1
                            
                            # Check if should auto-apply
                            should_apply = False
                            if auto_apply and opportunity['score'] >= self.config['auto_apply_threshold']:
                                risk_level = self.confidence_system.assess_risk_level(
                                    opportunity['type'],
                                    {'lines_changed': opportunity['line_end'] - opportunity['line_start']}
                                )
                                
                                should_auto, reason = self.confidence_system.should_auto_apply(
                                    opportunity['type'],
                                    risk_level
                                )
                                
                                if should_auto:
                                    should_apply = True
                                    results['auto_applied'] += 1
                            
                            if should_apply:
                                # Apply the improvement
                                applied = self.staged_improver.apply_staged_improvement(
                                    staged.metadata['staging_id']
                                )
                                if applied:
                                    results['applied'] += 1
                                    
                                    # Record outcome for learning
                                    if self.config['learning_enabled']:
                                        # Simple success tracking for now
                                        self.learning_system.record_outcome(
                                            self._opportunity_to_improvement(opportunity),
                                            success=True,
                                            metrics={'confidence': opportunity['score']}
                                        )
                else:
                    # Direct application without staging
                    # Would implement direct application here
                    pass
                
                results['details'].append({
                    'file': opportunity['file'],
                    'description': opportunity['description'],
                    'staged': results['staged'] > 0,
                    'validated': results['validated'] > 0,
                    'applied': results['applied'] > 0
                })
                
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'file': opportunity['file'],
                    'description': opportunity['description'],
                    'error': str(e)
                })
        
        return results
    
    def _opportunity_to_improvement(self, opportunity: Dict[str, Any]) -> CodeImprovement:
        """Convert opportunity back to CodeImprovement for learning system."""
        return CodeImprovement(
            type=opportunity['type'],
            description=opportunity['description'],
            original_code=opportunity['original_code'],
            improved_code=opportunity['improved_code'],
            explanation=opportunity.get('explanation', ''),
            confidence=opportunity['score'],
            line_start=opportunity['line_start'],
            line_end=opportunity['line_end'],
            impact_analysis=opportunity.get('impact_analysis', {}),
            test_suggestions=opportunity.get('test_suggestions', [])
        )
    
    async def run_improvement_cycle(self, max_improvements: int = 10,
                                  auto_apply: bool = False) -> Dict[str, Any]:
        """Run a complete improvement cycle.
        
        Args:
            max_improvements: Maximum improvements to process
            auto_apply: Whether to auto-apply high-confidence improvements
            
        Returns:
            Summary of the improvement cycle
        """
        print("🤖 Starting intelligent improvement cycle...")
        
        # Find improvements
        opportunities = await self.find_intelligent_improvements(
            max_improvements=max_improvements
        )
        
        if not opportunities:
            print("No improvement opportunities found.")
            return {
                'success': False,
                'reason': 'No improvements found',
                'opportunities': 0
            }
        
        print(f"Found {len(opportunities)} improvement opportunities:")
        for opp in opportunities[:5]:  # Show first 5
            print(f"  - {opp['type'].value}: {opp['description']} (score: {opp['score']:.2f})")
        
        # Apply improvements
        results = await self.apply_intelligent_improvements(
            opportunities,
            auto_apply=auto_apply
        )
        
        # Generate summary
        summary = {
            'success': True,
            'opportunities': len(opportunities),
            'staged': results['staged'],
            'validated': results['validated'],
            'applied': results['applied'],
            'auto_applied': results['auto_applied'],
            'failed': results['failed'],
            'confidence_score': self.confidence_system.metrics.confidence_score,
            'learning_insights': self._get_learning_insights()
        }
        
        print(f"\n✅ Improvement cycle complete:")
        print(f"   Opportunities: {summary['opportunities']}")
        print(f"   Staged: {summary['staged']}")
        print(f"   Validated: {summary['validated']}")
        print(f"   Applied: {summary['applied']} (auto: {summary['auto_applied']})")
        
        return summary
    
    def _get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system."""
        if not self.config['learning_enabled']:
            return {}
        
        report = self.learning_system.generate_learning_report()
        
        return {
            'total_learned': report['summary']['patterns_learned'],
            'success_rate': report['summary']['overall_success_rate'],
            'best_patterns': [
                p['pattern'] for p in report.get('best_patterns', [])[:3]
            ],
            'trend': report.get('recent_trend', {}).get('trend', 'unknown')
        }
    
    def generate_intelligence_report(self) -> Dict[str, Any]:
        """Generate a comprehensive intelligence report."""
        report = {
            'configuration': {
                'min_confidence': self.config['min_confidence'],
                'auto_apply_threshold': self.config['auto_apply_threshold'],
                'context_awareness': self.config['context_awareness_enabled'],
                'learning_enabled': self.config['learning_enabled']
            },
            'confidence_metrics': {
                'score': self.confidence_system.metrics.confidence_score,
                'total_improvements': self.confidence_system.metrics.total_improvements,
                'successful': self.confidence_system.metrics.successful_improvements,
                'can_auto_apply': self.confidence_system.metrics.total_improvements >= 5
            }
        }
        
        # Add context analysis if available
        if self.config['context_awareness_enabled']:
            report['context_analysis'] = self.context_improver.generate_context_report()
        
        # Add learning insights
        if self.config['learning_enabled']:
            report['learning_insights'] = self.learning_system.generate_learning_report()
        
        # Add staging report if enabled
        if self.staging_enabled and self.staged_improver:
            report['staging_status'] = self.staged_improver.generate_staging_report()
        
        return report