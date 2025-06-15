"""
Experiment Designer

Automatically designs and proposes experiments to test new capabilities,
validate improvements, and explore system boundaries.
"""

import os
import json
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import numpy as np


class ExperimentDesigner:
    """Designs experiments for system improvement."""
    
    def __init__(self, ai_brain, capability_analyzer, outcome_learning):
        """Initialize experiment designer.
        
        Args:
            ai_brain: AI brain for experiment design
            capability_analyzer: System capability analyzer
            outcome_learning: Outcome learning system
        """
        self.ai_brain = ai_brain
        self.capability_analyzer = capability_analyzer
        self.outcome_learning = outcome_learning
        self.experiment_queue = []
        self.running_experiments = {}
        self.experiment_history = []
        self.hypothesis_bank = []
        self.experiment_templates = self._load_experiment_templates()
        
    def _load_experiment_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load experiment templates.
        
        Returns:
            Experiment templates
        """
        return {
            'capability_test': {
                'name': 'Capability Test',
                'purpose': 'Test new capability effectiveness',
                'variables': ['input_complexity', 'load_level', 'integration_depth'],
                'metrics': ['success_rate', 'performance', 'resource_usage'],
                'duration': 'short'
            },
            'performance_benchmark': {
                'name': 'Performance Benchmark',
                'purpose': 'Measure performance improvements',
                'variables': ['task_type', 'scale', 'concurrency'],
                'metrics': ['latency', 'throughput', 'accuracy'],
                'duration': 'medium'
            },
            'integration_test': {
                'name': 'Integration Test',
                'purpose': 'Test component integration',
                'variables': ['components', 'data_flow', 'error_conditions'],
                'metrics': ['compatibility', 'data_integrity', 'error_handling'],
                'duration': 'medium'
            },
            'stress_test': {
                'name': 'Stress Test',
                'purpose': 'Find system limits',
                'variables': ['load_factor', 'duration', 'resource_constraints'],
                'metrics': ['breaking_point', 'recovery_time', 'degradation_curve'],
                'duration': 'long'
            },
            'ab_test': {
                'name': 'A/B Test',
                'purpose': 'Compare approaches',
                'variables': ['approach_a', 'approach_b', 'test_conditions'],
                'metrics': ['effectiveness', 'efficiency', 'user_preference'],
                'duration': 'long'
            }
        }
    
    async def design_experiment(self,
                               hypothesis: str,
                               experiment_type: str = None) -> Dict[str, Any]:
        """Design an experiment to test a hypothesis.
        
        Args:
            hypothesis: Hypothesis to test
            experiment_type: Type of experiment (optional)
            
        Returns:
            Experiment design
        """
        print(f"Designing experiment for: {hypothesis}")
        
        # Select experiment type if not specified
        if not experiment_type:
            experiment_type = await self._select_experiment_type(hypothesis)
        
        # Get template
        template = self.experiment_templates.get(
            experiment_type, 
            self.experiment_templates['capability_test']
        )
        
        # Design experiment
        design = await self._create_experiment_design(hypothesis, template)
        
        # Validate design
        validation = await self._validate_experiment_design(design)
        
        if validation['valid']:
            design['validation'] = validation
            design['id'] = self._generate_experiment_id(hypothesis)
            design['status'] = 'designed'
            design['created'] = datetime.now(timezone.utc).isoformat()
            
            # Add to queue
            self.experiment_queue.append(design)
            
            return design
        else:
            return {
                'error': 'Invalid experiment design',
                'issues': validation['issues']
            }
    
    async def generate_hypothesis_bank(self) -> List[Dict[str, Any]]:
        """Generate hypotheses for experimentation.
        
        Returns:
            List of hypotheses
        """
        # Get system state
        capabilities = await self.capability_analyzer.analyze_current_capabilities()
        gaps = await self.capability_analyzer.identify_gaps()
        
        # Get learning insights
        learning_insights = {}
        if self.outcome_learning:
            learning_insights = await self.outcome_learning.get_recommendations()
        
        prompt = f"""
        Generate experimental hypotheses for system improvement:
        
        Current Capabilities:
        {json.dumps(capabilities.get('capability_coverage', {}), indent=2)}
        
        Identified Gaps:
        {json.dumps(gaps.get('missing_capabilities', [])[:5], indent=2)}
        
        Learning Insights:
        {json.dumps(learning_insights, indent=2)}
        
        Generate 5-7 testable hypotheses that could:
        1. Validate new capabilities
        2. Improve existing functions
        3. Explore novel combinations
        4. Test system boundaries
        5. Optimize performance
        
        For each hypothesis provide:
        - statement: Clear hypothesis statement
        - rationale: Why this is worth testing
        - expected_impact: Potential system improvement
        - risk_level: low/medium/high
        - experiment_type: Suggested experiment type
        
        Format as JSON array.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        hypotheses = self._parse_json_response(response)
        
        if isinstance(hypotheses, list):
            self.hypothesis_bank.extend(hypotheses)
            return hypotheses
        
        return []
    
    async def _select_experiment_type(self, hypothesis: str) -> str:
        """Select appropriate experiment type for hypothesis.
        
        Args:
            hypothesis: Hypothesis to test
            
        Returns:
            Experiment type
        """
        prompt = f"""
        Select the best experiment type for this hypothesis:
        
        Hypothesis: {hypothesis}
        
        Available Types:
        {json.dumps({
            k: {'purpose': v['purpose'], 'duration': v['duration']}
            for k, v in self.experiment_templates.items()
        }, indent=2)}
        
        Return the experiment type key that best matches.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        content = response.get('content', '').strip().lower()
        
        # Find matching type
        for exp_type in self.experiment_templates:
            if exp_type in content:
                return exp_type
        
        return 'capability_test'  # Default
    
    async def _create_experiment_design(self,
                                       hypothesis: str,
                                       template: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed experiment design.
        
        Args:
            hypothesis: Hypothesis to test
            template: Experiment template
            
        Returns:
            Experiment design
        """
        prompt = f"""
        Create detailed experiment design:
        
        Hypothesis: {hypothesis}
        Template: {json.dumps(template, indent=2)}
        
        Design should include:
        1. Specific test cases (at least 3)
        2. Control variables
        3. Success criteria
        4. Resource requirements
        5. Risk mitigation
        6. Data collection plan
        
        Format as JSON with:
        - name: Experiment name
        - hypothesis: The hypothesis
        - methodology: How to conduct experiment
        - test_cases: List of specific test cases
        - controls: Control variables
        - metrics: What to measure
        - success_criteria: Definition of success
        - resources_needed: Required resources
        - risks: Potential risks and mitigations
        - expected_duration: Time estimate
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        design = self._parse_json_response(response)
        
        # Add template info
        design['template'] = template['name']
        design['experiment_type'] = template['purpose']
        
        return design
    
    async def _validate_experiment_design(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment design.
        
        Args:
            design: Experiment design
            
        Returns:
            Validation result
        """
        issues = []
        
        # Check required fields
        required_fields = ['name', 'hypothesis', 'methodology', 'test_cases', 'metrics']
        for field in required_fields:
            if field not in design or not design[field]:
                issues.append(f"Missing required field: {field}")
        
        # Check test cases
        if 'test_cases' in design:
            if not isinstance(design['test_cases'], list):
                issues.append("Test cases must be a list")
            elif len(design['test_cases']) < 2:
                issues.append("Need at least 2 test cases")
        
        # Check resources
        if 'resources_needed' in design:
            resources = design['resources_needed']
            if isinstance(resources, dict):
                for resource, amount in resources.items():
                    if not isinstance(amount, (int, float)) or amount < 0:
                        issues.append(f"Invalid resource amount for {resource}")
        
        # AI validation
        ai_validation = await self._ai_validate_design(design)
        issues.extend(ai_validation.get('issues', []))
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'suggestions': ai_validation.get('suggestions', [])
        }
    
    async def _ai_validate_design(self, design: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to validate experiment design.
        
        Args:
            design: Experiment design
            
        Returns:
            AI validation result
        """
        prompt = f"""
        Validate this experiment design:
        
        {json.dumps(design, indent=2)}
        
        Check for:
        1. Logical consistency
        2. Testability
        3. Appropriate scope
        4. Clear success criteria
        5. Adequate controls
        
        Return validation as JSON with:
        - issues: List of problems found
        - suggestions: Improvements to make
        - risk_assessment: Overall risk level
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    async def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run an experiment.
        
        Args:
            experiment_id: Experiment to run
            
        Returns:
            Experiment execution result
        """
        # Find experiment
        experiment = next(
            (e for e in self.experiment_queue if e['id'] == experiment_id),
            None
        )
        
        if not experiment:
            return {'error': 'Experiment not found'}
        
        print(f"Running experiment: {experiment['name']}")
        
        # Move to running
        self.experiment_queue.remove(experiment)
        experiment['status'] = 'running'
        experiment['start_time'] = datetime.now(timezone.utc)
        self.running_experiments[experiment_id] = experiment
        
        # Execute test cases
        results = {
            'test_results': [],
            'metrics': defaultdict(list),
            'observations': []
        }
        
        for i, test_case in enumerate(experiment['test_cases']):
            print(f"Executing test case {i+1}/{len(experiment['test_cases'])}")
            
            test_result = await self._execute_test_case(
                test_case, 
                experiment['methodology']
            )
            
            results['test_results'].append(test_result)
            
            # Collect metrics
            for metric, value in test_result.get('metrics', {}).items():
                results['metrics'][metric].append(value)
        
        # Analyze results
        analysis = await self._analyze_experiment_results(experiment, results)
        
        # Complete experiment
        experiment['status'] = 'completed'
        experiment['end_time'] = datetime.now(timezone.utc)
        experiment['results'] = results
        experiment['analysis'] = analysis
        
        # Move to history
        del self.running_experiments[experiment_id]
        self.experiment_history.append(experiment)
        
        # Learn from results
        if self.outcome_learning:
            await self.outcome_learning.record_outcome(
                {'type': 'experiment', 'experiment': experiment},
                {'success': analysis.get('hypothesis_supported', False)}
            )
        
        return {
            'experiment_id': experiment_id,
            'hypothesis_supported': analysis.get('hypothesis_supported', False),
            'key_findings': analysis.get('key_findings', []),
            'recommendations': analysis.get('recommendations', [])
        }
    
    async def _execute_test_case(self,
                                test_case: Dict[str, Any],
                                methodology: str) -> Dict[str, Any]:
        """Execute a single test case.
        
        Args:
            test_case: Test case to execute
            methodology: Experiment methodology
            
        Returns:
            Test case results
        """
        # Simulate test execution
        # In production, would actually run the test
        
        result = {
            'test_case': test_case,
            'executed': datetime.now(timezone.utc).isoformat(),
            'metrics': {},
            'observations': []
        }
        
        # Simulate metrics
        if isinstance(test_case, dict):
            # Generate realistic metrics based on test type
            result['metrics'] = {
                'success_rate': random.uniform(0.6, 0.95),
                'performance': random.uniform(0.5, 1.0),
                'resource_usage': random.uniform(0.3, 0.8)
            }
            
            # Simulate observations
            result['observations'] = [
                f"Test case completed successfully",
                f"Performance within expected range"
            ]
        
        return result
    
    async def _analyze_experiment_results(self,
                                        experiment: Dict[str, Any],
                                        results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze experiment results.
        
        Args:
            experiment: Experiment design
            results: Experiment results
            
        Returns:
            Analysis
        """
        prompt = f"""
        Analyze experiment results:
        
        Experiment:
        - Hypothesis: {experiment['hypothesis']}
        - Success Criteria: {json.dumps(experiment.get('success_criteria', {}), indent=2)}
        
        Results:
        {json.dumps(results, indent=2)}
        
        Provide analysis:
        1. Was the hypothesis supported? (true/false)
        2. Key findings (list)
        3. Unexpected observations
        4. Statistical significance (if applicable)
        5. Recommendations for next steps
        6. Implications for the system
        
        Format as JSON.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        analysis = self._parse_json_response(response)
        
        # Add statistical analysis
        if results['metrics']:
            analysis['statistics'] = self._calculate_statistics(results['metrics'])
        
        return analysis
    
    def _calculate_statistics(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate basic statistics for metrics.
        
        Args:
            metrics: Metric data
            
        Returns:
            Statistics
        """
        stats = {}
        
        for metric, values in metrics.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'samples': len(values)
                }
        
        return stats
    
    async def propose_followup_experiments(self,
                                         experiment_id: str) -> List[Dict[str, Any]]:
        """Propose follow-up experiments based on results.
        
        Args:
            experiment_id: Completed experiment
            
        Returns:
            Follow-up experiment proposals
        """
        # Find experiment
        experiment = next(
            (e for e in self.experiment_history if e['id'] == experiment_id),
            None
        )
        
        if not experiment:
            return []
        
        prompt = f"""
        Propose follow-up experiments based on these results:
        
        Original Hypothesis: {experiment['hypothesis']}
        
        Results Summary:
        - Hypothesis Supported: {experiment['analysis'].get('hypothesis_supported', 'Unknown')}
        - Key Findings: {json.dumps(experiment['analysis'].get('key_findings', []), indent=2)}
        - Unexpected Observations: {json.dumps(
            experiment['results'].get('observations', [])[:5], 
            indent=2
        )}
        
        Propose 2-3 follow-up experiments that:
        1. Dig deeper into interesting findings
        2. Test edge cases discovered
        3. Validate unexpected results
        4. Extend successful approaches
        5. Address limitations found
        
        Format as JSON array with same structure as hypotheses.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        followups = self._parse_json_response(response)
        
        if isinstance(followups, list):
            # Link to parent experiment
            for followup in followups:
                followup['parent_experiment'] = experiment_id
            
            return followups
        
        return []
    
    def get_experiment_queue(self) -> List[Dict[str, Any]]:
        """Get pending experiments.
        
        Returns:
            Experiment queue
        """
        return [
            {
                'id': e['id'],
                'name': e['name'],
                'hypothesis': e['hypothesis'],
                'type': e.get('template', 'Unknown'),
                'created': e['created']
            }
            for e in self.experiment_queue
        ]
    
    def get_experiment_insights(self) -> Dict[str, Any]:
        """Get insights from all experiments.
        
        Returns:
            Experiment insights
        """
        if not self.experiment_history:
            return {'status': 'No experiments completed yet'}
        
        total = len(self.experiment_history)
        supported = sum(
            1 for e in self.experiment_history 
            if e['analysis'].get('hypothesis_supported', False)
        )
        
        # Group by type
        by_type = defaultdict(lambda: {'total': 0, 'supported': 0})
        for exp in self.experiment_history:
            exp_type = exp.get('template', 'Unknown')
            by_type[exp_type]['total'] += 1
            if exp['analysis'].get('hypothesis_supported', False):
                by_type[exp_type]['supported'] += 1
        
        return {
            'total_experiments': total,
            'hypotheses_supported': supported,
            'support_rate': supported / total if total > 0 else 0,
            'by_type': dict(by_type),
            'recent_findings': self._get_recent_findings(),
            'top_recommendations': self._get_top_recommendations()
        }
    
    def _get_recent_findings(self, n: int = 5) -> List[str]:
        """Get recent key findings.
        
        Args:
            n: Number of findings
            
        Returns:
            Recent findings
        """
        findings = []
        
        for exp in reversed(self.experiment_history[-10:]):
            findings.extend(exp['analysis'].get('key_findings', []))
        
        return findings[:n]
    
    def _get_top_recommendations(self, n: int = 5) -> List[str]:
        """Get top recommendations from experiments.
        
        Args:
            n: Number of recommendations
            
        Returns:
            Top recommendations
        """
        all_recommendations = []
        
        for exp in self.experiment_history:
            recommendations = exp['analysis'].get('recommendations', [])
            all_recommendations.extend(recommendations)
        
        # In production, would rank by importance
        return all_recommendations[-n:] if all_recommendations else []
    
    def _generate_experiment_id(self, hypothesis: str) -> str:
        """Generate unique experiment ID.
        
        Args:
            hypothesis: Experiment hypothesis
            
        Returns:
            Experiment ID
        """
        content = f"{hypothesis}{datetime.now().isoformat()}"
        return f"exp_{hashlib.md5(content.encode()).hexdigest()[:8]}"
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Any:
        """Parse JSON from AI response."""
        content = response.get('content', '')
        
        try:
            import re
            
            # Look for JSON array
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                return json.loads(array_match.group())
            
            # Look for JSON object
            obj_match = re.search(r'\{[\s\S]*\}', content)
            if obj_match:
                return json.loads(obj_match.group())
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error parsing JSON: {e}")
        
        return {}


async def demonstrate_experiment_designer():
    """Demonstrate experiment design."""
    print("=== Experiment Designer Demo ===\n")
    
    # Mock AI brain
    class MockAIBrain:
        async def generate_enhanced_response(self, prompt):
            if "Generate experimental hypotheses" in prompt:
                return {
                    'content': '''[
                        {
                            "statement": "Combining task generation with outcome learning will improve task quality by 40%",
                            "rationale": "Learning from outcomes can guide better task generation",
                            "expected_impact": "Significant improvement in task success rates",
                            "risk_level": "low",
                            "experiment_type": "ab_test"
                        }
                    ]'''
                }
            elif "Create detailed experiment design" in prompt:
                return {
                    'content': '''{
                        "name": "Task Generation Enhancement Test",
                        "hypothesis": "Learning-enhanced task generation improves quality",
                        "methodology": "Compare standard vs learning-enhanced generation",
                        "test_cases": [
                            {"name": "baseline", "approach": "standard"},
                            {"name": "enhanced", "approach": "with_learning"}
                        ],
                        "metrics": ["task_quality", "success_rate"],
                        "success_criteria": {"improvement": 0.3}
                    }'''
                }
            return {'content': '{}'}
    
    ai_brain = MockAIBrain()
    designer = ExperimentDesigner(ai_brain, None, None)
    
    # Generate hypotheses
    print("Generating experimental hypotheses...")
    hypotheses = await designer.generate_hypothesis_bank()
    
    print(f"\nGenerated {len(hypotheses)} hypotheses:")
    for hyp in hypotheses:
        print(f"- {hyp['statement']}")
    
    # Design experiment
    if hypotheses:
        print(f"\nDesigning experiment for first hypothesis...")
        experiment = await designer.design_experiment(hypotheses[0]['statement'])
        
        if 'error' not in experiment:
            print(f"Experiment designed: {experiment['name']}")
            print(f"ID: {experiment['id']}")
            
            # Run experiment
            print("\nRunning experiment...")
            result = await designer.run_experiment(experiment['id'])
            
            print(f"Hypothesis supported: {result['hypothesis_supported']}")
            print(f"Key findings: {result['key_findings']}")
    
    # Show insights
    print("\n=== Experiment Insights ===")
    insights = designer.get_experiment_insights()
    print(f"Total experiments: {insights.get('total_experiments', 0)}")
    print(f"Support rate: {insights.get('support_rate', 0):.0%}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_experiment_designer())