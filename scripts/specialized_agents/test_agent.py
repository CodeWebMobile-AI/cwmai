"""
TestAgent - Testing and Quality Assurance Expert

Specializes in creating comprehensive test suites, providing testability feedback,
and ensuring code quality through various testing methodologies.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import sys
sys.path.append('..')

from base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class TestAgent(BaseAgent):
    """Agent specialized in testing and quality assurance."""
    
    @property
    def agent_type(self) -> str:
        return "tester"
    
    @property
    def persona(self) -> str:
        return """You are a seasoned QA engineer and testing expert with deep knowledge of 
        testing methodologies, frameworks, and best practices. You think like both a user 
        and a hacker to find edge cases and potential issues. You excel at writing 
        comprehensive test suites including unit tests, integration tests, end-to-end tests, 
        and performance tests. You provide constructive feedback on code testability and 
        always advocate for high test coverage and quality standards."""
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.TESTING,
            AgentCapability.CODE_REVIEW,
            AgentCapability.ANALYSIS
        ]
    
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the task from a testing perspective."""
        work_item = context.work_item
        
        # Check for existing code
        main_code = context.get_artifact('main_code')
        
        prompt = f"""
        Analyze this task from a testing and QA perspective:
        
        Task: {work_item.title}
        Description: {work_item.description}
        Type: {work_item.task_type}
        
        {f"Code to test: {json.dumps(main_code, indent=2)}" if main_code else ""}
        
        Provide analysis including:
        1. Testing strategy and approach
        2. Test types needed (unit, integration, e2e, performance, security)
        3. Testing frameworks to use
        4. Critical test scenarios
        5. Edge cases to consider
        6. Testability assessment
        7. Coverage targets
        8. Performance benchmarks
        
        Format as JSON with keys: testing_strategy, test_types, frameworks, 
        critical_scenarios, edge_cases, testability_assessment, 
        coverage_targets, performance_benchmarks
        """
        
        response = await self._call_ai_model(prompt)
        
        expected_format = {
            'testing_strategy': str,
            'test_types': list,
            'frameworks': dict,
            'critical_scenarios': list,
            'edge_cases': list,
            'testability_assessment': dict,
            'coverage_targets': dict,
            'performance_benchmarks': dict
        }
        
        return self._parse_ai_response(response, expected_format)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Create comprehensive test suite for the task."""
        start_time = time.time()
        
        try:
            # Analyze the task
            analysis = await self.analyze(context)
            
            # Get code artifacts if available
            main_code = context.get_artifact('main_code')
            
            # Generate test suite
            test_suite = await self._generate_test_suite(context, analysis, main_code)
            
            # Provide testability feedback
            testability_feedback = await self._analyze_testability(context, main_code, analysis)
            
            # Create test plan
            test_plan = await self._create_test_plan(context, analysis, test_suite)
            
            # Generate test data
            test_data = await self._generate_test_data(context, test_suite)
            
            # Store artifacts
            artifacts_created = []
            
            if context.blackboard:
                # Store in blackboard
                await context.blackboard.write_artifact(
                    f"test_suite_{context.work_item.id}",
                    test_suite,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"testability_feedback_{context.work_item.id}",
                    testability_feedback,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"test_plan_{context.work_item.id}",
                    test_plan,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"test_data_{context.work_item.id}",
                    test_data,
                    self.agent_id
                )
                artifacts_created = [
                    f"test_suite_{context.work_item.id}",
                    f"testability_feedback_{context.work_item.id}",
                    f"test_plan_{context.work_item.id}",
                    f"test_data_{context.work_item.id}"
                ]
            else:
                # Store in context
                context.add_artifact('test_suite', test_suite, self.agent_id)
                context.add_artifact('testability_feedback', testability_feedback, self.agent_id)
                context.add_artifact('test_plan', test_plan, self.agent_id)
                context.add_artifact('test_data', test_data, self.agent_id)
                artifacts_created = ['test_suite', 'testability_feedback', 'test_plan', 'test_data']
            
            # Generate insights
            insights = [
                f"Generated {len(test_suite.get('tests', []))} test cases",
                f"Coverage target: {analysis.get('coverage_targets', {}).get('overall', 'Unknown')}",
                f"Identified {len(analysis.get('edge_cases', []))} edge cases",
                f"Testability score: {testability_feedback.get('overall_score', 'N/A')}/10"
            ]
            
            # Generate recommendations
            recommendations = []
            if testability_feedback.get('overall_score', 10) < 7:
                recommendations.append("Refactor code to improve testability")
            if len(testability_feedback.get('issues', [])) > 0:
                recommendations.append(f"Address {len(testability_feedback.get('issues', []))} testability issues")
            if main_code and not main_code:
                recommendations.append("Waiting for CodeAgent to generate code before creating specific tests")
            recommendations.append("Set up continuous integration to run tests automatically")
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=True,
                output={
                    'test_suite': test_suite,
                    'testability_feedback': testability_feedback,
                    'test_plan': test_plan,
                    'test_data': test_data,
                    'analysis': analysis
                },
                artifacts_created=artifacts_created,
                insights=insights,
                recommendations=recommendations,
                confidence=0.85,
                execution_time=execution_time,
                metadata={
                    'tests_count': len(test_suite.get('tests', [])),
                    'testability_score': testability_feedback.get('overall_score', 0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"TestAgent execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                output={'error': str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def _generate_test_suite(self, context: AgentContext, analysis: Dict[str, Any], 
                                 main_code: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive test suite."""
        prompt = f"""
        Generate a comprehensive test suite for:
        Task: {context.work_item.title}
        
        Testing strategy: {analysis.get('testing_strategy')}
        Test types needed: {json.dumps(analysis.get('test_types', []))}
        Frameworks: {json.dumps(analysis.get('frameworks', {}))}
        
        {f"Code to test: {json.dumps(main_code, indent=2)}" if main_code else "Generate generic test structure"}
        
        Create tests including:
        1. Unit tests for individual functions/methods
        2. Integration tests for component interactions
        3. Edge case tests
        4. Error handling tests
        5. Performance tests (if applicable)
        6. Security tests (if applicable)
        
        Format as JSON with structure:
        {{
            "tests": [
                {{
                    "test_id": "...",
                    "name": "...",
                    "type": "unit/integration/e2e/performance/security",
                    "description": "...",
                    "code": "...",
                    "assertions": [...],
                    "expected_result": "...",
                    "edge_cases": [...]
                }}
            ],
            "setup_code": "...",
            "teardown_code": "...",
            "test_utilities": [...],
            "mocks_needed": [...]
        }}
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            test_suite = json.loads(response)
            return test_suite
        except:
            # Fallback test structure
            return {
                "tests": [
                    {
                        "test_id": "test_001",
                        "name": "test_basic_functionality",
                        "type": "unit",
                        "description": "Test basic functionality",
                        "code": "def test_basic():\n    assert True",
                        "assertions": ["Basic assertion"],
                        "expected_result": "Pass",
                        "edge_cases": []
                    }
                ],
                "setup_code": "# Test setup",
                "teardown_code": "# Test teardown",
                "test_utilities": [],
                "mocks_needed": []
            }
    
    async def _analyze_testability(self, context: AgentContext, main_code: Optional[Dict[str, Any]], 
                                 analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code testability and provide feedback."""
        if not main_code:
            return {
                "overall_score": 5,
                "issues": ["No code available to analyze"],
                "suggestions": ["Waiting for code generation"],
                "testable_components": [],
                "untestable_components": []
            }
        
        prompt = f"""
        Analyze the testability of this code:
        {json.dumps(main_code, indent=2)}
        
        Evaluate:
        1. Code modularity and separation of concerns
        2. Dependency injection and mocking capabilities
        3. Side effects and external dependencies
        4. Function/method complexity
        5. Clear inputs and outputs
        6. Error handling testability
        
        Provide feedback as JSON with:
        - overall_score (1-10)
        - issues (list of testability problems)
        - suggestions (list of improvements)
        - testable_components (easily testable parts)
        - untestable_components (hard to test parts)
        - refactoring_recommendations
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            feedback = json.loads(response)
            return feedback
        except:
            return {
                "overall_score": 7,
                "issues": [],
                "suggestions": ["Consider dependency injection", "Reduce function complexity"],
                "testable_components": ["Pure functions"],
                "untestable_components": ["External API calls"],
                "refactoring_recommendations": []
            }
    
    async def _create_test_plan(self, context: AgentContext, analysis: Dict[str, Any], 
                              test_suite: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed test execution plan."""
        test_plan = {
            "phases": [
                {
                    "phase": "Unit Testing",
                    "tests": [t for t in test_suite.get('tests', []) if t.get('type') == 'unit'],
                    "priority": "High",
                    "estimated_time": "2 hours"
                },
                {
                    "phase": "Integration Testing",
                    "tests": [t for t in test_suite.get('tests', []) if t.get('type') == 'integration'],
                    "priority": "High",
                    "estimated_time": "3 hours"
                },
                {
                    "phase": "End-to-End Testing",
                    "tests": [t for t in test_suite.get('tests', []) if t.get('type') == 'e2e'],
                    "priority": "Medium",
                    "estimated_time": "4 hours"
                },
                {
                    "phase": "Performance Testing",
                    "tests": [t for t in test_suite.get('tests', []) if t.get('type') == 'performance'],
                    "priority": "Low",
                    "estimated_time": "2 hours"
                }
            ],
            "execution_order": ["Unit", "Integration", "E2E", "Performance"],
            "environment_requirements": analysis.get('frameworks', {}),
            "coverage_goals": analysis.get('coverage_targets', {}),
            "risk_areas": analysis.get('edge_cases', [])
        }
        
        return test_plan
    
    async def _generate_test_data(self, context: AgentContext, test_suite: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test data for the test suite."""
        test_data = {
            "valid_inputs": [
                {"type": "normal", "data": {}, "description": "Standard valid input"},
                {"type": "boundary", "data": {}, "description": "Boundary value input"}
            ],
            "invalid_inputs": [
                {"type": "null", "data": None, "description": "Null input"},
                {"type": "malformed", "data": {}, "description": "Malformed input"}
            ],
            "edge_cases": [
                {"type": "empty", "data": {}, "description": "Empty input"},
                {"type": "large", "data": {}, "description": "Very large input"}
            ],
            "performance_data": [
                {"type": "load", "data": {}, "description": "Load test data"},
                {"type": "stress", "data": {}, "description": "Stress test data"}
            ]
        }
        
        return test_data
    
    async def review_artifact(self, artifact_key: str, artifact_value: Any, 
                            created_by: str, context: AgentContext) -> Dict[str, Any]:
        """Review artifacts from other agents from a testing perspective."""
        review = await super().review_artifact(artifact_key, artifact_value, created_by, context)
        
        # Specific test reviews
        if 'code' in artifact_key:
            review['feedback'].append("Code appears testable but needs comprehensive test coverage")
            review['feedback'].append("Consider adding more error handling for better test scenarios")
            review['approval'] = False  # Don't approve code without tests
        elif 'security' in artifact_key:
            review['feedback'].append("Security findings should be covered by security tests")
        elif 'docs' in artifact_key:
            review['feedback'].append("Documentation should include testing guidelines")
        
        review['confidence'] = 0.9
        return review