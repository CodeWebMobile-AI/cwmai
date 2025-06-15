"""
Test Suite for External Learning System

Comprehensive tests for all components of the external learning system:
- ExternalAgentDiscoverer
- CapabilityExtractor  
- CapabilitySynthesizer
- ExternalKnowledgeIntegrator
- Enhanced ResearchEvolutionEngine
- SafeSelfImprover external integration
- Production orchestrator external cycles
"""

import asyncio
import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from typing import Dict, List, Any

# Import CWMAI external learning components
import sys
sys.path.append('scripts')

from external_agent_discoverer import ExternalAgentDiscoverer, DiscoveryConfig, RepositoryAnalysis, CapabilityType
from capability_extractor import CapabilityExtractor, ExtractedCapability, ExtractionResult, IntegrationComplexity
from capability_synthesizer import CapabilitySynthesizer, SynthesizedCapability, SynthesisStrategy, SynthesisComplexity
from external_knowledge_integrator import ExternalKnowledgeIntegrator, IntegrationPlan, IntegrationResult, IntegrationStrategy
from research_evolution_engine import ResearchEvolutionEngine
from safe_self_improver import SafeSelfImprover, ModificationType
from production_orchestrator import ProductionOrchestrator
from state_manager import StateManager


class TestExternalAgentDiscoverer(unittest.TestCase):
    """Test ExternalAgentDiscoverer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = DiscoveryConfig(
            max_repositories_per_scan=5,
            min_stars=1,
            search_topics=['test-topic']
        )
        self.discoverer = ExternalAgentDiscoverer(self.config)
        
    def test_discovery_config_initialization(self):
        """Test discovery configuration initialization."""
        self.assertEqual(self.config.max_repositories_per_scan, 5)
        self.assertEqual(self.config.min_stars, 1)
        self.assertIn('test-topic', self.config.search_topics)
        
    def test_repository_analysis_creation(self):
        """Test repository analysis data structure."""
        analysis = RepositoryAnalysis(
            url="https://github.com/test/repo",
            name="test-repo",
            description="Test repository",
            language="Python",
            stars=100,
            forks=20,
            last_updated="2024-01-01",
            health_score=0.8,
            capabilities=[CapabilityType.TASK_ORCHESTRATION],
            architecture_patterns=['plugin_architecture'],
            key_files=[],
            integration_difficulty=0.3,
            license="MIT",
            documentation_quality=0.7,
            test_coverage=0.6,
            performance_indicators={},
            security_assessment={},
            compatibility_score=0.8
        )
        
        self.assertEqual(analysis.name, "test-repo")
        self.assertEqual(analysis.stars, 100)
        self.assertIn(CapabilityType.TASK_ORCHESTRATION, analysis.capabilities)
        
    @patch('requests.get')
    async def test_repository_search_mock(self, mock_get):
        """Test repository search with mocked API."""
        # Mock GitHub API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'items': [
                {
                    'html_url': 'https://github.com/test/repo1',
                    'name': 'repo1',
                    'description': 'Test repo 1',
                    'language': 'Python',
                    'stargazers_count': 50,
                    'forks_count': 10
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test search
        results = await self.discoverer._search_github_topic('test-topic')
        self.assertIsInstance(results, list)
        
    def test_compatibility_score_calculation(self):
        """Test compatibility score calculation."""
        analysis = RepositoryAnalysis(
            url="https://github.com/test/repo",
            name="test-repo", 
            description="Test",
            language="Python",  # Compatible language
            stars=100,
            forks=20,
            last_updated="2024-01-01",
            health_score=0.8,
            capabilities=[CapabilityType.TASK_ORCHESTRATION],  # High-value capability
            architecture_patterns=['plugin_architecture'],  # Compatible pattern
            key_files=[],
            integration_difficulty=0.3,
            license="MIT",
            documentation_quality=0.7,
            test_coverage=0.6,
            performance_indicators={},
            security_assessment={},
            compatibility_score=0.0  # Will be calculated
        )
        
        score = self.discoverer._calculate_compatibility_score(analysis)
        self.assertGreater(score, 0.5)  # Should be high due to Python + good capability


class TestCapabilityExtractor(unittest.TestCase):
    """Test CapabilityExtractor functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.extractor = CapabilityExtractor()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    def test_capability_patterns_initialization(self):
        """Test capability detection patterns initialization."""
        patterns = self.extractor.capability_patterns
        self.assertIn(CapabilityType.TASK_ORCHESTRATION, patterns)
        self.assertIn(CapabilityType.MULTI_AGENT_COORDINATION, patterns)
        
    def test_extracted_capability_creation(self):
        """Test ExtractedCapability data structure."""
        capability = ExtractedCapability(
            id="test_cap_1",
            name="Test Capability",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="Test capability for task orchestration",
            source_repository="https://github.com/test/repo",
            source_files=["test.py"],
            extraction_method="pattern_matching",
            integration_complexity=IntegrationComplexity.SIMPLE
        )
        
        self.assertEqual(capability.name, "Test Capability")
        self.assertEqual(capability.capability_type, CapabilityType.TASK_ORCHESTRATION)
        self.assertEqual(capability.integration_complexity, IntegrationComplexity.SIMPLE)
        
    def test_file_analysis(self):
        """Test file analysis functionality."""
        # Create test Python file
        test_file = os.path.join(self.temp_dir, "test_agent.py")
        with open(test_file, 'w') as f:
            f.write("""
class TaskManager:
    def __init__(self):
        self.tasks = []
        
    def add_task(self, task):
        self.tasks.append(task)
        
    def execute_task(self, task_id):
        # Execute task logic
        pass
""")
        
        # Test pattern detection
        with open(test_file, 'r') as f:
            content = f.read()
            
        # Should detect task orchestration patterns
        patterns = self.extractor.capability_patterns[CapabilityType.TASK_ORCHESTRATION]
        found_patterns = []
        for pattern_def in patterns:
            for pattern in pattern_def.get('patterns', []):
                matches = self.extractor._apply_pattern(content, {'type': 'regex', 'patterns': [pattern]})
                if matches:
                    found_patterns.append(pattern)
                    
        self.assertGreater(len(found_patterns), 0)
        
    def test_integration_complexity_assessment(self):
        """Test integration complexity assessment."""
        # Simple capability
        simple_cap = ExtractedCapability(
            id="simple",
            name="Simple Function",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="Simple function",
            source_repository="test",
            source_files=["test.py"],
            extraction_method="pattern_matching",
            integration_complexity=IntegrationComplexity.SIMPLE,
            dependencies=[],  # No dependencies
            external_apis=[]   # No external APIs
        )
        
        # Complex capability  
        complex_cap = ExtractedCapability(
            id="complex",
            name="Complex System",
            capability_type=CapabilityType.MULTI_AGENT_COORDINATION,
            description="Complex multi-agent system",
            source_repository="test",
            source_files=["test.py"],
            extraction_method="architecture_mapping",
            integration_complexity=IntegrationComplexity.COMPLEX,
            dependencies=['tensorflow', 'pytorch', 'numpy', 'scipy', 'pandas', 'requests'],
            external_apis=['https://api.example.com']
        )
        
        # Simple should be easier to integrate
        self.assertEqual(simple_cap.integration_complexity, IntegrationComplexity.SIMPLE)
        self.assertEqual(complex_cap.integration_complexity, IntegrationComplexity.COMPLEX)


class TestCapabilitySynthesizer(unittest.TestCase):
    """Test CapabilitySynthesizer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.synthesizer = CapabilitySynthesizer()
        
    def test_synthesis_patterns_initialization(self):
        """Test synthesis patterns initialization."""
        patterns = self.synthesizer.synthesis_patterns
        self.assertGreater(len(patterns), 0)
        
        # Check for expected patterns
        pattern_names = [p.pattern_name for p in patterns]
        self.assertIn('task_queue_to_orchestration', pattern_names)
        
    def test_cwmai_patterns_loading(self):
        """Test CWMAI patterns loading."""
        patterns = self.synthesizer.cwmai_patterns
        self.assertGreater(len(patterns), 0)
        
        pattern_names = [p['name'] for p in patterns]
        self.assertIn('task_orchestration', pattern_names)
        self.assertIn('ai_brain_pattern', pattern_names)
        
    def test_strategy_selection(self):
        """Test synthesis strategy selection."""
        # High compatibility capability
        high_compat_capability = ExtractedCapability(
            id="high_compat",
            name="High Compatibility Capability",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="Highly compatible capability",
            source_repository="test",
            source_files=["test.py"],
            extraction_method="pattern_matching",
            integration_complexity=IntegrationComplexity.SIMPLE
        )
        
        high_compat = {
            'architectural_alignment': 0.9,
            'pattern_matches': [{'similarity': 0.8}],
            'interface_compatibility': {'test_interface': {'compatibility_score': 0.8}}
        }
        
        strategy = self.synthesizer._select_synthesis_strategy(high_compat_capability, high_compat)
        self.assertEqual(strategy, SynthesisStrategy.DIRECT_ADAPTATION)
        
        # Low compatibility capability
        low_compat = {
            'architectural_alignment': 0.3,
            'pattern_matches': [],
            'interface_compatibility': {}
        }
        
        strategy = self.synthesizer._select_synthesis_strategy(high_compat_capability, low_compat)
        self.assertEqual(strategy, SynthesisStrategy.HYBRID_SYNTHESIS)
        
    def test_synthesis_complexity_assessment(self):
        """Test synthesis complexity assessment."""
        simple_cap = ExtractedCapability(
            id="simple",
            name="Simple Capability",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="Simple capability",
            source_repository="test",
            source_files=["test.py"],
            extraction_method="pattern_matching",
            integration_complexity=IntegrationComplexity.SIMPLE,
            classes=[{'name': 'SimpleClass'}],
            functions=[{'name': 'simple_function'}],
            dependencies=['json']
        )
        
        complexity = self.synthesizer._assess_synthesis_complexity(
            simple_cap, SynthesisStrategy.DIRECT_ADAPTATION
        )
        
        self.assertIn(complexity, [SynthesisComplexity.TRIVIAL, SynthesisComplexity.SIMPLE])


class TestExternalKnowledgeIntegrator(unittest.TestCase):
    """Test ExternalKnowledgeIntegrator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.integrator = ExternalKnowledgeIntegrator()
        
    def test_integration_strategies(self):
        """Test integration strategy enumeration."""
        strategies = list(IntegrationStrategy)
        self.assertIn(IntegrationStrategy.DIRECT_COPY, strategies)
        self.assertIn(IntegrationStrategy.ADAPTER_PATTERN, strategies)
        self.assertIn(IntegrationStrategy.PLUGIN_SYSTEM, strategies)
        
    def test_integration_plan_creation(self):
        """Test integration plan creation."""
        capability = ExtractedCapability(
            id="test_cap",
            name="Test Capability",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="Test capability",
            source_repository="test",
            source_files=["test.py"],
            extraction_method="pattern_matching",
            integration_complexity=IntegrationComplexity.SIMPLE
        )
        
        # Mock create_integration_plan
        async def mock_create_plan():
            return IntegrationPlan(
                capability_id=capability.id,
                capability_name=capability.name,
                integration_strategy=IntegrationStrategy.DIRECT_COPY,
                target_modules=['task_manager.py'],
                modification_steps=[],
                test_requirements=['Unit tests'],
                rollback_plan={},
                estimated_effort_hours=2.0,
                risk_assessment={'overall_risk_level': 'low'},
                success_criteria=['Integration successful']
            )
            
        # Test plan attributes
        plan = asyncio.run(mock_create_plan())
        self.assertEqual(plan.capability_id, capability.id)
        self.assertEqual(plan.integration_strategy, IntegrationStrategy.DIRECT_COPY)
        self.assertEqual(plan.estimated_effort_hours, 2.0)
        
    def test_capability_recommendations(self):
        """Test capability integration recommendations."""
        # Create test capabilities
        capabilities = [
            ExtractedCapability(
                id="high_value",
                name="High Value Capability",
                capability_type=CapabilityType.TASK_ORCHESTRATION,
                description="High value capability",
                source_repository="test",
                source_files=["test.py"],
                extraction_method="pattern_matching",
                integration_complexity=IntegrationComplexity.SIMPLE,
                code_quality_score=0.9,
                extraction_confidence=0.8
            ),
            ExtractedCapability(
                id="low_value",
                name="Low Value Capability", 
                capability_type=CapabilityType.DATA_PROCESSING,
                description="Low value capability",
                source_repository="test",
                source_files=["test.py"],
                extraction_method="pattern_matching",
                integration_complexity=IntegrationComplexity.COMPLEX,
                code_quality_score=0.3,
                extraction_confidence=0.4
            )
        ]
        
        recommendations = self.integrator.get_integration_recommendations(capabilities)
        
        # Should recommend high value capability
        self.assertGreater(len(recommendations), 0)
        if recommendations:
            top_rec = recommendations[0]
            self.assertEqual(top_rec['capability_name'], "High Value Capability")


class TestResearchEvolutionEngineExternal(unittest.TestCase):
    """Test ResearchEvolutionEngine external learning enhancements."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_state_manager = Mock()
        self.mock_ai_brain = Mock()
        self.engine = ResearchEvolutionEngine(
            state_manager=self.mock_state_manager,
            ai_brain=self.mock_ai_brain
        )
        
    def test_external_components_initialization(self):
        """Test external learning components initialization."""
        self.assertIsNotNone(self.engine.external_agent_discoverer)
        self.assertIsNotNone(self.engine.capability_extractor)
        self.assertIsNotNone(self.engine.capability_synthesizer)
        self.assertIsNotNone(self.engine.knowledge_integrator)
        
    def test_external_research_configuration(self):
        """Test external research configuration."""
        config = self.engine.config
        self.assertTrue(config['enable_external_agent_research'])
        self.assertIn('ai_papers_repositories', config)
        self.assertIn('https://github.com/masamasa59/ai-agent-papers', config['ai_papers_repositories'])
        
    def test_external_metrics_tracking(self):
        """Test external learning metrics tracking."""
        metrics = self.engine.metrics
        external_metrics = [key for key in metrics.keys() if key.startswith('external_')]
        
        expected_metrics = [
            'external_research_cycles',
            'external_repositories_analyzed', 
            'capabilities_extracted',
            'capabilities_synthesized',
            'external_integrations_successful'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, external_metrics)
            
    async def test_external_research_status(self):
        """Test external research status reporting."""
        status = self.engine.get_external_research_status()
        
        self.assertIn('external_research_enabled', status)
        self.assertIn('external_components_status', status)
        self.assertIn('ai_papers_repositories', status)


class TestSafeSelfImproverExternal(unittest.TestCase):
    """Test SafeSelfImprover external capability integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize git repo for testing
        os.chdir(self.temp_dir)
        os.system('git init')
        os.system('git config user.email "test@example.com"')
        os.system('git config user.name "Test User"')
        
        self.improver = SafeSelfImprover(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir('/')
        shutil.rmtree(self.temp_dir)
        
    def test_external_integration_modification_type(self):
        """Test external integration modification type."""
        self.assertIn(ModificationType.EXTERNAL_INTEGRATION, ModificationType)
        
    def test_external_integration_statistics(self):
        """Test external integration statistics."""
        stats = self.improver.get_external_integration_statistics()
        
        expected_keys = [
            'total_external_integrations',
            'successful_external_integrations', 
            'external_integration_success_rate',
            'external_sources',
            'external_capability_types'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            
    def test_repository_trust_validation(self):
        """Test repository trust validation."""
        # Trusted repository
        trusted_repo = "https://github.com/microsoft/autogen"
        self.assertTrue(self.improver._validate_source_repository_trust(trusted_repo))
        
        # Untrusted repository
        untrusted_repo = "https://github.com/random/untrusted"
        self.assertFalse(self.improver._validate_source_repository_trust(untrusted_repo))
        
        # AI papers repository (specified in request)
        papers_repo = "https://github.com/masamasa59/ai-agent-papers"
        self.assertTrue(self.improver._validate_source_repository_trust(papers_repo))


class TestProductionOrchestratorExternal(unittest.TestCase):
    """Test ProductionOrchestrator external learning integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.mode.value = 'test'
        self.mock_config.log_level = 'INFO'
        self.mock_config.get_enabled_cycles.return_value = {}
        self.mock_config.validate.return_value = True
        
        self.orchestrator = ProductionOrchestrator(self.mock_config)
        
    def test_external_learning_cycle_tracking(self):
        """Test external learning cycle tracking."""
        # Check that external_learning is tracked
        self.assertIn('external_learning', self.orchestrator.cycle_history)
        self.assertIn('external_learning', self.orchestrator.cycle_counts)
        
    def test_external_learning_status(self):
        """Test external learning status reporting."""
        status = self.orchestrator.get_external_learning_status()
        
        expected_keys = [
            'external_learning_enabled',
            'total_cycles',
            'total_repositories_discovered',
            'total_integrations_executed',
            'external_research_components_status'
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)
            
    async def test_external_learning_cycle_execution(self):
        """Test external learning cycle execution."""
        # Mock research engine with external capabilities
        self.orchestrator.research_engine.external_agent_discoverer = Mock()
        self.orchestrator.research_engine._execute_external_agent_research = AsyncMock(
            return_value={
                'repositories_discovered': [{'name': 'test-repo'}],
                'capabilities_extracted': [{'name': 'test-capability'}],
                'capabilities_synthesized': [],
                'integrations_planned': []
            }
        )
        
        result = await self.orchestrator._execute_external_learning_cycle()
        
        self.assertEqual(result['cycle_type'], 'external_learning')
        self.assertIn('repositories_discovered', result)
        self.assertIn('capabilities_extracted', result)


class TestExternalLearningSystemIntegration(unittest.TestCase):
    """Test integration between all external learning components."""
    
    def setUp(self):
        """Set up integrated test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    async def test_end_to_end_external_learning_flow(self):
        """Test complete external learning flow."""
        # 1. Discovery
        discoverer = ExternalAgentDiscoverer(DiscoveryConfig(max_repositories_per_scan=1))
        
        # Mock repository analysis
        mock_repo = RepositoryAnalysis(
            url="https://github.com/test/external-agent",
            name="external-agent",
            description="Test external agent",
            language="Python",
            stars=100,
            forks=20,
            last_updated="2024-01-01",
            health_score=0.8,
            capabilities=[CapabilityType.TASK_ORCHESTRATION],
            architecture_patterns=['plugin_architecture'],
            key_files=[],
            integration_difficulty=0.3,
            license="MIT",
            documentation_quality=0.7,
            test_coverage=0.6,
            performance_indicators={},
            security_assessment={},
            compatibility_score=0.8
        )
        
        # 2. Extraction
        extractor = CapabilityExtractor()
        
        mock_capability = ExtractedCapability(
            id="test_external_cap",
            name="External Task Manager",
            capability_type=CapabilityType.TASK_ORCHESTRATION,
            description="External task management capability",
            source_repository=mock_repo.url,
            source_files=["external_task_manager.py"],
            extraction_method="pattern_matching",
            integration_complexity=IntegrationComplexity.SIMPLE,
            classes=[{
                'name': 'ExternalTaskManager',
                'docstring': 'External task manager class',
                'methods': ['add_task', 'execute_task']
            }],
            functions=[{
                'name': 'create_task',
                'docstring': 'Create a new task',
                'args': ['task_type', 'description']
            }],
            code_quality_score=0.8,
            extraction_confidence=0.9
        )
        
        # 3. Synthesis
        synthesizer = CapabilitySynthesizer()
        
        # Mock synthesis result
        synthesized = SynthesizedCapability(
            original_capability=mock_capability,
            synthesis_strategy=SynthesisStrategy.DIRECT_ADAPTATION,
            synthesis_complexity=SynthesisComplexity.SIMPLE,
            synthesized_classes=[{
                'name': 'ExternalTaskManager',
                'cwmai_adapted': True,
                'integration_points': ['task_manager.py']
            }],
            synthesis_confidence=0.9,
            architectural_alignment=0.8,
            quality_preservation=0.9
        )
        
        # 4. Integration planning
        integrator = ExternalKnowledgeIntegrator()
        
        # Mock integration plan
        integration_plan = IntegrationPlan(
            capability_id=mock_capability.id,
            capability_name=mock_capability.name,
            integration_strategy=IntegrationStrategy.DIRECT_COPY,
            target_modules=['task_manager.py'],
            modification_steps=[{
                'action': 'add_class',
                'target': 'task_manager.py',
                'content': 'ExternalTaskManager class'
            }],
            test_requirements=['Unit tests for external task manager'],
            rollback_plan={'method': 'git_reset'},
            estimated_effort_hours=2.0,
            risk_assessment={'overall_risk_level': 'low'},
            success_criteria=['Integration successful', 'Tests pass']
        )
        
        # Verify flow components work together
        self.assertEqual(mock_capability.source_repository, mock_repo.url)
        self.assertEqual(synthesized.original_capability.id, mock_capability.id)
        self.assertEqual(integration_plan.capability_id, mock_capability.id)
        
        # Verify safety measures
        self.assertGreater(mock_capability.extraction_confidence, 0.8)
        self.assertGreater(synthesized.synthesis_confidence, 0.8)
        self.assertEqual(integration_plan.risk_assessment['overall_risk_level'], 'low')


def run_external_learning_tests():
    """Run all external learning system tests."""
    print("=== Running External Learning System Tests ===\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestExternalAgentDiscoverer,
        TestCapabilityExtractor,
        TestCapabilitySynthesizer,
        TestExternalKnowledgeIntegrator,
        TestResearchEvolutionEngineExternal,
        TestSafeSelfImproverExternal,
        TestProductionOrchestratorExternal,
        TestExternalLearningSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_external_learning_tests()
    exit(0 if success else 1)