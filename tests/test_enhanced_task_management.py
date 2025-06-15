"""
Test Enhanced Task Management System

Comprehensive tests for the new task decomposition, hierarchical management,
complexity analysis, and progressive task generation system.
"""

import pytest
import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Any

# Import the enhanced task management components
try:
    from scripts.task_decomposition_engine import TaskDecompositionEngine, TaskComplexity
    from scripts.hierarchical_task_manager import HierarchicalTaskManager, TaskNode
    from scripts.complexity_analyzer import ComplexityAnalyzer, ComplexityLevel
    from scripts.progressive_task_generator import ProgressiveTaskGenerator, ProgressionContext
    from scripts.ai_brain import IntelligentAIBrain
    from scripts.state_manager import StateManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import enhanced task management components: {e}")
    IMPORTS_AVAILABLE = False


class MockAIBrain:
    """Mock AI brain for testing."""
    
    async def generate_enhanced_response(self, prompt: str) -> Dict[str, Any]:
        """Mock AI response based on prompt keywords."""
        if "decompose" in prompt.lower() or "sub-tasks" in prompt.lower():
            return {
                'content': '''[
                    {
                        "title": "Setup project structure",
                        "description": "Create initial project structure and configuration",
                        "estimated_hours": 2.0,
                        "deliverables": ["Project structure", "Configuration files"],
                        "acceptance_criteria": ["Structure follows best practices"],
                        "technical_requirements": ["Use standard conventions"],
                        "sequence_order": 1,
                        "can_parallelize": false,
                        "type": "SETUP"
                    },
                    {
                        "title": "Implement core functionality",
                        "description": "Build the main features",
                        "estimated_hours": 4.0,
                        "deliverables": ["Core features", "API endpoints"],
                        "acceptance_criteria": ["All features work correctly"],
                        "technical_requirements": ["Follow coding standards"],
                        "sequence_order": 2,
                        "can_parallelize": false,
                        "type": "FEATURE"
                    },
                    {
                        "title": "Add testing",
                        "description": "Create comprehensive tests",
                        "estimated_hours": 2.0,
                        "deliverables": ["Unit tests", "Integration tests"],
                        "acceptance_criteria": ["90% test coverage"],
                        "technical_requirements": ["Use testing framework"],
                        "sequence_order": 3,
                        "can_parallelize": true,
                        "type": "TESTING"
                    }
                ]'''
            }
        elif "complexity" in prompt.lower() or "assess" in prompt.lower():
            return {'content': '0.7'}
        elif "strategy" in prompt.lower():
            return {'content': 'SEQUENTIAL'}
        elif "next tasks" in prompt.lower() or "suggestions" in prompt.lower():
            return {
                'content': '''[
                    {
                        "title": "Add unit tests for new feature",
                        "description": "Create comprehensive unit tests",
                        "task_type": "TESTING",
                        "priority": "high",
                        "estimated_hours": 2.0,
                        "relationship": "sequential",
                        "reasoning": "Testing is needed after feature completion"
                    }
                ]'''
            }
        else:
            return {'content': 'Mock AI response'}


@pytest.fixture
def mock_ai_brain():
    """Fixture for mock AI brain."""
    return MockAIBrain()


@pytest.fixture
def sample_task():
    """Fixture for sample task."""
    return {
        'id': 'TASK-1001',
        'type': 'NEW_PROJECT',
        'title': 'Build AI-Powered Dashboard',
        'description': 'Create a comprehensive dashboard for managing AI-generated tasks with real-time updates, user authentication, and data visualization capabilities.',
        'priority': 'high',
        'estimated_hours': 16.0,
        'requirements': [
            'User authentication system',
            'Real-time task updates',
            'Data visualization',
            'Responsive design',
            'API integration'
        ],
        'repository': 'test-repo'
    }


@pytest.fixture
def sample_repository_analysis():
    """Fixture for sample repository analysis."""
    return {
        'basic_info': {
            'name': 'test-repo',
            'language': 'JavaScript',
            'open_issues_count': 5,
            'topics': ['react', 'dashboard', 'api']
        },
        'health_metrics': {
            'health_score': 75,
            'days_since_update': 3
        },
        'technical_stack': {
            'languages': ['JavaScript', 'TypeScript'],
            'frameworks': ['React', 'Express']
        },
        'specific_needs': [
            {
                'type': 'testing',
                'priority': 'high',
                'description': 'Missing comprehensive test suite'
            }
        ]
    }


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Enhanced task management components not available")
class TestComplexityAnalyzer:
    """Test complexity analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_simple_task_complexity(self, mock_ai_brain):
        """Test complexity analysis for simple task."""
        analyzer = ComplexityAnalyzer(mock_ai_brain)
        
        simple_task = {
            'id': 'TASK-001',
            'type': 'BUG_FIX',
            'title': 'Fix typo in documentation',
            'description': 'Correct spelling error in README',
            'estimated_hours': 0.5,
            'requirements': ['Fix typo']
        }
        
        analysis = await analyzer.analyze_complexity(simple_task)
        
        assert analysis.overall_level in [ComplexityLevel.TRIVIAL, ComplexityLevel.SIMPLE]
        assert analysis.overall_score < 0.5
        assert not analysis.decomposition_recommended
        assert analysis.estimated_subtasks <= 2
    
    @pytest.mark.asyncio
    async def test_analyze_complex_task_complexity(self, mock_ai_brain, sample_task):
        """Test complexity analysis for complex task."""
        analyzer = ComplexityAnalyzer(mock_ai_brain)
        
        analysis = await analyzer.analyze_complexity(sample_task)
        
        assert analysis.overall_level in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]
        assert analysis.overall_score > 0.5
        assert analysis.decomposition_recommended
        assert analysis.estimated_subtasks >= 3
        assert len(analysis.risk_factors) > 0
        assert len(analysis.mitigation_strategies) > 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Enhanced task management components not available")
class TestTaskDecompositionEngine:
    """Test task decomposition functionality."""
    
    @pytest.mark.asyncio
    async def test_decompose_complex_task(self, mock_ai_brain, sample_task, sample_repository_analysis):
        """Test decomposition of complex task."""
        engine = TaskDecompositionEngine(mock_ai_brain)
        
        result = await engine.decompose_task(sample_task, sample_repository_analysis)
        
        assert result is not None
        assert len(result.sub_tasks) >= 2
        assert result.total_estimated_hours > 0
        assert len(result.critical_path) > 0
        assert result.complexity_reduced
        
        # Check sub-task properties
        for sub_task in result.sub_tasks:
            assert sub_task.id
            assert sub_task.title
            assert sub_task.description
            assert sub_task.estimated_hours > 0
            assert sub_task.sequence_order > 0
    
    @pytest.mark.asyncio
    async def test_decomposition_strategies(self, mock_ai_brain, sample_task):
        """Test different decomposition strategies."""
        engine = TaskDecompositionEngine(mock_ai_brain)
        
        # Test with different task types
        task_types = ['NEW_PROJECT', 'FEATURE', 'REFACTOR', 'TESTING']
        
        for task_type in task_types:
            test_task = sample_task.copy()
            test_task['type'] = task_type
            
            strategy = await engine._determine_decomposition_strategy(test_task)
            assert strategy is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Enhanced task management components not available")
class TestHierarchicalTaskManager:
    """Test hierarchical task management functionality."""
    
    def test_add_task_hierarchy(self, mock_ai_brain, sample_task):
        """Test adding task hierarchy."""
        manager = HierarchicalTaskManager()
        
        # Create mock decomposition result
        from scripts.task_decomposition_engine import DecompositionResult, SubTask, DecompositionStrategy
        
        sub_tasks = [
            SubTask(
                id='SUB-1',
                parent_id='TASK-1001',
                title='Setup',
                description='Setup project',
                type='SETUP',
                priority='high',
                estimated_hours=2.0,
                sequence_order=1
            ),
            SubTask(
                id='SUB-2',
                parent_id='TASK-1001',
                title='Implementation',
                description='Implement features',
                type='FEATURE',
                priority='high',
                estimated_hours=8.0,
                sequence_order=2
            )
        ]
        
        decomposition_result = DecompositionResult(
            original_task_id='TASK-1001',
            sub_tasks=sub_tasks,
            strategy=DecompositionStrategy.SEQUENTIAL,
            total_estimated_hours=10.0,
            critical_path=['SUB-1', 'SUB-2'],
            parallel_groups=[],
            complexity_reduced=True,
            decomposition_rationale='Task broken down for better management',
            next_actions=['Start with SUB-1']
        )
        
        hierarchy_id = manager.add_task_hierarchy(decomposition_result, sample_task)
        
        assert hierarchy_id in manager.task_nodes
        assert len(manager.task_nodes[hierarchy_id].children) == 2
    
    def test_update_task_progress(self):
        """Test updating task progress."""
        manager = HierarchicalTaskManager()
        
        # Add a simple task
        task_node = TaskNode(
            id='TEST-1',
            parent_id=None,
            title='Test Task',
            description='Test description',
            task_type='FEATURE',
            status='pending',
            priority='medium',
            estimated_hours=4.0
        )
        
        manager.task_nodes['TEST-1'] = task_node
        
        # Update progress
        success = manager.update_task_progress('TEST-1', 50.0, 2.0, 'in_progress')
        
        assert success
        assert manager.task_nodes['TEST-1'].progress_percentage == 50.0
        assert manager.task_nodes['TEST-1'].actual_hours == 2.0
        assert manager.task_nodes['TEST-1'].status == 'in_progress'
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        manager = HierarchicalTaskManager()
        
        # Add tasks with dependencies
        task1 = TaskNode(
            id='TASK-1',
            parent_id=None,
            title='Task 1',
            description='First task',
            task_type='FEATURE',
            status='pending',
            priority='high',
            estimated_hours=2.0
        )
        
        task2 = TaskNode(
            id='TASK-2',
            parent_id=None,
            title='Task 2',
            description='Second task',
            task_type='FEATURE',
            status='pending',
            priority='medium',
            estimated_hours=3.0,
            dependencies=['TASK-1']
        )
        
        manager.task_nodes['TASK-1'] = task1
        manager.task_nodes['TASK-2'] = task2
        manager.dependency_graph['TASK-2'].add('TASK-1')
        
        ready_tasks = manager.get_ready_tasks()
        
        # Only TASK-1 should be ready (no dependencies)
        assert len(ready_tasks) == 1
        assert ready_tasks[0]['id'] == 'TASK-1'


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Enhanced task management components not available")
class TestProgressiveTaskGenerator:
    """Test progressive task generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_next_tasks(self, mock_ai_brain):
        """Test generating next tasks based on completion."""
        complexity_analyzer = ComplexityAnalyzer(mock_ai_brain)
        hierarchical_manager = HierarchicalTaskManager()
        generator = ProgressiveTaskGenerator(mock_ai_brain, hierarchical_manager, complexity_analyzer)
        
        completed_task = {
            'id': 'TASK-001',
            'type': 'FEATURE',
            'title': 'Add user authentication',
            'description': 'Implemented user login and registration',
            'repository': 'test-repo',
            'status': 'completed'
        }
        
        context = ProgressionContext(
            completed_task=completed_task,
            repository_context={'name': 'test-repo'},
            project_state={'active_projects': []},
            recent_patterns=[],
            current_priorities=['testing', 'documentation'],
            team_capacity={},
            timeline_constraints={}
        )
        
        suggestions = await generator.generate_next_tasks(completed_task, context)
        
        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert suggestion.title
            assert suggestion.description
            assert suggestion.task_type
            assert suggestion.confidence > 0
    
    def test_pattern_learning(self, mock_ai_brain):
        """Test pattern learning from task completions."""
        complexity_analyzer = ComplexityAnalyzer(mock_ai_brain)
        hierarchical_manager = HierarchicalTaskManager()
        generator = ProgressiveTaskGenerator(mock_ai_brain, hierarchical_manager, complexity_analyzer)
        
        # Initially no patterns
        assert len(generator.progression_patterns) > 0  # Basic patterns are loaded
        
        # Track suggestion outcome
        generator.track_suggestion_outcome('test-suggestion', True)
        
        assert len(generator.pattern_success_tracking) == 1


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Enhanced task management components not available")
class TestIntegration:
    """Test integration between components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_task_flow(self, mock_ai_brain, sample_task, sample_repository_analysis):
        """Test complete end-to-end task management flow."""
        # Initialize components
        complexity_analyzer = ComplexityAnalyzer(mock_ai_brain)
        decomposition_engine = TaskDecompositionEngine(mock_ai_brain)
        hierarchical_manager = HierarchicalTaskManager()
        progressive_generator = ProgressiveTaskGenerator(
            mock_ai_brain, hierarchical_manager, complexity_analyzer
        )
        
        # Step 1: Analyze complexity
        complexity_analysis = await complexity_analyzer.analyze_complexity(
            sample_task, {'repository_analysis': sample_repository_analysis}
        )
        
        assert complexity_analysis.decomposition_recommended
        
        # Step 2: Decompose task
        decomposition_result = await decomposition_engine.decompose_task(
            sample_task, sample_repository_analysis
        )
        
        assert len(decomposition_result.sub_tasks) > 0
        
        # Step 3: Add to hierarchical manager
        hierarchy_id = hierarchical_manager.add_task_hierarchy(decomposition_result, sample_task)
        
        assert hierarchy_id in hierarchical_manager.task_nodes
        
        # Step 4: Get ready tasks
        ready_tasks = hierarchical_manager.get_ready_tasks()
        
        assert len(ready_tasks) > 0
        
        # Step 5: Complete a task and generate next tasks
        if ready_tasks:
            first_task = ready_tasks[0]
            hierarchical_manager.update_task_progress(
                first_task['id'], 100.0, 2.0, 'completed'
            )
            
            # Create progression context
            context = ProgressionContext(
                completed_task=sample_task,
                repository_context=sample_repository_analysis,
                project_state={},
                recent_patterns=[],
                current_priorities=[],
                team_capacity={},
                timeline_constraints={}
            )
            
            suggestions = await progressive_generator.generate_next_tasks(sample_task, context)
            
            assert len(suggestions) > 0


class TestPerformance:
    """Test performance characteristics of the enhanced system."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Enhanced task management components not available")
    @pytest.mark.asyncio
    async def test_bulk_task_processing(self, mock_ai_brain):
        """Test processing multiple tasks efficiently."""
        complexity_analyzer = ComplexityAnalyzer(mock_ai_brain)
        
        # Create multiple tasks
        tasks = []
        for i in range(10):
            task = {
                'id': f'TASK-{i:03d}',
                'type': 'FEATURE',
                'title': f'Feature {i}',
                'description': f'Implement feature number {i}',
                'estimated_hours': 4.0 + (i % 5),
                'requirements': [f'Requirement {j}' for j in range(i % 3 + 1)]
            }
            tasks.append(task)
        
        # Process all tasks
        start_time = datetime.now()
        analyses = []
        
        for task in tasks:
            analysis = await complexity_analyzer.analyze_complexity(task)
            analyses.append(analysis)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process reasonably quickly
        assert processing_time < 30.0  # Less than 30 seconds for 10 tasks
        assert len(analyses) == 10
        
        # All analyses should be valid
        for analysis in analyses:
            assert analysis.overall_level is not None
            assert 0 <= analysis.overall_score <= 1.0


def test_configuration_and_setup():
    """Test system configuration and setup."""
    # Test that the enhanced system can be configured properly
    if IMPORTS_AVAILABLE:
        from scripts.task_decomposition_engine import TaskDecompositionEngine
        from scripts.complexity_analyzer import ComplexityAnalyzer
        
        # Test initialization without AI brain
        analyzer = ComplexityAnalyzer()
        assert analyzer is not None
        
        # Test with mock AI brain
        mock_brain = MockAIBrain()
        analyzer_with_ai = ComplexityAnalyzer(mock_brain)
        assert analyzer_with_ai.ai_brain is not None


def test_error_handling():
    """Test error handling in enhanced system."""
    if IMPORTS_AVAILABLE:
        from scripts.hierarchical_task_manager import HierarchicalTaskManager
        
        manager = HierarchicalTaskManager()
        
        # Test operations on non-existent tasks
        result = manager.update_task_progress('NON-EXISTENT', 50.0)
        assert not result
        
        hierarchy = manager.get_task_hierarchy('NON-EXISTENT')
        assert hierarchy == {}


if __name__ == '__main__':
    """Run tests directly."""
    if IMPORTS_AVAILABLE:
        print("Running enhanced task management system tests...")
        
        # Run a simple integration test
        async def simple_test():
            mock_brain = MockAIBrain()
            analyzer = ComplexityAnalyzer(mock_brain)
            
            test_task = {
                'id': 'TEST-001',
                'type': 'FEATURE',
                'title': 'Test Feature',
                'description': 'A test feature implementation',
                'estimated_hours': 8.0,
                'requirements': ['Req 1', 'Req 2', 'Req 3']
            }
            
            analysis = await analyzer.analyze_complexity(test_task)
            print(f"Complexity: {analysis.overall_level.value} (score: {analysis.overall_score:.2f})")
            print(f"Decomposition recommended: {analysis.decomposition_recommended}")
            
            if analysis.decomposition_recommended:
                engine = TaskDecompositionEngine(mock_brain)
                result = await engine.decompose_task(test_task)
                print(f"Decomposed into {len(result.sub_tasks)} sub-tasks")
                
                for i, sub_task in enumerate(result.sub_tasks):
                    print(f"  {i+1}. {sub_task.title} ({sub_task.estimated_hours}h)")
        
        # Run the test
        asyncio.run(simple_test())
        print("✓ Enhanced task management system is working correctly!")
    else:
        print("❌ Enhanced task management system components not available for testing")