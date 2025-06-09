"""
Unit tests for DynamicSwarmIntelligence module.

Tests multi-agent AI swarm coordination, consensus building, and collective decision making.
Follows AAA pattern with comprehensive coverage of swarm intelligence functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from dynamic_swarm import DynamicSwarmIntelligence


class TestDynamicSwarmIntelligence:
    """Test suite for DynamicSwarmIntelligence class."""

    def test_init_with_ai_brain(self):
        """Test swarm initialization with AI brain."""
        # Arrange
        mock_ai_brain = Mock()
        
        # Act
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        # Assert
        assert swarm.ai_brain == mock_ai_brain
        assert hasattr(swarm, 'agents')
        assert hasattr(swarm, 'consensus_threshold')

    def test_init_default_agents(self):
        """Test swarm initializes with default agents."""
        # Arrange
        mock_ai_brain = Mock()
        
        # Act
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        # Assert
        assert len(swarm.agents) > 0
        assert all('role' in agent for agent in swarm.agents)
        assert all('model' in agent for agent in swarm.agents)

    @pytest.mark.asyncio
    async def test_process_task_swarm_basic(self):
        """Test basic task processing with swarm."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        test_task = {
            'id': 'test-task',
            'type': 'FEATURE',
            'description': 'Test task for swarm processing'
        }
        
        test_context = {
            'active_projects': [],
            'charter': {'PRIMARY_PURPOSE': 'Test'}
        }
        
        with patch.object(swarm, '_get_individual_analysis') as mock_individual:
            with patch.object(swarm, '_build_consensus') as mock_consensus:
                mock_individual.return_value = {
                    'recommendation': 'proceed',
                    'confidence': 0.8,
                    'challenges': ['challenge1']
                }
                mock_consensus.return_value = {
                    'consensus_priority': 'high',
                    'recommendation': 'proceed'
                }
                
                # Act
                result = await swarm.process_task_swarm(test_task, test_context)
                
                # Assert
                assert result is not None
                assert 'collective_review' in result
                assert 'duration_seconds' in result

    @pytest.mark.asyncio
    async def test_get_individual_analysis_success(self):
        """Test individual agent analysis success."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        agent = {
            'id': 'test_agent',
            'role': 'analyzer',
            'model': 'claude'
        }
        
        test_task = {'description': 'Test task'}
        test_context = {'active_projects': []}
        
        with patch.object(swarm.ai_brain, 'generate_enhanced_response') as mock_generate:
            mock_generate.return_value = {
                'content': 'Analysis: Task looks good. Recommendation: proceed',
                'confidence': 0.85
            }
            
            # Act
            analysis = await swarm._get_individual_analysis(agent, test_task, test_context)
            
            # Assert
            assert analysis is not None
            assert analysis['agent_id'] == 'test_agent'
            assert 'recommendation' in analysis
            assert 'confidence' in analysis

    @pytest.mark.asyncio
    async def test_get_individual_analysis_error(self):
        """Test individual agent analysis with error."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        agent = {
            'id': 'test_agent',
            'role': 'analyzer',
            'model': 'claude'
        }
        
        test_task = {'description': 'Test task'}
        test_context = {'active_projects': []}
        
        with patch.object(swarm.ai_brain, 'generate_enhanced_response') as mock_generate:
            mock_generate.side_effect = Exception("AI API Error")
            
            # Act
            analysis = await swarm._get_individual_analysis(agent, test_task, test_context)
            
            # Assert
            assert analysis is not None
            assert 'error' in analysis
            assert analysis['agent_id'] == 'test_agent'

    def test_build_consensus_majority(self):
        """Test consensus building with majority agreement."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        analyses = [
            {'recommendation': 'proceed', 'confidence': 0.9, 'priority': 8},
            {'recommendation': 'proceed', 'confidence': 0.8, 'priority': 7},
            {'recommendation': 'reject', 'confidence': 0.6, 'priority': 5}
        ]
        
        # Act
        consensus = swarm._build_consensus(analyses)
        
        # Assert
        assert consensus is not None
        assert 'consensus_priority' in consensus
        assert 'recommendation' in consensus

    def test_build_consensus_no_clear_majority(self):
        """Test consensus building without clear majority."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        analyses = [
            {'recommendation': 'proceed', 'confidence': 0.5, 'priority': 6},
            {'recommendation': 'reject', 'confidence': 0.5, 'priority': 6},
            {'recommendation': 'defer', 'confidence': 0.5, 'priority': 6}
        ]
        
        # Act
        consensus = swarm._build_consensus(analyses)
        
        # Assert
        assert consensus is not None
        assert consensus['recommendation'] in ['proceed', 'reject', 'defer', 'review_needed']

    def test_parse_agent_response_valid(self):
        """Test parsing valid agent response."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        response_content = """
        Analysis: This is a good task.
        Recommendation: proceed
        Priority: 8
        Confidence: 0.85
        Challenges: Limited resources, Time constraints
        """
        
        # Act
        parsed = swarm._parse_agent_response(response_content)
        
        # Assert
        assert parsed is not None
        assert parsed['recommendation'] == 'proceed'
        assert parsed['priority'] == 8
        assert parsed['confidence'] == 0.85
        assert len(parsed['challenges']) == 2

    def test_parse_agent_response_minimal(self):
        """Test parsing minimal agent response."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        response_content = "Recommendation: proceed"
        
        # Act
        parsed = swarm._parse_agent_response(response_content)
        
        # Assert
        assert parsed is not None
        assert parsed['recommendation'] == 'proceed'
        assert 'priority' in parsed
        assert 'confidence' in parsed

    def test_enable_debug_logging(self):
        """Test enabling debug logging."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        # Act
        swarm.enable_debug_logging("DEBUG")
        
        # Assert
        assert swarm.debug_enabled is True

    def test_get_debug_summary(self):
        """Test getting debug summary."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        # Act
        summary = swarm.get_debug_summary()
        
        # Assert
        assert summary is not None
        assert 'swarm_config' in summary
        assert 'performance_metrics' in summary

    def test_calculate_consensus_strength(self):
        """Test consensus strength calculation."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        strong_consensus = [
            {'recommendation': 'proceed', 'confidence': 0.9},
            {'recommendation': 'proceed', 'confidence': 0.8},
            {'recommendation': 'proceed', 'confidence': 0.85}
        ]
        
        weak_consensus = [
            {'recommendation': 'proceed', 'confidence': 0.6},
            {'recommendation': 'reject', 'confidence': 0.7},
            {'recommendation': 'proceed', 'confidence': 0.5}
        ]
        
        # Act
        strong_strength = swarm._calculate_consensus_strength(strong_consensus)
        weak_strength = swarm._calculate_consensus_strength(weak_consensus)
        
        # Assert
        assert strong_strength > weak_strength
        assert 0 <= strong_strength <= 1
        assert 0 <= weak_strength <= 1

    def test_filter_outlier_analyses(self):
        """Test filtering outlier analyses."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        analyses = [
            {'priority': 8, 'confidence': 0.9},
            {'priority': 7, 'confidence': 0.8},
            {'priority': 2, 'confidence': 0.3},  # Outlier
            {'priority': 8, 'confidence': 0.85}
        ]
        
        # Act
        filtered = swarm._filter_outlier_analyses(analyses)
        
        # Assert
        assert len(filtered) < len(analyses)
        assert all(analysis['confidence'] >= 0.5 for analysis in filtered)

    @pytest.mark.asyncio
    async def test_swarm_resilience_agent_failure(self):
        """Test swarm resilience when some agents fail."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        test_task = {'description': 'Test task'}
        test_context = {'active_projects': []}
        
        # Mock some agents to fail, others to succeed
        success_response = {
            'content': 'Recommendation: proceed',
            'confidence': 0.8
        }
        
        with patch.object(swarm.ai_brain, 'generate_enhanced_response') as mock_generate:
            # First call fails, second succeeds, third fails, fourth succeeds
            mock_generate.side_effect = [
                Exception("API Error"),
                success_response,
                Exception("Timeout"),
                success_response
            ]
            
            # Act
            result = await swarm.process_task_swarm(test_task, test_context)
            
            # Assert
            assert result is not None
            assert 'collective_review' in result
            # Should work with partial agent responses

    def test_agent_specialization(self):
        """Test different agent specializations."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        # Act
        agents = swarm.agents
        
        # Assert
        roles = [agent['role'] for agent in agents]
        assert 'technical_analyst' in roles
        assert 'strategic_planner' in roles
        assert len(set(roles)) > 1  # Should have different roles

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test swarm performance under multiple concurrent tasks."""
        # Arrange
        mock_ai_brain = Mock()
        swarm = DynamicSwarmIntelligence(mock_ai_brain)
        
        tasks = [
            {'id': f'task-{i}', 'description': f'Task {i}'}
            for i in range(3)
        ]
        context = {'active_projects': []}
        
        with patch.object(swarm, '_get_individual_analysis') as mock_analysis:
            mock_analysis.return_value = {
                'recommendation': 'proceed',
                'confidence': 0.8
            }
            
            # Act
            results = await asyncio.gather(*[
                swarm.process_task_swarm(task, context)
                for task in tasks
            ])
            
            # Assert
            assert len(results) == 3
            assert all(result is not None for result in results)