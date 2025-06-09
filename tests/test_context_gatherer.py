"""
Unit tests for ContextGatherer module.

Tests context collection, analysis, and aggregation from various sources.
Follows AAA pattern with comprehensive coverage of context gathering functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from context_gatherer import ContextGatherer


class TestContextGatherer:
    """Test suite for ContextGatherer class."""

    def test_init_default(self):
        """Test ContextGatherer initialization with defaults."""
        # Arrange & Act
        gatherer = ContextGatherer()
        
        # Assert
        assert gatherer is not None
        assert hasattr(gatherer, 'context_sources')
        assert hasattr(gatherer, 'cache_timeout')

    @patch.dict('os.environ', {'CLAUDE_PAT': 'test_token'})
    def test_init_with_github_token(self):
        """Test ContextGatherer initialization with GitHub token."""
        # Arrange & Act
        gatherer = ContextGatherer()
        
        # Assert
        assert gatherer.github_token == 'test_token'

    def test_gather_project_context_basic(self):
        """Test basic project context gathering."""
        # Arrange
        gatherer = ContextGatherer()
        project_data = {
            'name': 'test-project',
            'language': 'Python',
            'stars': 10,
            'forks': 2,
            'issues': 3
        }
        
        # Act
        context = gatherer.gather_project_context(project_data)
        
        # Assert
        assert context is not None
        assert 'project_health' in context
        assert 'activity_level' in context
        assert 'technology_stack' in context

    def test_gather_project_context_empty_data(self):
        """Test project context gathering with empty data."""
        # Arrange
        gatherer = ContextGatherer()
        
        # Act
        context = gatherer.gather_project_context({})
        
        # Assert
        assert context is not None
        assert context['project_health'] == 'unknown'

    @patch('context_gatherer.Github')
    def test_gather_github_context_success(self, mock_github_class):
        """Test successful GitHub context gathering."""
        # Arrange
        gatherer = ContextGatherer()
        gatherer.github_token = 'test_token'
        
        mock_github = Mock()
        mock_repo = Mock()
        mock_issues = [Mock(), Mock()]
        mock_pulls = [Mock()]
        
        mock_repo.get_issues.return_value = mock_issues
        mock_repo.get_pulls.return_value = mock_pulls
        mock_repo.stargazers_count = 15
        mock_repo.forks_count = 3
        
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Act
        context = gatherer.gather_github_context('test/repo')
        
        # Assert
        assert context is not None
        assert context['open_issues'] == 2
        assert context['open_pulls'] == 1
        assert context['stars'] == 15

    def test_gather_github_context_no_token(self):
        """Test GitHub context gathering without token."""
        # Arrange
        gatherer = ContextGatherer()
        gatherer.github_token = None
        
        # Act
        context = gatherer.gather_github_context('test/repo')
        
        # Assert
        assert context is not None
        assert context['error'] == 'no_github_token'

    @patch('context_gatherer.Github')
    def test_gather_github_context_api_error(self, mock_github_class):
        """Test GitHub context gathering with API error."""
        # Arrange
        gatherer = ContextGatherer()
        gatherer.github_token = 'test_token'
        
        mock_github_class.side_effect = Exception("API Error")
        
        # Act
        context = gatherer.gather_github_context('test/repo')
        
        # Assert
        assert context is not None
        assert 'error' in context

    def test_analyze_market_trends_basic(self):
        """Test basic market trends analysis."""
        # Arrange
        gatherer = ContextGatherer()
        technology_stack = ['Python', 'React', 'Docker']
        
        # Act
        trends = gatherer.analyze_market_trends(technology_stack)
        
        # Assert
        assert trends is not None
        assert 'trending_technologies' in trends
        assert 'market_demand' in trends

    def test_analyze_market_trends_empty_stack(self):
        """Test market trends analysis with empty technology stack."""
        # Arrange
        gatherer = ContextGatherer()
        
        # Act
        trends = gatherer.analyze_market_trends([])
        
        # Assert
        assert trends is not None
        assert trends['trending_technologies'] == []

    def test_gather_system_metrics(self):
        """Test system metrics gathering."""
        # Arrange
        gatherer = ContextGatherer()
        
        # Act
        with patch('psutil.cpu_percent', return_value=25.5):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.0
                metrics = gatherer.gather_system_metrics()
        
        # Assert
        assert metrics is not None
        assert 'cpu_usage' in metrics
        assert 'memory_usage' in metrics
        assert metrics['cpu_usage'] == 25.5

    def test_aggregate_context_sources(self):
        """Test aggregation of multiple context sources."""
        # Arrange
        gatherer = ContextGatherer()
        
        project_context = {'project_health': 'good', 'stars': 10}
        github_context = {'open_issues': 2, 'open_pulls': 1}
        system_context = {'cpu_usage': 30.0, 'memory_usage': 50.0}
        
        # Act
        aggregated = gatherer.aggregate_context(
            project_context, github_context, system_context
        )
        
        # Assert
        assert aggregated is not None
        assert aggregated['project_health'] == 'good'
        assert aggregated['open_issues'] == 2
        assert aggregated['cpu_usage'] == 30.0

    def test_context_caching(self):
        """Test context caching functionality."""
        # Arrange
        gatherer = ContextGatherer()
        cache_key = 'test_project_context'
        test_context = {'cached': True, 'timestamp': datetime.now().isoformat()}
        
        # Act
        gatherer.cache_context(cache_key, test_context)
        cached_result = gatherer.get_cached_context(cache_key)
        
        # Assert
        assert cached_result is not None
        assert cached_result['cached'] is True

    def test_context_cache_expiration(self):
        """Test context cache expiration."""
        # Arrange
        gatherer = ContextGatherer()
        gatherer.cache_timeout = 0.1  # Very short timeout for testing
        cache_key = 'expiring_context'
        test_context = {'will_expire': True}
        
        # Act
        gatherer.cache_context(cache_key, test_context)
        
        import time
        time.sleep(0.2)  # Wait for expiration
        
        expired_result = gatherer.get_cached_context(cache_key)
        
        # Assert
        assert expired_result is None

    def test_analyze_code_quality_metrics(self):
        """Test code quality metrics analysis."""
        # Arrange
        gatherer = ContextGatherer()
        
        code_metrics = {
            'lines_of_code': 5000,
            'test_coverage': 85.0,
            'complexity_score': 7.2,
            'tech_debt_hours': 12.5
        }
        
        # Act
        quality_analysis = gatherer.analyze_code_quality(code_metrics)
        
        # Assert
        assert quality_analysis is not None
        assert 'quality_score' in quality_analysis
        assert 'recommendations' in quality_analysis

    def test_gather_dependency_context(self):
        """Test dependency context gathering."""
        # Arrange
        gatherer = ContextGatherer()
        
        dependencies = [
            {'name': 'requests', 'version': '2.32.3', 'vulnerabilities': []},
            {'name': 'numpy', 'version': '1.24.3', 'vulnerabilities': ['CVE-2023-1234']}
        ]
        
        # Act
        dep_context = gatherer.gather_dependency_context(dependencies)
        
        # Assert
        assert dep_context is not None
        assert 'total_dependencies' in dep_context
        assert 'security_vulnerabilities' in dep_context
        assert dep_context['total_dependencies'] == 2

    def test_analyze_team_activity(self):
        """Test team activity analysis."""
        # Arrange
        gatherer = ContextGatherer()
        
        activity_data = [
            {'author': 'dev1', 'commits': 15, 'last_commit': '2025-01-08'},
            {'author': 'dev2', 'commits': 8, 'last_commit': '2025-01-09'}
        ]
        
        # Act
        team_analysis = gatherer.analyze_team_activity(activity_data)
        
        # Assert
        assert team_analysis is not None
        assert 'active_contributors' in team_analysis
        assert 'commit_frequency' in team_analysis

    def test_gather_environment_context(self):
        """Test environment context gathering."""
        # Arrange
        gatherer = ContextGatherer()
        
        # Act
        with patch.dict('os.environ', {'NODE_ENV': 'production', 'DEBUG': 'false'}):
            env_context = gatherer.gather_environment_context()
        
        # Assert
        assert env_context is not None
        assert 'environment_type' in env_context
        assert 'configuration' in env_context

    def test_analyze_performance_metrics(self):
        """Test performance metrics analysis."""
        # Arrange
        gatherer = ContextGatherer()
        
        performance_data = {
            'response_time_avg': 150.5,
            'throughput_rps': 45.2,
            'error_rate': 0.02,
            'memory_usage_mb': 512.8
        }
        
        # Act
        perf_analysis = gatherer.analyze_performance_metrics(performance_data)
        
        # Assert
        assert perf_analysis is not None
        assert 'performance_score' in perf_analysis
        assert 'bottlenecks' in perf_analysis

    def test_context_validation(self):
        """Test context data validation."""
        # Arrange
        gatherer = ContextGatherer()
        
        valid_context = {
            'timestamp': datetime.now().isoformat(),
            'source': 'test',
            'data': {'key': 'value'}
        }
        
        invalid_context = {
            'data': {'key': 'value'}
            # Missing required fields
        }
        
        # Act
        valid_result = gatherer.validate_context(valid_context)
        invalid_result = gatherer.validate_context(invalid_context)
        
        # Assert
        assert valid_result is True
        assert invalid_result is False

    def test_context_enrichment(self):
        """Test context enrichment with additional metadata."""
        # Arrange
        gatherer = ContextGatherer()
        
        basic_context = {
            'project_name': 'test-project',
            'language': 'Python'
        }
        
        # Act
        enriched_context = gatherer.enrich_context(basic_context)
        
        # Assert
        assert enriched_context is not None
        assert 'timestamp' in enriched_context
        assert 'context_version' in enriched_context
        assert enriched_context['project_name'] == 'test-project'

    def test_security_context_gathering(self):
        """Test security-related context gathering."""
        # Arrange
        gatherer = ContextGatherer()
        
        security_data = {
            'open_security_issues': 2,
            'last_security_scan': '2025-01-08',
            'dependency_vulnerabilities': 1
        }
        
        # Act
        security_context = gatherer.gather_security_context(security_data)
        
        # Assert
        assert security_context is not None
        assert 'security_score' in security_context
        assert 'risk_level' in security_context

    def test_context_filtering_sensitive_data(self):
        """Test filtering of sensitive data from context."""
        # Arrange
        gatherer = ContextGatherer()
        
        context_with_secrets = {
            'api_key': 'secret_key_123',
            'password': 'super_secret',
            'username': 'testuser',
            'config': {'debug': True}
        }
        
        # Act
        filtered_context = gatherer.filter_sensitive_data(context_with_secrets)
        
        # Assert
        assert filtered_context is not None
        assert filtered_context['api_key'] == '[REDACTED]'
        assert filtered_context['password'] == '[REDACTED]'
        assert filtered_context['username'] == 'testuser'  # Not sensitive
        assert filtered_context['config']['debug'] is True

    def test_context_source_prioritization(self):
        """Test prioritization of context sources."""
        # Arrange
        gatherer = ContextGatherer()
        
        contexts = [
            {'source': 'github', 'priority': 'high', 'data': {'stars': 100}},
            {'source': 'metrics', 'priority': 'medium', 'data': {'cpu': 30}},
            {'source': 'cache', 'priority': 'low', 'data': {'cached': True}}
        ]
        
        # Act
        prioritized = gatherer.prioritize_context_sources(contexts)
        
        # Assert
        assert prioritized is not None
        assert len(prioritized) == 3
        assert prioritized[0]['source'] == 'github'  # Highest priority first

    def test_error_handling_invalid_sources(self):
        """Test error handling for invalid context sources."""
        # Arrange
        gatherer = ContextGatherer()
        
        # Act
        result = gatherer.gather_context_from_source('invalid_source')
        
        # Assert
        assert result is not None
        assert 'error' in result
        assert result['error'] == 'invalid_source'