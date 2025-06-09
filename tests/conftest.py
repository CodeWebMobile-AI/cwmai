"""
Pytest configuration and shared fixtures for CWMAI unit tests.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone
import tempfile

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    return datetime(2025, 1, 9, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_github_token():
    """Mock GitHub token for testing."""
    return "test_token_123"


@pytest.fixture
def mock_api_keys():
    """Mock API keys for AI providers."""
    return {
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'OPENAI_API_KEY': 'test_openai_key',
        'GEMINI_API_KEY': 'test_gemini_key',
        'DEEPSEEK_API_KEY': 'test_deepseek_key',
        'CLAUDE_PAT': 'test_github_token'
    }


@pytest.fixture
def temp_state_file():
    """Create a temporary state file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{}')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return {
        'id': 'test-task-001',
        'title': 'Implement user authentication',
        'description': 'Add secure login functionality',
        'type': 'FEATURE',
        'priority': 'high',
        'status': 'pending',
        'created_at': '2025-01-09T12:00:00Z',
        'estimated_hours': 4.0
    }


@pytest.fixture
def sample_state():
    """Sample system state for testing."""
    return {
        "charter": {
            "primary_goal": "innovation",
            "secondary_goal": "community_engagement",
            "constraints": ["maintain_quality", "ensure_security"]
        },
        "projects": {
            "test-project": {
                "health_score": 85,
                "last_checked": "2025-01-09T12:00:00Z",
                "action_history": [],
                "metrics": {
                    "stars": 5,
                    "forks": 2,
                    "issues_open": 1,
                    "pull_requests_open": 0
                }
            }
        },
        "system_performance": {
            "total_cycles": 10,
            "successful_actions": 8,
            "failed_actions": 2,
            "learning_metrics": {
                "decision_accuracy": 0.8,
                "resource_efficiency": 0.75,
                "goal_achievement": 0.9
            }
        },
        "task_queue": [],
        "version": "1.0.0",
        "last_updated": "2025-01-09T12:00:00Z"
    }


@pytest.fixture
def mock_github_api():
    """Mock GitHub API for testing."""
    mock_github = Mock()
    mock_repo = Mock()
    mock_org = Mock()
    
    # Configure mock organization
    mock_org.get_repos.return_value = [mock_repo]
    mock_github.get_organization.return_value = mock_org
    
    # Configure mock repository
    mock_repo.name = 'test-repo'
    mock_repo.full_name = 'test-org/test-repo'
    mock_repo.archived = False
    mock_repo.disabled = False
    mock_repo.stargazers_count = 5
    mock_repo.forks_count = 2
    mock_repo.open_issues_count = 1
    mock_repo.get_pulls.return_value = []
    
    # Configure file operations
    mock_file = Mock()
    mock_file.content = b'{"test": "data"}'
    mock_file.sha = 'test_sha'
    mock_repo.get_contents.return_value = mock_file
    mock_repo.create_file.return_value = {'commit': {'sha': 'new_sha'}}
    mock_repo.update_file.return_value = {'commit': {'sha': 'updated_sha'}}
    
    return mock_github