#!/usr/bin/env python3
"""
Test Suite for Staged Improvements System

Tests the complete staged improvement workflow including:
- Staging improvements
- Validation
- Monitoring
- Progressive confidence
- A/B testing
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from staged_self_improver import StagedSelfImprover, StagedImprovement
from improvement_validator import ImprovementValidator
from staged_improvement_monitor import StagedImprovementMonitor
from progressive_confidence import ProgressiveConfidence, RiskLevel
from safe_self_improver import ModificationType, Modification
from ab_test_runner import ABTestRunner, ABTestConfig


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test Python file with improvement opportunities
    test_file = os.path.join(temp_dir, 'test_module.py')
    with open(test_file, 'w') as f:
        f.write('''
def process_items(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result

def get_value(data, key):
    if key in data:
        value = data[key]
    else:
        value = None
    return value

class DataProcessor:
    def calculate(self, x, y):
        # Missing docstring
        return x + y
''')
    
    # Initialize git repo
    os.system(f'cd {temp_dir} && git init && git add . && git commit -m "Initial commit"')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def staged_improver(temp_repo):
    """Create a staged self-improver instance."""
    return StagedSelfImprover(repo_path=temp_repo, max_changes_per_day=10)


class TestStagedSelfImprover:
    """Test the staged self-improver functionality."""
    
    def test_initialization(self, staged_improver):
        """Test proper initialization of staged improver."""
        assert staged_improver is not None
        assert os.path.exists(staged_improver.staging_dir)
        assert os.path.exists(staged_improver.validated_dir)
        assert os.path.exists(staged_improver.applied_dir)
    
    def test_find_improvements(self, staged_improver):
        """Test finding improvement opportunities."""
        opportunities = staged_improver.analyze_improvement_opportunities()
        
        assert len(opportunities) > 0
        
        # Should find list comprehension opportunity
        optimization_found = any(
            opp['type'] == ModificationType.OPTIMIZATION 
            for opp in opportunities
        )
        assert optimization_found
        
        # Should find missing docstring
        doc_found = any(
            opp['type'] == ModificationType.DOCUMENTATION
            for opp in opportunities
        )
        assert doc_found
    
    def test_stage_improvement(self, staged_improver):
        """Test staging an improvement."""
        # Find opportunities
        opportunities = staged_improver.analyze_improvement_opportunities()
        assert len(opportunities) > 0
        
        # Propose improvement
        opp = opportunities[0]
        modification = staged_improver.propose_improvement(
            target_file=opp['file'],
            improvement_type=opp['type'],
            description=opp['description']
        )
        
        assert modification is not None
        assert modification.safety_score >= 0.8
        
        # Stage improvement
        staged = staged_improver.stage_improvement(modification)
        
        assert staged is not None
        assert os.path.exists(staged.staged_path)
        assert staged.metadata['staging_id'] in staged_improver.staged_improvements
    
    @pytest.mark.asyncio
    async def test_validate_improvement(self, staged_improver):
        """Test validating a staged improvement."""
        # Stage an improvement first
        opportunities = staged_improver.analyze_improvement_opportunities()
        modification = staged_improver.propose_improvement(
            target_file=opportunities[0]['file'],
            improvement_type=opportunities[0]['type'],
            description=opportunities[0]['description']
        )
        staged = staged_improver.stage_improvement(modification)
        
        # Validate
        validation_result = await staged_improver.validate_staged_improvement(
            staged.metadata['staging_id']
        )
        
        assert validation_result is not None
        assert 'ready_to_apply' in validation_result
        assert 'syntax_valid' in validation_result
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, staged_improver):
        """Test batch staging and validation."""
        opportunities = staged_improver.analyze_improvement_opportunities()
        
        # Batch stage
        staged_ids = await staged_improver.stage_batch_improvements(
            opportunities,
            max_batch=2
        )
        
        assert len(staged_ids) <= 2
        
        # Batch validate
        validation_results = await staged_improver.validate_batch(staged_ids)
        
        assert len(validation_results) == len(staged_ids)
        for result in validation_results.values():
            assert 'ready_to_apply' in result


class TestImprovementValidator:
    """Test the improvement validator."""
    
    @pytest.mark.asyncio
    async def test_syntax_validation(self, temp_repo):
        """Test syntax validation."""
        validator = ImprovementValidator(temp_repo)
        
        # Create a file with syntax error
        bad_file = os.path.join(temp_repo, 'bad_syntax.py')
        with open(bad_file, 'w') as f:
            f.write('def bad_function(\n')
        
        errors = []
        result = validator._validate_syntax(bad_file, errors)
        
        assert result is False
        assert len(errors) > 0
        assert 'SYNTAX_ERROR' in errors[0]
    
    @pytest.mark.asyncio
    async def test_security_scan(self, temp_repo):
        """Test security scanning."""
        validator = ImprovementValidator(temp_repo)
        
        # Create file with security issues
        unsafe_file = os.path.join(temp_repo, 'unsafe.py')
        with open(unsafe_file, 'w') as f:
            f.write('''
import os
def dangerous():
    os.system('rm -rf /')
    eval(input())
''')
        
        errors = []
        warnings = []
        result = validator._security_scan(unsafe_file, errors, warnings)
        
        assert result is False
        assert len(errors) > 0
        assert any('SECURITY' in error for error in errors)


class TestStagedImprovementMonitor:
    """Test the monitoring system."""
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, temp_repo):
        """Test monitoring start and stop."""
        monitor = StagedImprovementMonitor(temp_repo)
        
        # Start monitoring
        metrics = await monitor.start_monitoring(
            staging_id="test_001",
            file_path=os.path.join(temp_repo, "test_module.py")
        )
        
        assert metrics is not None
        assert metrics.staging_id == "test_001"
        assert "test_001" in monitor.active_monitors
        
        # Wait a bit for metrics collection
        await asyncio.sleep(2)
        
        # Stop monitoring
        final_metrics = await monitor.stop_monitoring(
            staging_id="test_001",
            improvement_applied=True
        )
        
        assert final_metrics is not None
        assert final_metrics.verdict in ['improved', 'degraded', 'neutral', 'unknown']
        assert "test_001" not in monitor.active_monitors


class TestProgressiveConfidence:
    """Test the progressive confidence system."""
    
    def test_initial_confidence(self, temp_repo):
        """Test initial confidence state."""
        confidence = ProgressiveConfidence(temp_repo)
        
        assert confidence.metrics.confidence_score >= 0
        assert confidence.metrics.confidence_score <= 1
        assert confidence.metrics.total_improvements == 0
    
    def test_should_auto_apply_initial(self, temp_repo):
        """Test auto-apply decision in initial state."""
        confidence = ProgressiveConfidence(temp_repo)
        
        should_apply, reason = confidence.should_auto_apply(
            ModificationType.OPTIMIZATION,
            RiskLevel.LOW
        )
        
        assert should_apply is False
        assert "initial manual period" in reason.lower()
    
    def test_record_outcome(self, temp_repo):
        """Test recording improvement outcomes."""
        confidence = ProgressiveConfidence(temp_repo)
        
        initial_count = confidence.metrics.total_improvements
        
        # Record a successful outcome
        confidence.record_outcome(
            staging_id="test_001",
            improvement_type=ModificationType.OPTIMIZATION,
            risk_level=RiskLevel.LOW,
            success=True,
            performance_impact=0.1
        )
        
        assert confidence.metrics.total_improvements == initial_count + 1
        assert confidence.metrics.successful_improvements > 0
    
    def test_risk_assessment(self, temp_repo):
        """Test risk level assessment."""
        confidence = ProgressiveConfidence(temp_repo)
        
        # Low risk for documentation
        risk = confidence.assess_risk_level(
            ModificationType.DOCUMENTATION,
            {'lines_changed': 5, 'target_file': 'README.md'}
        )
        assert risk == RiskLevel.LOW
        
        # Higher risk for security changes
        risk = confidence.assess_risk_level(
            ModificationType.SECURITY,
            {'lines_changed': 50, 'target_file': 'auth.py'}
        )
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]


class TestABTestRunner:
    """Test the A/B testing functionality."""
    
    @pytest.mark.asyncio
    async def test_ab_test_config(self, temp_repo):
        """Test A/B test configuration."""
        runner = ABTestRunner(temp_repo)
        config = ABTestConfig(
            duration_seconds=60,
            traffic_split=0.5,
            min_samples=10
        )
        
        assert config.duration_seconds == 60
        assert config.traffic_split == 0.5
        assert config.auto_promote is True
    
    @pytest.mark.asyncio
    async def test_ab_test_lifecycle(self, staged_improver):
        """Test running an A/B test (simplified)."""
        # First create and validate a staged improvement
        opportunities = staged_improver.analyze_improvement_opportunities()
        if not opportunities:
            pytest.skip("No improvement opportunities found")
        
        # Stage improvement
        staged_ids = await staged_improver.stage_batch_improvements(
            opportunities, max_batch=1
        )
        
        if not staged_ids:
            pytest.skip("Could not stage improvements")
        
        # Validate
        validation_results = await staged_improver.validate_batch(staged_ids)
        
        # Create A/B test runner
        runner = ABTestRunner(staged_improver.repo_path)
        
        # Start test with very short duration
        config = ABTestConfig(
            duration_seconds=2,  # Very short for testing
            traffic_split=0.5,
            min_samples=5,
            auto_promote=False
        )
        
        try:
            test_id = await runner.start_ab_test(staged_ids[0], config)
            assert test_id is not None
            
            # Check status
            status = runner.get_test_status(test_id)
            assert status is not None
            assert status['status'] == 'running'
            
            # Wait for completion
            await asyncio.sleep(3)
            
            # Check final status
            final_status = runner.get_test_status(test_id)
            assert final_status['status'] == 'completed'
            
        except ValueError as e:
            if "not validated" in str(e):
                pytest.skip("Improvement not validated")
            raise


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, staged_improver):
        """Test the complete staged improvement workflow."""
        # 1. Find improvements
        opportunities = staged_improver.analyze_improvement_opportunities()
        assert len(opportunities) > 0
        
        # 2. Stage improvements
        staged_ids = await staged_improver.stage_batch_improvements(
            opportunities, max_batch=1
        )
        assert len(staged_ids) > 0
        
        # 3. Validate
        validation_results = await staged_improver.validate_batch(staged_ids)
        assert len(validation_results) > 0
        
        # 4. Check staging report
        report = staged_improver.generate_staging_report()
        assert report is not None
        assert report['summary']['total_staged'] > 0
        
        # 5. Apply if validated (in test mode, we don't actually apply)
        validated = staged_improver.get_staged_improvements('validated')
        if validated:
            # Would apply here in production
            assert len(validated) > 0


# Test utilities
def create_test_file_with_improvements():
    """Create a test file with known improvement opportunities."""
    content = '''
# Test file with improvement opportunities

def process_data(data):
    """Process data items."""
    results = []
    for item in data:
        results.append(item.strip().lower())
    return results

def check_value(dictionary, key):
    if key in dictionary:
        val = dictionary[key]
    else:
        val = "default"
    return val

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        result = 0
        for i in range(b):
            result = result + a
        return result
'''
    return content


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])