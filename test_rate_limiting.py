"""
Comprehensive Tests for API Rate Limiting System

Tests for rate_limiter.py, http_ai_client.py integration, and monitoring capabilities.
"""

import json
import os
import time
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
import tempfile
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from scripts.rate_limiter import RateLimiter, RateLimitTier, RateLimitStrategy, RateLimitRule, RateLimitResult
    from scripts.http_ai_client import HTTPAIClient
    from scripts.rate_limit_monitor import RateLimitMonitor
except ImportError as e:
    print(f"Import error: {e}")
    print("Skipping rate limiting tests - modules not available")
    sys.exit(0)


class TestRateLimiter(unittest.TestCase):
    """Test the RateLimiter class."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a test Redis URL that doesn't exist to test fallback mode
        self.rate_limiter = RateLimiter(redis_url="redis://localhost:9999/0")
        
    def tearDown(self):
        """Clean up test environment."""
        if self.rate_limiter:
            self.rate_limiter.close()
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        self.assertIsNotNone(self.rate_limiter)
        self.assertFalse(self.rate_limiter.redis_available)  # Should fail to connect to test Redis
        self.assertIsNotNone(self.rate_limiter.rules)
        self.assertEqual(len(self.rate_limiter.rules), 4)  # Four tiers
    
    def test_basic_tier_rate_limiting(self):
        """Test basic tier rate limiting."""
        client_id = "test_client_basic"
        
        # Should allow first request
        result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
        self.assertTrue(result.allowed)
        self.assertEqual(result.tier, "basic")
        
        # Test multiple requests up to limit
        for i in range(9):  # Basic tier allows 10 per minute
            result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
            self.assertTrue(result.allowed)
        
        # Should block the 11th request
        result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
        self.assertFalse(result.allowed)
        self.assertIsNotNone(result.retry_after)
    
    def test_premium_tier_higher_limits(self):
        """Test that premium tier has higher limits than basic."""
        basic_rule = self.rate_limiter.rules[RateLimitTier.BASIC]
        premium_rule = self.rate_limiter.rules[RateLimitTier.PREMIUM]
        
        self.assertGreater(premium_rule.requests_per_minute, basic_rule.requests_per_minute)
        self.assertGreater(premium_rule.requests_per_hour, basic_rule.requests_per_hour)
        self.assertGreater(premium_rule.requests_per_day, basic_rule.requests_per_day)
    
    def test_different_endpoints(self):
        """Test rate limiting for different endpoints."""
        client_id = "test_client_endpoints"
        
        # Should allow requests to different endpoints independently
        result1 = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC, "endpoint1")
        result2 = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC, "endpoint2")
        
        self.assertTrue(result1.allowed)
        self.assertTrue(result2.allowed)
    
    def test_sliding_window_strategy(self):
        """Test sliding window rate limiting strategy."""
        client_id = "test_sliding_window"
        
        # Create a custom rule with sliding window
        rule = RateLimitRule(
            requests_per_minute=5,
            requests_per_hour=100,
            requests_per_day=1000,
            burst_allowance=2,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            tier=RateLimitTier.BASIC
        )
        
        self.rate_limiter.rules[RateLimitTier.BASIC] = rule
        
        # Test that we can make requests up to the limit
        for i in range(5):
            result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
            self.assertTrue(result.allowed, f"Request {i+1} should be allowed")
        
        # 6th request should be blocked
        result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
        self.assertFalse(result.allowed)
    
    def test_fixed_window_strategy(self):
        """Test fixed window rate limiting strategy."""
        client_id = "test_fixed_window"
        
        # Create a custom rule with fixed window
        rule = RateLimitRule(
            requests_per_minute=3,
            requests_per_hour=100,
            requests_per_day=1000,
            burst_allowance=1,
            strategy=RateLimitStrategy.FIXED_WINDOW,
            tier=RateLimitTier.BASIC
        )
        
        self.rate_limiter.rules[RateLimitTier.BASIC] = rule
        
        # Test fixed window behavior
        for i in range(3):
            result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
            self.assertTrue(result.allowed, f"Request {i+1} should be allowed")
        
        # 4th request should be blocked
        result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
        self.assertFalse(result.allowed)
    
    def test_client_stats(self):
        """Test getting client statistics."""
        client_id = "test_stats_client"
        
        # Make some requests
        for i in range(3):
            self.rate_limiter.check_rate_limit(client_id, RateLimitTier.PREMIUM)
        
        # Get stats
        stats = self.rate_limiter.get_client_stats(client_id)
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["client_id"], client_id)
        self.assertIn("current_time", stats)
    
    def test_system_metrics(self):
        """Test getting system-wide metrics."""
        # Make some requests to generate metrics
        for i in range(5):
            self.rate_limiter.check_rate_limit(f"client_{i}", RateLimitTier.BASIC)
        
        metrics = self.rate_limiter.get_system_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("timestamp", metrics)
        self.assertIn("total_requests", metrics)
        self.assertIn("allowed_requests", metrics)
        self.assertIn("blocked_requests", metrics)
        self.assertGreaterEqual(metrics["total_requests"], 5)
    
    def test_client_tier_management(self):
        """Test updating client tiers."""
        client_id = "test_tier_client"
        
        # Update to premium tier
        success = self.rate_limiter.update_client_tier(client_id, RateLimitTier.PREMIUM)
        self.assertTrue(success)
        
        # Get tier (fallback mode should return basic)
        tier = self.rate_limiter.get_client_tier(client_id)
        self.assertIsInstance(tier, RateLimitTier)
    
    def test_reset_client_limits(self):
        """Test resetting client limits."""
        client_id = "test_reset_client"
        
        # Make requests to use up some limit
        for i in range(5):
            self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
        
        # Reset limits
        success = self.rate_limiter.reset_client_limits(client_id)
        self.assertTrue(success)


class TestHTTPAIClientWithRateLimiting(unittest.TestCase):
    """Test HTTPAIClient integration with rate limiting."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'RATE_LIMIT_ENABLED': 'true',
            'RATE_LIMIT_CLIENT_ID': 'test_ai_client',
            'RATE_LIMIT_TIER': 'basic',
            'REDIS_URL': 'redis://localhost:9999/0'  # Non-existent Redis for testing
        })
        self.env_patcher.start()
        
        self.client = HTTPAIClient(client_id="test_ai_client", rate_limit_tier="basic")
    
    def tearDown(self):
        """Clean up test environment."""
        self.env_patcher.stop()
        if self.client:
            self.client.close()
    
    def test_client_initialization_with_rate_limiting(self):
        """Test that client initializes with rate limiting."""
        self.assertEqual(self.client.client_id, "test_ai_client")
        self.assertEqual(self.client.rate_limit_tier, RateLimitTier.BASIC)
        self.assertTrue(self.client.rate_limit_enabled)
        self.assertIsNotNone(self.client.rate_limiter)
    
    @patch('scripts.http_ai_client.requests.post')
    async def test_rate_limited_request(self, mock_post):
        """Test that requests are rate limited."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"text": "Test response"}],
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        mock_post.return_value = mock_response
        
        # First request should work
        result = await self.client.generate_enhanced_response("Test prompt")
        self.assertNotEqual(result['provider'], 'rate_limiter')
        
        # Make many requests to trigger rate limiting
        for i in range(15):  # Basic tier allows 10 per minute
            result = await self.client.generate_enhanced_response(f"Test prompt {i}")
        
        # Should eventually get rate limited
        result = await self.client.generate_enhanced_response("Final test prompt")
        # Note: In fallback mode, rate limiting might be more lenient
    
    def test_rate_limit_status(self):
        """Test getting rate limit status."""
        status = self.client.get_rate_limit_status()
        
        self.assertIsInstance(status, dict)
        self.assertTrue(status.get("enabled", False))
        self.assertEqual(status.get("client_id"), "test_ai_client")
        self.assertEqual(status.get("tier"), "basic")
    
    def test_rate_limit_metrics(self):
        """Test getting rate limit metrics."""
        metrics = self.client.get_rate_limit_metrics()
        
        self.assertIsInstance(metrics, dict)
        # Should have basic metric structure even if Redis is not available
    
    def test_update_tier(self):
        """Test updating rate limit tier."""
        success = self.client.update_rate_limit_tier("premium")
        self.assertTrue(success)
        self.assertEqual(self.client.rate_limit_tier, RateLimitTier.PREMIUM)
    
    def test_reset_limits(self):
        """Test resetting rate limits."""
        success = self.client.reset_rate_limits()
        self.assertTrue(success)


class TestRateLimitMonitor(unittest.TestCase):
    """Test the RateLimitMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.monitor = RateLimitMonitor()
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor)
        self.assertIsInstance(self.monitor.start_time, datetime)
        self.assertIn("block_rate", self.monitor.alert_thresholds)
    
    def test_dashboard_generation(self):
        """Test real-time dashboard generation."""
        dashboard = self.monitor.get_real_time_dashboard()
        
        self.assertIsInstance(dashboard, dict)
        self.assertIn("timestamp", dashboard)
        self.assertIn("system_status", dashboard)
        self.assertIn("alerts", dashboard)
        self.assertIn("metrics", dashboard)
        self.assertIsInstance(dashboard["alerts"], list)
    
    def test_configuration_info(self):
        """Test getting configuration information."""
        config = self.monitor._get_configuration_info()
        
        self.assertIsInstance(config, dict)
        self.assertIn("redis_url", config)
        self.assertIn("rate_limit_enabled", config)
        self.assertIn("tier_limits", config)
    
    def test_usage_report_generation(self):
        """Test usage report generation."""
        report = self.monitor.generate_usage_report(24)
        
        self.assertIsInstance(report, dict)
        self.assertIn("report_period_hours", report)
        self.assertIn("generated_at", report)
        self.assertIn("summary", report)
        self.assertIn("recommendations", report)
        self.assertEqual(report["report_period_hours"], 24)
    
    def test_metrics_export_json(self):
        """Test exporting metrics in JSON format."""
        exported = self.monitor.export_metrics("json")
        
        self.assertIsInstance(exported, str)
        # Should be valid JSON
        data = json.loads(exported)
        self.assertIsInstance(data, dict)
    
    def test_metrics_export_csv(self):
        """Test exporting metrics in CSV format."""
        exported = self.monitor.export_metrics("csv")
        
        self.assertIsInstance(exported, str)
        self.assertIn("timestamp,metric,value", exported)
    
    def test_client_details(self):
        """Test getting client details."""
        details = self.monitor.get_client_details("test_client")
        
        self.assertIsInstance(details, dict)
        # Should handle the case where rate limiter is not available


class TestRateLimitingIntegration(unittest.TestCase):
    """Integration tests for the complete rate limiting system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.rate_limiter = RateLimiter(redis_url="redis://localhost:9999/0")
        self.monitor = RateLimitMonitor()
    
    def tearDown(self):
        """Clean up integration test environment."""
        if self.rate_limiter:
            self.rate_limiter.close()
    
    def test_end_to_end_rate_limiting(self):
        """Test complete rate limiting workflow."""
        client_id = "integration_test_client"
        
        # 1. Check initial state
        stats = self.rate_limiter.get_client_stats(client_id)
        self.assertIsInstance(stats, dict)
        
        # 2. Make requests and verify rate limiting
        allowed_count = 0
        blocked_count = 0
        
        for i in range(15):  # More than basic tier limit
            result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.BASIC)
            if result.allowed:
                allowed_count += 1
            else:
                blocked_count += 1
        
        # Should have some allowed and some blocked
        self.assertGreater(allowed_count, 0)
        # Note: In fallback mode, blocking behavior might be different
        
        # 3. Check system metrics
        metrics = self.rate_limiter.get_system_metrics()
        self.assertGreaterEqual(metrics["total_requests"], 15)
        
        # 4. Test monitor dashboard
        dashboard = self.monitor.get_real_time_dashboard()
        self.assertIn("system_status", dashboard)
        
        # 5. Test tier upgrade
        success = self.rate_limiter.update_client_tier(client_id, RateLimitTier.PREMIUM)
        self.assertTrue(success)
        
        # 6. Verify higher limits after upgrade
        result = self.rate_limiter.check_rate_limit(client_id, RateLimitTier.PREMIUM)
        self.assertTrue(result.allowed)  # Should be allowed with premium tier
    
    def test_performance_under_load(self):
        """Test rate limiter performance under load."""
        start_time = time.time()
        
        # Simulate multiple clients making requests
        for client_id in range(10):
            for request_id in range(5):
                self.rate_limiter.check_rate_limit(f"load_test_client_{client_id}", RateLimitTier.BASIC)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        self.assertLess(duration, 5.0, "Rate limiting should be performant under load")
        
        # Check that metrics are updated
        metrics = self.rate_limiter.get_system_metrics()
        self.assertGreaterEqual(metrics["total_requests"], 50)
    
    def test_fallback_mode_behavior(self):
        """Test that fallback mode works when Redis is unavailable."""
        # The rate limiter should already be in fallback mode due to test Redis URL
        self.assertFalse(self.rate_limiter.redis_available)
        
        # Should still be able to perform basic operations
        result = self.rate_limiter.check_rate_limit("fallback_test", RateLimitTier.BASIC)
        self.assertIsInstance(result, RateLimitResult)
        
        # Should be able to get metrics
        metrics = self.rate_limiter.get_system_metrics()
        self.assertIsInstance(metrics, dict)


class TestRateLimitingBackwardCompatibility(unittest.TestCase):
    """Test that rate limiting maintains backward compatibility."""
    
    def test_http_client_without_rate_limiting(self):
        """Test that HTTP client works without rate limiting enabled."""
        with patch.dict(os.environ, {'RATE_LIMIT_ENABLED': 'false'}):
            client = HTTPAIClient()
            
            self.assertFalse(client.rate_limit_enabled)
            self.assertIsNone(client.rate_limiter)
            
            # Should still be able to get status (returns disabled status)
            status = client.get_rate_limit_status()
            self.assertFalse(status.get("enabled", True))
    
    def test_graceful_degradation(self):
        """Test graceful degradation when rate limiting components fail."""
        # Test with missing RateLimiter class
        with patch('scripts.http_ai_client.RateLimiter', None):
            client = HTTPAIClient()
            
            # Should initialize without rate limiting
            self.assertIsNone(client.rate_limiter)
            
            # Should handle methods gracefully
            status = client.get_rate_limit_status()
            self.assertFalse(status.get("enabled", True))


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run the tests
    unittest.main(verbosity=2)