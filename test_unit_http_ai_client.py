#!/usr/bin/env python3
"""
Comprehensive unit tests for HTTP AI Client module.

Tests cover:
- HTTP client initialization and configuration
- API request handling
- Error handling and retries
- Response parsing
- Authentication mechanisms
- External dependency mocking
"""

import unittest
import json
import requests
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.http_ai_client import HTTPAIClient


class TestHTTPAIClientInitialization(unittest.TestCase):
    """Test HTTPAIClient initialization."""
    
    def test_init_with_default_parameters(self):
        """Test HTTPAIClient initialization with default parameters."""
        # Arrange & Act
        client = HTTPAIClient()
        
        # Assert
        self.assertIsNotNone(client.session)
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 3)
        self.assertIsInstance(client.headers, dict)
    
    def test_init_with_custom_parameters(self):
        """Test HTTPAIClient initialization with custom parameters."""
        # Arrange
        api_key = "test_api_key"
        base_url = "https://api.test.com"
        timeout = 60
        max_retries = 5
        
        # Act
        client = HTTPAIClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Assert
        self.assertEqual(client.api_key, api_key)
        self.assertEqual(client.base_url, base_url)
        self.assertEqual(client.timeout, timeout)
        self.assertEqual(client.max_retries, max_retries)
    
    def test_init_sets_proper_headers(self):
        """Test that initialization sets proper HTTP headers."""
        # Arrange
        api_key = "test_key_123"
        
        # Act
        client = HTTPAIClient(api_key=api_key)
        
        # Assert
        self.assertIn("Authorization", client.headers)
        self.assertIn("Content-Type", client.headers)
        self.assertEqual(client.headers["Content-Type"], "application/json")


class TestHTTPAIClientRequests(unittest.TestCase):
    """Test HTTP request functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = HTTPAIClient(api_key="test_key")
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_request_success(self, mock_post):
        """Test successful POST request."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Act
        response = self.client.post("/test", {"key": "value"})
        
        # Assert
        self.assertEqual(response["result"], "success")
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
    
    @patch('scripts.http_ai_client.requests.Session.get')
    def test_get_request_success(self, mock_get):
        """Test successful GET request."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test_data"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Act
        response = self.client.get("/test")
        
        # Assert
        self.assertEqual(response["data"], "test_data")
        mock_get.assert_called_once()
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_request_with_retry_on_failure(self, mock_post):
        """Test POST request with retry mechanism on failure."""
        # Arrange
        # First call fails, second succeeds
        failed_response = Mock()
        failed_response.status_code = 500
        failed_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"result": "success"}
        success_response.raise_for_status = Mock()
        
        mock_post.side_effect = [failed_response, success_response]
        
        # Act
        response = self.client.post("/test", {"key": "value"})
        
        # Assert
        self.assertEqual(response["result"], "success")
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_request_max_retries_exceeded(self, mock_post):
        """Test POST request when max retries are exceeded."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
        mock_post.return_value = mock_response
        
        # Act & Assert
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.post("/test", {"key": "value"})
        
        # Should retry max_retries times
        self.assertEqual(mock_post.call_count, self.client.max_retries)


class TestHTTPAIClientErrorHandling(unittest.TestCase):
    """Test error handling in HTTP AI Client."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = HTTPAIClient(api_key="test_key")
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_request_connection_error(self, mock_post):
        """Test POST request with connection error."""
        # Arrange
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        # Act & Assert
        with self.assertRaises(requests.exceptions.ConnectionError):
            self.client.post("/test", {"key": "value"})
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_request_timeout_error(self, mock_post):
        """Test POST request with timeout error."""
        # Arrange
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        # Act & Assert
        with self.assertRaises(requests.exceptions.Timeout):
            self.client.post("/test", {"key": "value"})
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_request_invalid_json_response(self, mock_post):
        """Test POST request with invalid JSON response."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Act & Assert
        with self.assertRaises(json.JSONDecodeError):
            self.client.post("/test", {"key": "value"})
    
    def test_validate_response_with_invalid_status_code(self):
        """Test response validation with invalid status code."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
        
        # Act & Assert
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client._validate_response(mock_response)


class TestHTTPAIClientAuthentication(unittest.TestCase):
    """Test authentication mechanisms."""
    
    def test_bearer_token_authentication(self):
        """Test Bearer token authentication header."""
        # Arrange
        api_key = "test_bearer_token"
        
        # Act
        client = HTTPAIClient(api_key=api_key, auth_type="bearer")
        
        # Assert
        self.assertEqual(client.headers["Authorization"], f"Bearer {api_key}")
    
    def test_api_key_authentication(self):
        """Test API key authentication header."""
        # Arrange
        api_key = "test_api_key"
        
        # Act
        client = HTTPAIClient(api_key=api_key, auth_type="api_key")
        
        # Assert
        self.assertEqual(client.headers["X-API-Key"], api_key)
    
    def test_custom_authentication(self):
        """Test custom authentication header."""
        # Arrange
        api_key = "test_custom_key"
        custom_headers = {"Custom-Auth": f"Custom {api_key}"}
        
        # Act
        client = HTTPAIClient(api_key=api_key, custom_headers=custom_headers)
        
        # Assert
        self.assertEqual(client.headers["Custom-Auth"], f"Custom {api_key}")


class TestHTTPAIClientResponseParsing(unittest.TestCase):
    """Test response parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = HTTPAIClient(api_key="test_key")
    
    def test_parse_json_response_success(self):
        """Test successful JSON response parsing."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {"key": "value"},
            "status": "success"
        }
        
        # Act
        parsed_data = self.client._parse_response(mock_response)
        
        # Assert
        self.assertEqual(parsed_data["status"], "success")
        self.assertEqual(parsed_data["data"]["key"], "value")
    
    def test_parse_response_with_error_field(self):
        """Test parsing response that contains error field."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {
            "error": "Invalid request",
            "code": 400
        }
        
        # Act
        parsed_data = self.client._parse_response(mock_response)
        
        # Assert
        self.assertIn("error", parsed_data)
        self.assertEqual(parsed_data["error"], "Invalid request")
    
    def test_parse_empty_response(self):
        """Test parsing empty response."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {}
        
        # Act
        parsed_data = self.client._parse_response(mock_response)
        
        # Assert
        self.assertEqual(parsed_data, {})


class TestHTTPAIClientConfiguration(unittest.TestCase):
    """Test configuration and settings."""
    
    def test_set_timeout(self):
        """Test setting custom timeout."""
        # Arrange
        client = HTTPAIClient()
        new_timeout = 120
        
        # Act
        client.set_timeout(new_timeout)
        
        # Assert
        self.assertEqual(client.timeout, new_timeout)
    
    def test_set_max_retries(self):
        """Test setting custom max retries."""
        # Arrange
        client = HTTPAIClient()
        new_max_retries = 10
        
        # Act
        client.set_max_retries(new_max_retries)
        
        # Assert
        self.assertEqual(client.max_retries, new_max_retries)
    
    def test_add_custom_header(self):
        """Test adding custom header."""
        # Arrange
        client = HTTPAIClient()
        header_name = "X-Custom-Header"
        header_value = "custom_value"
        
        # Act
        client.add_header(header_name, header_value)
        
        # Assert
        self.assertEqual(client.headers[header_name], header_value)
    
    def test_remove_header(self):
        """Test removing header."""
        # Arrange
        client = HTTPAIClient()
        header_name = "X-Remove-Me"
        client.headers[header_name] = "value"
        
        # Act
        client.remove_header(header_name)
        
        # Assert
        self.assertNotIn(header_name, client.headers)


class TestHTTPAIClientSpecialMethods(unittest.TestCase):
    """Test special methods and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = HTTPAIClient(api_key="test_key")
    
    def test_context_manager_usage(self):
        """Test using client as context manager."""
        # Act & Assert
        with HTTPAIClient(api_key="test_key") as client:
            self.assertIsNotNone(client.session)
        
        # Session should be closed after context exit
        # (This would need to be implemented in the actual class)
    
    def test_client_with_ssl_verification_disabled(self):
        """Test client with SSL verification disabled."""
        # Arrange & Act
        client = HTTPAIClient(api_key="test_key", verify_ssl=False)
        
        # Assert
        self.assertFalse(client.verify_ssl)
    
    def test_client_with_custom_user_agent(self):
        """Test client with custom user agent."""
        # Arrange
        custom_user_agent = "TestBot/1.0"
        
        # Act
        client = HTTPAIClient(api_key="test_key", user_agent=custom_user_agent)
        
        # Assert
        self.assertEqual(client.headers["User-Agent"], custom_user_agent)
    
    @patch('scripts.http_ai_client.requests.Session.post')
    def test_post_with_files(self, mock_post):
        """Test POST request with file uploads."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"file_uploaded": True}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        files = {"file": ("test.txt", "file content", "text/plain")}
        
        # Act
        response = self.client.post_with_files("/upload", files=files)
        
        # Assert
        self.assertTrue(response["file_uploaded"])
        mock_post.assert_called_once()


class TestHTTPAIClientPerformance(unittest.TestCase):
    """Test performance-related functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = HTTPAIClient(api_key="test_key")
    
    @patch('scripts.http_ai_client.time.sleep')
    def test_exponential_backoff_retry(self, mock_sleep):
        """Test exponential backoff in retry mechanism."""
        # Arrange
        with patch('scripts.http_ai_client.requests.Session.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429  # Rate limited
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Rate Limited")
            mock_post.return_value = mock_response
            
            # Act
            try:
                self.client.post("/test", {"key": "value"})
            except requests.exceptions.HTTPError:
                pass
            
            # Assert - Should have called sleep for backoff
            self.assertTrue(mock_sleep.called)
    
    def test_request_statistics_tracking(self):
        """Test that client tracks request statistics."""
        # Arrange
        client = HTTPAIClient(api_key="test_key", track_stats=True)
        
        # Act
        stats = client.get_request_stats()
        
        # Assert
        self.assertIn("total_requests", stats)
        self.assertIn("successful_requests", stats)
        self.assertIn("failed_requests", stats)
        self.assertIn("average_response_time", stats)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)