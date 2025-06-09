"""
Unit tests for HTTPAIClient module.

Tests HTTP AI provider integration, response handling, and error scenarios.
Follows AAA pattern with comprehensive mocking of external dependencies.
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import aiohttp

from http_ai_client import HTTPAIClient


class TestHTTPAIClient:
    """Test suite for HTTPAIClient class."""

    def test_init_no_api_keys(self):
        """Test HTTPAIClient initialization without API keys."""
        # Arrange & Act
        with patch.dict('os.environ', {}, clear=True):
            client = HTTPAIClient()
        
        # Assert
        assert client.anthropic_api_key is None
        assert client.openai_api_key is None
        assert client.gemini_api_key is None
        assert client.deepseek_api_key is None
        assert all(not available for available in client.providers_available.values())

    def test_init_with_api_keys(self, mock_api_keys):
        """Test HTTPAIClient initialization with API keys."""
        # Arrange & Act
        with patch.dict('os.environ', mock_api_keys):
            client = HTTPAIClient()
        
        # Assert
        assert client.anthropic_api_key == 'test_anthropic_key'
        assert client.openai_api_key == 'test_openai_key'
        assert client.gemini_api_key == 'test_gemini_key'
        assert client.deepseek_api_key == 'test_deepseek_key'
        assert all(client.providers_available.values())

    def test_sanitize_headers_with_sensitive_data(self):
        """Test header sanitization hides sensitive information."""
        # Arrange
        client = HTTPAIClient()
        headers = {
            'Authorization': 'Bearer secret_token',
            'X-API-Key': 'secret_key',
            'Content-Type': 'application/json',
            'User-Agent': 'TestAgent'
        }
        
        # Act
        sanitized = client._sanitize_headers(headers)
        
        # Assert
        assert sanitized['Authorization'] == '***'
        assert sanitized['X-API-Key'] == '***'
        assert sanitized['Content-Type'] == 'application/json'
        assert sanitized['User-Agent'] == 'TestAgent'

    def test_sanitize_headers_no_sensitive_data(self):
        """Test header sanitization with no sensitive data."""
        # Arrange
        client = HTTPAIClient()
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'TestAgent',
            'Accept': 'application/json'
        }
        
        # Act
        sanitized = client._sanitize_headers(headers)
        
        # Assert
        assert sanitized == headers

    @pytest.mark.asyncio
    async def test_generate_enhanced_response_mock_fallback(self):
        """Test enhanced response generation with mock fallback."""
        # Arrange
        client = HTTPAIClient()
        prompt = "Test prompt"
        
        # Act
        response = await client.generate_enhanced_response(prompt)
        
        # Assert
        assert response['content'] == 'Mock AI response - no providers available'
        assert response['provider'] == 'mock'
        assert response['confidence'] == 0.1

    @pytest.mark.asyncio
    async def test_generate_enhanced_response_with_claude_preference(self):
        """Test enhanced response generation with Claude preference."""
        # Arrange
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        with patch.object(client, '_call_anthropic_http') as mock_anthropic:
            mock_anthropic.return_value = {
                'content': 'Claude response',
                'provider': 'anthropic',
                'confidence': 0.9
            }
            
            # Act
            response = await client.generate_enhanced_response("test", model='claude')
            
            # Assert
            assert response['content'] == 'Claude response'
            assert response['provider'] == 'anthropic'
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_enhanced_response_with_gpt_preference(self):
        """Test enhanced response generation with GPT preference."""
        # Arrange
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        with patch.object(client, '_call_openai_http') as mock_openai:
            mock_openai.return_value = {
                'content': 'GPT response',
                'provider': 'openai',
                'confidence': 0.85
            }
            
            # Act
            response = await client.generate_enhanced_response("test", model='gpt')
            
            # Assert
            assert response['content'] == 'GPT response'
            assert response['provider'] == 'openai'
            mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_anthropic_http_success(self):
        """Test successful Anthropic API call."""
        # Arrange
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        mock_response_data = {
            'content': [{'text': 'Test response from Claude'}],
            'usage': {'input_tokens': 10, 'output_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_anthropic_http("test prompt", "req_001")
            
            # Assert
            assert result['content'] == 'Test response from Claude'
            assert result['provider'] == 'anthropic'
            assert result['input_tokens'] == 10
            assert result['output_tokens'] == 5

    @pytest.mark.asyncio
    async def test_call_anthropic_http_error_response(self):
        """Test Anthropic API call with error response."""
        # Arrange
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text.return_value = 'Bad Request'
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_anthropic_http("test prompt", "req_001")
            
            # Assert
            assert 'error' in result
            assert result['provider'] == 'anthropic'

    @pytest.mark.asyncio
    async def test_call_openai_http_success(self):
        """Test successful OpenAI API call."""
        # Arrange
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        mock_response_data = {
            'choices': [{'message': {'content': 'Test response from GPT'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_openai_http("test prompt", "req_001")
            
            # Assert
            assert result['content'] == 'Test response from GPT'
            assert result['provider'] == 'openai'
            assert result['input_tokens'] == 10
            assert result['output_tokens'] == 5

    @pytest.mark.asyncio
    async def test_call_gemini_http_success(self):
        """Test successful Gemini API call."""
        # Arrange
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        mock_response_data = {
            'candidates': [{'content': {'parts': [{'text': 'Test response from Gemini'}]}}],
            'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_gemini_http("test prompt", "req_001")
            
            # Assert
            assert result['content'] == 'Test response from Gemini'
            assert result['provider'] == 'gemini'
            assert result['input_tokens'] == 10
            assert result['output_tokens'] == 5

    @pytest.mark.asyncio
    async def test_call_deepseek_http_success(self):
        """Test successful DeepSeek API call."""
        # Arrange
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        mock_response_data = {
            'choices': [{'message': {'content': 'Test response from DeepSeek'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}
        }
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_deepseek_http("test prompt", "req_001")
            
            # Assert
            assert result['content'] == 'Test response from DeepSeek'
            assert result['provider'] == 'deepseek'
            assert result['input_tokens'] == 10
            assert result['output_tokens'] == 5

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        # Arrange
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Request timed out")
            
            # Act
            result = await client._call_anthropic_http("test prompt", "req_001")
            
            # Assert
            assert 'error' in result
            assert 'timeout' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of connection errors."""
        # Arrange
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = aiohttp.ClientConnectorError(
                connection_key=None, os_error=OSError("Connection failed")
            )
            
            # Act
            result = await client._call_openai_http("test prompt", "req_001")
            
            # Assert
            assert 'error' in result
            assert 'connection' in result['error'].lower()

    def test_performance_metrics_tracking(self):
        """Test performance metrics are tracked correctly."""
        # Arrange
        client = HTTPAIClient()
        initial_count = client.request_count
        initial_error_count = client.error_count
        
        # Act
        client.request_count += 1
        client.error_count += 1
        client.total_response_time += 1.5
        
        # Assert
        assert client.request_count == initial_count + 1
        assert client.error_count == initial_error_count + 1
        assert client.total_response_time == 1.5

    def test_debug_mode_toggle(self):
        """Test debug mode can be toggled."""
        # Arrange
        client = HTTPAIClient()
        initial_debug = client.debug_mode
        
        # Act
        client.debug_mode = not initial_debug
        
        # Assert
        assert client.debug_mode != initial_debug

    @pytest.mark.asyncio
    async def test_response_parsing_edge_cases(self):
        """Test response parsing handles edge cases."""
        # Arrange
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        # Test empty response
        mock_response_data = {'content': []}
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = mock_response_data
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_anthropic_http("test prompt", "req_001")
            
            # Assert
            assert 'error' in result or result['content'] == ''

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test handling of rate limit responses."""
        # Arrange
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = HTTPAIClient()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 429
            mock_response.text.return_value = 'Rate limit exceeded'
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Act
            result = await client._call_openai_http("test prompt", "req_001")
            
            # Assert
            assert 'error' in result
            assert 'rate limit' in result['error'].lower()

    def test_provider_availability_check(self):
        """Test provider availability checking."""
        # Arrange
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}, clear=True):
            client = HTTPAIClient()
        
        # Act & Assert
        assert client.providers_available['anthropic'] is True
        assert client.providers_available['openai'] is False
        assert client.providers_available['gemini'] is False
        assert client.providers_available['deepseek'] is False