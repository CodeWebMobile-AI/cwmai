# CWMAI API Documentation

## Overview

The CWMAI API provides sophisticated rate limiting with Redis for the autonomous AI task management system. It offers RESTful endpoints, WebSocket connections, and comprehensive monitoring capabilities.

## Features

- **Advanced Rate Limiting**: Multiple algorithms (sliding window, fixed window, token bucket)
- **Redis Integration**: Persistent storage with automatic fallback to in-memory
- **Real-time Updates**: WebSocket support for live system monitoring
- **Performance Monitoring**: Detailed metrics and health checks
- **AI Integration**: Rate-limited access to AI providers
- **Task Management**: Create and manage development tasks
- **Scalable Architecture**: Designed for high-throughput applications

## Quick Start

### Prerequisites

- Python 3.11+
- Redis Server (optional - will fallback to in-memory)
- Required environment variables:
  ```bash
  export ANTHROPIC_API_KEY="your-anthropic-key"
  export REDIS_HOST="localhost"  # optional
  export REDIS_PORT="6379"       # optional
  export REDIS_PASSWORD=""       # optional
  ```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (optional)
redis-server

# Run the API server
python api_server.py

# Or with custom settings
python api_server.py --host 0.0.0.0 --port 8000
```

### Access

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

## API Endpoints

### Core Endpoints

#### `GET /`
Get API information and available features.

**Response:**
```json
{
  "name": "CWMAI API",
  "version": "1.0.0",
  "description": "Autonomous AI Task Management System API",
  "features": ["Redis-based rate limiting", "AI provider integration", ...],
  "endpoints": {...}
}
```

#### `GET /health`
Health check with component status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-09T15:24:45Z",
  "redis": "connected",
  "ai_providers": {"anthropic": true, "openai": false},
  "uptime": 3600.5
}
```

#### `GET /status`
**Rate Limit**: 30 requests/minute

Comprehensive system status with optional metrics and task information.

**Query Parameters:**
- `include_metrics` (bool): Include performance metrics (default: true)
- `include_tasks` (bool): Include task information (default: true)

**Response:**
```json
{
  "timestamp": "2025-06-09T15:24:45Z",
  "system": "CWMAI",
  "api_version": "1.0.0",
  "status": "operational",
  "metrics": {...},
  "tasks": {...}
}
```

#### `GET /metrics`
**Rate Limit**: 60 requests/minute

Detailed API performance metrics.

**Response:**
```json
{
  "api_metrics": {
    "total_requests": 1000,
    "successful_requests": 950,
    "failed_requests": 30,
    "rate_limited_requests": 20,
    "average_response_time": 0.125,
    "uptime_seconds": 3600.5
  },
  "redis_metrics": {...},
  "rate_limit_config": {...}
}
```

### AI Endpoints

#### `POST /ai/generate`
**Rate Limit**: 10 requests/minute

Generate AI responses with rate limiting protection.

**Request Body:**
```json
{
  "prompt": "Your prompt here",
  "model": "claude",  // optional: claude, gpt, gemini, deepseek
  "max_tokens": 4000,  // optional
  "temperature": 0.7   // optional
}
```

**Response:**
```json
{
  "content": "AI generated response...",
  "provider": "anthropic",
  "model": "claude-3-7-sonnet",
  "confidence": 0.9,
  "request_id": "req_123",
  "response_time": 1.234,
  "timestamp": "2025-06-09T15:24:45Z"
}
```

### Task Management

#### `POST /tasks`
**Rate Limit**: 20 requests/minute

Create a new development task.

**Request Body:**
```json
{
  "title": "Implement new feature",
  "description": "Detailed task description",
  "priority": "medium",  // low, medium, high
  "task_type": "feature",  // feature, bug, docs, test
  "estimated_hours": 4.0
}
```

**Response:**
```json
{
  "id": "TASK-1733760285",
  "title": "Implement new feature",
  "description": "Detailed task description",
  "priority": "medium",
  "type": "feature",
  "estimated_hours": 4.0,
  "status": "pending",
  "created_at": "2025-06-09T15:24:45Z"
}
```

#### `GET /tasks`
**Rate Limit**: 60 requests/minute

List tasks with pagination.

**Query Parameters:**
- `limit` (int): Number of tasks to return (default: 10)
- `offset` (int): Number of tasks to skip (default: 0)

**Response:**
```json
{
  "tasks": [...],
  "total": 50,
  "limit": 10,
  "offset": 0
}
```

### WebSocket Endpoint

#### `WebSocket /ws`
**Rate Limit**: 100 requests/minute

Real-time updates and bidirectional communication.

**Connection Message:**
```json
{
  "type": "connection_established",
  "timestamp": "2025-06-09T15:24:45Z",
  "client_count": 5
}
```

**Ping/Pong:**
```json
// Send
{"type": "ping"}

// Receive
{"type": "pong", "timestamp": "2025-06-09T15:24:45Z"}
```

**Update Messages:**
```json
{
  "type": "ai_response",
  "timestamp": "2025-06-09T15:24:45Z",
  "provider": "anthropic",
  "success": true
}
```

## Rate Limiting

### Rate Limit Rules

| Endpoint | Limit | Window | Algorithm |
|----------|-------|--------|-----------|
| General API | 60 requests | 1 minute | Sliding Window |
| AI Generate | 10 requests | 1 minute | Sliding Window |
| Tasks | 20 requests | 1 minute | Sliding Window |
| WebSocket | 100 requests | 1 minute | Sliding Window |
| Burst | 10 requests | 1 second | Token Bucket |

### Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1733760345
X-RateLimit-Rule: api_general
Retry-After: 60  // Only when rate limited
```

### Rate Limit Exceeded

When rate limits are exceeded, the API returns:

```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests",
  "retry_after": 60
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_HOST` | Redis server hostname | localhost |
| `REDIS_PORT` | Redis server port | 6379 |
| `REDIS_PASSWORD` | Redis authentication | None |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | Required |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `GEMINI_API_KEY` | Google Gemini API key | Optional |

### Rate Limit Configuration

Rate limits can be customized by modifying the `RateLimitConfig` in `api_server.py`:

```python
rate_config = RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000,
    ai_requests_per_minute=10,
    ai_requests_per_hour=100,
    burst_requests=10
)
```

## Performance and Monitoring

### Benchmarking

Run performance benchmarks:

```bash
python scripts/api_performance_benchmarks.py
```

This will test:
- Basic endpoint performance
- Rate limiting effectiveness
- Concurrent request handling
- Cache performance
- Memory usage
- Stress testing

### Testing

Run the comprehensive test suite:

```bash
python test_api_rate_limiting.py
```

Tests cover:
- Rate limiting algorithms
- Cache functionality
- API endpoints
- Error handling
- Integration scenarios

### Monitoring

The API provides comprehensive monitoring through:

1. **Health Checks**: `/health` endpoint
2. **Metrics**: `/metrics` endpoint with detailed statistics
3. **Logging**: Structured logging with request tracking
4. **WebSocket Updates**: Real-time system events

## Architecture

### Components

1. **FastAPI Application**: High-performance async web framework
2. **Rate Limiter**: Multi-algorithm rate limiting with Redis persistence
3. **Cache Manager**: Intelligent caching with TTL and tagging
4. **AI Client**: HTTP-based AI provider integration
5. **WebSocket Manager**: Real-time bidirectional communication

### Storage

- **Primary**: Redis for persistence and scalability
- **Fallback**: In-memory storage when Redis unavailable
- **Hybrid**: Automatic fallback ensures high availability

### Security

- API key validation for AI providers
- Rate limiting prevents abuse
- Request sanitization and validation
- Comprehensive error handling

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

### Error Response Format

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2025-06-09T15:24:45Z",
  "request_id": "req_123"
}
```

## Integration with CWMAI

The API seamlessly integrates with the existing CWMAI autonomous task management system:

- **AI Brain Integration**: Leverages existing AI provider connections
- **Task Management**: Connects with existing task creation and tracking
- **GitHub Integration**: Maintains compatibility with GitHub workflows
- **State Management**: Integrates with system state persistence

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Setup

```bash
# Production environment variables
export REDIS_HOST=redis.production.com
export REDIS_PORT=6379
export REDIS_PASSWORD=secure_password
export ANTHROPIC_API_KEY=prod_key
```

### Scaling Considerations

- Redis clustering for high availability
- Load balancing with multiple API instances
- Connection pooling for Redis
- Monitoring and alerting setup

## Support

For issues, questions, or contributions:

1. Check the comprehensive test suite for examples
2. Run benchmarks to understand performance characteristics
3. Review the monitoring endpoints for system health
4. Consult the existing CWMAI documentation for system integration