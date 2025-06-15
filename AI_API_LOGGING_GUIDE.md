# AI API Communication Logging Guide

This guide explains how to use the AI API communication logging system to monitor all interactions with AI providers in real-time.

## Overview

The AI API logging system provides comprehensive monitoring of all AI API communications, including:
- Request/response tracking
- Performance metrics
- Error monitoring
- Cache performance
- Cost tracking
- Provider usage statistics

## Components

### 1. AI API Logger (`scripts/ai_api_logger.py`)
The core logging module that:
- Captures all AI API requests and responses
- Writes structured JSON logs to `ai_api_communication.log`
- Tracks performance metrics and statistics
- Supports privacy modes (can redact sensitive data)

### 2. AI API Log Viewer (`scripts/ai_api_log_viewer.py`)
Interactive tool for viewing logs in real-time:
```bash
# Follow logs in real-time
python scripts/ai_api_log_viewer.py -f

# Show only errors
python scripts/ai_api_log_viewer.py --event request_error

# Filter by provider
python scripts/ai_api_log_viewer.py --provider anthropic -f

# Verbose mode with content previews
python scripts/ai_api_log_viewer.py -f -v

# Show statistics only
python scripts/ai_api_log_viewer.py --stats --no-follow
```

## Log Format

Each log entry is a JSON object with the following structure:

### Request Start
```json
{
  "event_type": "request_start",
  "request_metadata": {
    "request_id": "req_123",
    "timestamp": "2024-01-01T12:00:00Z",
    "provider": "anthropic",
    "model": "claude-3-7-sonnet",
    "request_type": "generate",
    "prompt_length": 500,
    "prompt_hash": "a1b2c3d4",
    "distributed": false,
    "cache_enabled": true
  },
  "prompt_preview": "First 500 chars of prompt..."
}
```

### Request Complete
```json
{
  "event_type": "request_complete",
  "request_id": "req_123",
  "timestamp": "2024-01-01T12:00:02Z",
  "response_metadata": {
    "response_length": 1200,
    "response_time": 2.1,
    "cached": false,
    "confidence": 0.9,
    "cost_estimate": 0.0025,
    "token_usage": {
      "input_tokens": 150,
      "output_tokens": 400,
      "total_tokens": 550
    }
  },
  "provider": "anthropic",
  "model": "claude-3-7-sonnet"
}
```

## Configuration

### Environment Variables

- `AI_API_LOG_LEVEL`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `AI_API_LOG_SENSITIVE`: Set to "true" to log full prompts/responses (default: false)
- `AI_API_MAX_LOG_LENGTH`: Maximum length of logged content (default: 500)
- `AI_API_FILE_LOGGING`: Set to "false" to disable file logging (default: true)

Example:
```bash
export AI_API_LOG_LEVEL=DEBUG
export AI_API_LOG_SENSITIVE=true
export AI_API_MAX_LOG_LENGTH=1000
```

## Real-time Monitoring

### Basic Monitoring
```bash
# Watch all AI API communications
tail -f ai_api_communication.log | jq '.'

# Using the dedicated viewer
python scripts/ai_api_log_viewer.py -f
```

### Advanced Filtering
```bash
# Monitor only Anthropic requests
python scripts/ai_api_log_viewer.py -f --provider anthropic

# Watch for errors
python scripts/ai_api_log_viewer.py -f --event request_error

# Get current statistics
python scripts/ai_api_log_viewer.py --stats --no-follow
```

## Integration

The AI API logger is automatically integrated with:
- `HTTPAIClient` in `scripts/http_ai_client.py`
- `EnhancedHTTPAIClient` in `scripts/enhanced_http_ai_client.py`

All AI API calls made through these clients are automatically logged.

## Performance Impact

The logging system is designed for minimal performance impact:
- Asynchronous file writing via background thread
- Efficient JSON serialization
- Optional content truncation
- Configurable logging levels

## Privacy and Security

- By default, full prompts and responses are NOT logged
- Only hashes and lengths are recorded for privacy
- Enable `AI_API_LOG_SENSITIVE=true` only in development
- Log files should be excluded from version control

## Troubleshooting

### No logs appearing
1. Check if `AI_API_FILE_LOGGING` is set to "true"
2. Verify the log file path has write permissions
3. Check if AI clients are properly initialized

### Performance issues
1. Reduce `AI_API_MAX_LOG_LENGTH`
2. Set `AI_API_LOG_LEVEL` to "WARNING" or "ERROR"
3. Disable sensitive data logging

### Log file too large
The system doesn't include automatic rotation. For production:
```bash
# Set up logrotate
sudo tee /etc/logrotate.d/ai_api_logs << EOF
/workspaces/cwmai/ai_api_communication.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 $USER $USER
}
EOF
```

## Example Usage

### Monitor Claude API Success Rate
```bash
# Get Claude-specific statistics
python scripts/ai_api_log_viewer.py --provider anthropic --stats
```

### Debug API Errors
```bash
# Watch for errors in real-time with verbose output
python scripts/ai_api_log_viewer.py -f -v --event request_error
```

### Track Costs
```bash
# Parse logs for cost analysis
cat ai_api_communication.log | jq -r '
  select(.event_type == "request_complete") |
  "\(.provider) \(.model): $\(.response_metadata.cost_estimate)"
' | sort | uniq -c
```