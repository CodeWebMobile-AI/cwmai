#!/usr/bin/env python3
"""
Debug structured output handler.
"""

import json
from scripts.structured_output_handler import StructuredOutputHandler

# Test the handler directly
handler = StructuredOutputHandler()

# Test schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
}

# Test each provider
for provider in ['gemini', 'openai', 'deepseek', 'anthropic']:
    print(f"\n{provider.upper()} Configuration:")
    config = handler.prepare_structured_output(
        provider, 
        "Extract person info",
        schema
    )
    print(json.dumps(config, indent=2))