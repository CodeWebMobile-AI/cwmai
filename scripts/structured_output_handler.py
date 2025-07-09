"""
Structured Output Handler

Unified handler for structured output across all AI providers.
Converts unified schema format to provider-specific implementations.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union


class StructuredOutputHandler:
    """Handles structured output configuration for different AI providers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prepare_structured_output(self, 
                                provider: str, 
                                prompt: str, 
                                schema: Dict[str, Any],
                                prefill: Optional[str] = None) -> Dict[str, Any]:
        """Prepare provider-specific structured output configuration.
        
        Args:
            provider: The AI provider ('gemini', 'openai', 'deepseek', 'anthropic')
            prompt: The user prompt
            schema: Unified JSON schema format
            prefill: Optional prefill text (mainly for Anthropic)
            
        Returns:
            Dict with provider-specific configuration
        """
        provider = provider.lower()
        
        if provider == 'gemini':
            return self._prepare_gemini_output(prompt, schema)
        elif provider == 'openai':
            return self._prepare_openai_output(prompt, schema)
        elif provider == 'deepseek':
            return self._prepare_deepseek_output(prompt, schema)
        elif provider == 'anthropic':
            return self._prepare_anthropic_output(prompt, schema, prefill)
        else:
            self.logger.warning(f"Unknown provider: {provider}, using default JSON prompt")
            return self._prepare_default_output(prompt, schema)
    
    def _prepare_gemini_output(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Gemini structured output configuration.
        
        Gemini uses response_mime_type and response_schema in generation_config.
        Gemini doesn't support 'additionalProperties' or 'required' fields.
        """
        # Clean schema for Gemini - remove unsupported fields
        def clean_schema_for_gemini(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if k not in ['additionalProperties', 'required']:
                        cleaned[k] = clean_schema_for_gemini(v)
                return cleaned
            elif isinstance(obj, list):
                return [clean_schema_for_gemini(item) for item in obj]
            else:
                return obj
        
        gemini_schema = clean_schema_for_gemini(schema)
        
        return {
            'prompt': prompt,
            'generation_config': {
                'response_mime_type': 'application/json',
                'response_schema': gemini_schema
            }
        }
    
    def _prepare_openai_output(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare OpenAI structured output configuration.
        
        OpenAI uses response_format with json_schema type.
        OpenAI requires 'additionalProperties' and 'required' fields.
        """
        # Deep copy and ensure all required fields for OpenAI
        import copy
        schema_copy = copy.deepcopy(schema)
        
        def add_openai_requirements(obj):
            if isinstance(obj, dict) and 'type' in obj and obj['type'] == 'object':
                if 'additionalProperties' not in obj:
                    obj['additionalProperties'] = False
                if 'properties' in obj and 'required' not in obj:
                    # Add all properties as required
                    obj['required'] = list(obj['properties'].keys())
                # Recursively process nested objects
                if 'properties' in obj:
                    for prop in obj['properties'].values():
                        add_openai_requirements(prop)
                if 'items' in obj:
                    add_openai_requirements(obj['items'])
            elif isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, dict):
                        add_openai_requirements(v)
        
        add_openai_requirements(schema_copy)
        
        return {
            'prompt': prompt,
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'structured_output',
                    'strict': True,
                    'schema': schema_copy
                }
            }
        }
    
    def _prepare_deepseek_output(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare DeepSeek structured output configuration.
        
        DeepSeek uses response_format with json_object type and requires
        'json' to be mentioned in the prompt with an example.
        """
        # Generate example JSON from schema
        example_json = self._generate_example_from_schema(schema)
        
        # Enhance prompt with JSON instruction and example
        enhanced_prompt = f"""{prompt}

Please respond with a JSON object following this structure:
{json.dumps(example_json, indent=2)}

Remember to output valid JSON only."""
        
        return {
            'prompt': enhanced_prompt,
            'response_format': {'type': 'json_object'}
        }
    
    def _prepare_anthropic_output(self, prompt: str, schema: Dict[str, Any], 
                                  prefill: Optional[str] = None) -> Dict[str, Any]:
        """Prepare Anthropic structured output configuration.
        
        Anthropic doesn't have native structured output, so we use:
        1. Clear schema definition in the prompt
        2. Example JSON
        3. Prefilling the assistant response
        """
        # Generate example JSON from schema
        example_json = self._generate_example_from_schema(schema)
        
        # Create detailed prompt with schema
        enhanced_prompt = f"""{prompt}

You must respond with a JSON object that follows this exact schema:
{json.dumps(schema, indent=2)}

Example of the expected format:
{json.dumps(example_json, indent=2)}

Important: Return ONLY the JSON object, no additional text or markdown formatting."""
        
        # Use prefill to start the JSON response
        if not prefill:
            prefill = "{"
        
        return {
            'prompt': enhanced_prompt,
            'prefill': prefill
        }
    
    def _prepare_default_output(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare default output for unknown providers."""
        example_json = self._generate_example_from_schema(schema)
        
        enhanced_prompt = f"""{prompt}

Please respond with a JSON object following this structure:
{json.dumps(example_json, indent=2)}

Return only valid JSON, no additional text."""
        
        return {'prompt': enhanced_prompt}
    
    def _generate_example_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an example JSON object from a schema.
        
        This helps providers that don't have native schema support.
        """
        if schema.get('type') == 'object':
            example = {}
            properties = schema.get('properties', {})
            
            for key, prop in properties.items():
                example[key] = self._get_example_value(prop)
            
            return example
        elif schema.get('type') == 'array':
            item_schema = schema.get('items', {'type': 'string'})
            return [self._get_example_value(item_schema)]
        else:
            return self._get_example_value(schema)
    
    def _get_example_value(self, prop: Dict[str, Any]) -> Any:
        """Get an example value for a property based on its type."""
        prop_type = prop.get('type', 'string')
        
        if prop_type == 'string':
            if 'enum' in prop:
                return prop['enum'][0] if prop['enum'] else 'example'
            return prop.get('description', 'example string')
        elif prop_type == 'number' or prop_type == 'integer':
            return 42
        elif prop_type == 'boolean':
            return True
        elif prop_type == 'array':
            item_schema = prop.get('items', {'type': 'string'})
            return [self._get_example_value(item_schema)]
        elif prop_type == 'object':
            return self._generate_example_from_schema(prop)
        else:
            return None
    
    def validate_schema(self, schema: Dict[str, Any]) -> bool:
        """Validate that a schema is well-formed.
        
        Args:
            schema: The JSON schema to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation
            if not isinstance(schema, dict):
                return False
            
            # Check for required type field
            if 'type' not in schema:
                self.logger.error("Schema missing 'type' field")
                return False
            
            # Validate object schemas
            if schema['type'] == 'object':
                if 'properties' not in schema:
                    self.logger.error("Object schema missing 'properties'")
                    return False
                
                # Check that properties is a dict
                if not isinstance(schema['properties'], dict):
                    self.logger.error("Schema 'properties' must be a dictionary")
                    return False
            
            # Validate array schemas
            elif schema['type'] == 'array':
                if 'items' not in schema:
                    self.logger.error("Array schema missing 'items'")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            return False
    
    def simplify_schema_for_provider(self, schema: Dict[str, Any], provider: str) -> Dict[str, Any]:
        """Simplify schema based on provider limitations.
        
        Some providers have limitations on schema complexity.
        
        Args:
            schema: The original schema
            provider: The target provider
            
        Returns:
            Simplified schema
        """
        provider = provider.lower()
        
        # Gemini sometimes has issues with deeply nested schemas
        if provider == 'gemini':
            return self._simplify_nested_schema(schema, max_depth=3)
        
        # DeepSeek works better with simpler schemas
        elif provider == 'deepseek':
            return self._simplify_nested_schema(schema, max_depth=2)
        
        # OpenAI and Anthropic can handle complex schemas
        return schema
    
    def _simplify_nested_schema(self, schema: Dict[str, Any], max_depth: int, 
                               current_depth: int = 0) -> Dict[str, Any]:
        """Simplify nested schemas by flattening beyond a certain depth."""
        if current_depth >= max_depth:
            # Convert to generic object or string
            return {'type': 'object' if schema.get('type') == 'object' else 'string'}
        
        simplified = schema.copy()
        
        if schema.get('type') == 'object' and 'properties' in schema:
            simplified['properties'] = {}
            for key, prop in schema['properties'].items():
                simplified['properties'][key] = self._simplify_nested_schema(
                    prop, max_depth, current_depth + 1
                )
        
        elif schema.get('type') == 'array' and 'items' in schema:
            simplified['items'] = self._simplify_nested_schema(
                schema['items'], max_depth, current_depth + 1
            )
        
        return simplified


# Example usage
if __name__ == "__main__":
    handler = StructuredOutputHandler()
    
    # Example schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age"]
    }
    
    # Test different providers
    for provider in ['gemini', 'openai', 'deepseek', 'anthropic']:
        print(f"\n{provider.upper()} Configuration:")
        config = handler.prepare_structured_output(
            provider, 
            "Extract person information",
            schema
        )
        print(json.dumps(config, indent=2))