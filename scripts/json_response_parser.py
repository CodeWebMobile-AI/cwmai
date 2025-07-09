"""
Unified JSON Response Parser

Provides robust JSON parsing for AI responses with proper error handling
and sanitization of invalid characters.

NOTE: This parser now prioritizes structured output from AI providers.
When using AI responses, prefer passing a response_schema to get
pre-parsed content via structured output APIs rather than relying on
text parsing.
"""

import json
import re
import logging
from typing import Dict, Any, Optional


class JSONResponseParser:
    """Unified parser for AI JSON responses."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize parser.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def parse(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse JSON from AI response with robust error handling.
        
        Args:
            response: AI response dict with 'content' field
            
        Returns:
            Parsed JSON dict or None
        """
        # Check if we already have parsed content from structured output
        if response.get('structured_output') and 'parsed_content' in response:
            self.logger.debug("Using pre-parsed content from structured output")
            return response['parsed_content']
        
        content = response.get('content', '')
        
        if not content:
            return None
        
        # Try multiple parsing strategies in order
        strategies = [
            self._parse_clean_json,
            self._parse_from_code_block,
            self._parse_json_object,
            self._parse_with_sanitization,
            self._parse_key_value_fallback
        ]
        
        for strategy in strategies:
            try:
                result = strategy(content)
                if result:
                    return result
            except Exception as e:
                self.logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        self.logger.warning("All JSON parsing strategies failed")
        return None
    
    def _parse_clean_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Try to parse content as clean JSON.
        
        Args:
            content: Raw content string
            
        Returns:
            Parsed JSON or None
        """
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return None
    
    def _parse_from_code_block(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from markdown code blocks.
        
        Args:
            content: Content with potential code blocks
            
        Returns:
            Parsed JSON or None
        """
        # Try to find JSON in code blocks
        code_block_patterns = [
            r'```(?:json)?\s*\n([\s\S]*?)\n```',
            r'```(?:json)?\s*([\s\S]*?)```',
            r'`([\s\S]*?)`'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    # Clean the match
                    cleaned = self._sanitize_json_string(match.strip())
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_json_object(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract and parse the first complete JSON object.
        
        Args:
            content: Content containing JSON
            
        Returns:
            Parsed JSON or None
        """
        # Find JSON object boundaries
        start = content.find('{')
        if start == -1:
            return None
        
        # Extract complete JSON object
        json_str = self._extract_balanced_json(content[start:])
        if not json_str:
            return None
        
        try:
            # Sanitize before parsing
            cleaned = self._sanitize_json_string(json_str)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
    
    def _parse_with_sanitization(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON after aggressive sanitization.
        
        Args:
            content: Raw content
            
        Returns:
            Parsed JSON or None
        """
        # Find anything that looks like JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                # Aggressive sanitization
                cleaned = self._sanitize_json_string(match)
                # Fix common issues
                cleaned = self._fix_json_issues(cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _parse_key_value_fallback(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract key-value pairs as last resort.
        
        Args:
            content: Content to parse
            
        Returns:
            Extracted data or None
        """
        result = {}
        
        # Common patterns for key-value extraction
        patterns = [
            (r'"(\w+)"\s*:\s*"([^"]*)"', 'string'),
            (r'"(\w+)"\s*:\s*(\d+)', 'number'),
            (r'"(\w+)"\s*:\s*(true|false)', 'boolean'),
            (r'"(\w+)"\s*:\s*\[([^\]]*)\]', 'array'),
        ]
        
        for pattern, value_type in patterns:
            matches = re.findall(pattern, content)
            for key, value in matches:
                if value_type == 'string':
                    result[key] = value
                elif value_type == 'number':
                    result[key] = int(value) if value.isdigit() else float(value)
                elif value_type == 'boolean':
                    result[key] = value.lower() == 'true'
                elif value_type == 'array':
                    # Simple array parsing
                    try:
                        result[key] = json.loads(f'[{value}]')
                    except:
                        result[key] = [v.strip().strip('"') for v in value.split(',')]
        
        return result if result else None
    
    def _sanitize_json_string(self, json_str: str) -> str:
        """Sanitize JSON string by removing invalid characters.
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Sanitized JSON string
        """
        # Remove control characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        # Fix escaped quotes that might be double-escaped
        json_str = json_str.replace('\\"', '"')
        json_str = json_str.replace('\\\\', '\\')
        
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Normalize newlines in strings
        json_str = json_str.replace('\n', '\\n')
        json_str = json_str.replace('\r', '\\r')
        json_str = json_str.replace('\t', '\\t')
        
        return json_str
    
    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues.
        
        Args:
            json_str: JSON string with potential issues
            
        Returns:
            Fixed JSON string
        """
        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix single quotes
        # Only replace single quotes that are actually string delimiters
        # This is tricky, so we'll be conservative
        if "'" in json_str and '"' not in json_str:
            json_str = json_str.replace("'", '"')
        
        # Ensure boolean values are lowercase
        json_str = json_str.replace('True', 'true')
        json_str = json_str.replace('False', 'false')
        json_str = json_str.replace('None', 'null')
        
        return json_str
    
    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract a balanced JSON object/array from text.
        
        Args:
            text: Text starting with { or [
            
        Returns:
            Complete JSON string or None
        """
        if not text or text[0] not in '{[':
            return None
        
        stack = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char in '{[':
                    stack.append(char)
                elif char in '}]':
                    if not stack:
                        return None
                    
                    expected = '{' if char == '}' else '['
                    if stack[-1] != expected:
                        return None
                    
                    stack.pop()
                    if not stack:
                        return text[:i+1]
        
        return None


# Create a singleton instance for easy import
default_parser = JSONResponseParser()
parse_json_response = default_parser.parse