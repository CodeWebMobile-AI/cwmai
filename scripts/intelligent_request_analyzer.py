"""
Intelligent Request Analyzer
============================

Uses AI to understand user requests and extract meaningful information
instead of relying on simple regex patterns.
"""

from typing import Dict, Any, Optional, List, Tuple
import re


class IntelligentRequestAnalyzer:
    """Analyzes user requests using AI to extract intent and key concepts."""
    
    def __init__(self, ai_brain):
        self.ai_brain = ai_brain
        
    async def analyze_tool_creation_request(self, request: str) -> Dict[str, Any]:
        """Use AI to understand what tool the user wants to create."""
        
        prompt = f"""Analyze this tool creation request and extract key information:

Request: "{request}"

Extract:
1. Tool name (snake_case, descriptive)
2. Primary search keyword (most relevant technical term)
3. Tool description (one line)
4. Key concepts (list of relevant terms)

Respond in this exact format:
TOOL_NAME: <name>
SEARCH_KEYWORD: <keyword>
DESCRIPTION: <description>
CONCEPTS: <concept1>, <concept2>, <concept3>

Example for "Create a tool to check if repositories have README files":
TOOL_NAME: check_readme_files
SEARCH_KEYWORD: readme
DESCRIPTION: Check if repositories have README files
CONCEPTS: readme, repository, file, check

Now analyze the given request."""

        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'max_length': 200,
            'model': 'gemini'
        })
        
        result_text = response.get('result', '')
        
        # Parse the structured response
        analysis = {
            'tool_name': self._extract_field(result_text, 'TOOL_NAME'),
            'search_keyword': self._extract_field(result_text, 'SEARCH_KEYWORD'),
            'description': self._extract_field(result_text, 'DESCRIPTION'),
            'concepts': self._extract_list_field(result_text, 'CONCEPTS')
        }
        
        # Fallback if AI doesn't follow format
        if not analysis['tool_name']:
            analysis['tool_name'] = self._generate_fallback_tool_name(request)
        if not analysis['search_keyword']:
            analysis['search_keyword'] = self._extract_best_keyword(request)
            
        return analysis
    
    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field value from structured text."""
        pattern = f"{field_name}:\\s*(.+?)(?:\\n|$)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_list_field(self, text: str, field_name: str) -> List[str]:
        """Extract a comma-separated list field from structured text."""
        value = self._extract_field(text, field_name)
        if value:
            return [item.strip() for item in value.split(',')]
        return []
    
    def _generate_fallback_tool_name(self, request: str) -> str:
        """Generate a tool name using simple heuristics as fallback."""
        # Remove common words and extract key terms
        words = request.lower().split()
        skip_words = {'a', 'an', 'the', 'to', 'for', 'that', 'if', 'all', 'my', 'build', 'create', 'make', 'tool'}
        
        key_words = []
        for word in words:
            cleaned = re.sub(r'[^\w]', '', word)  # Remove punctuation
            if cleaned and cleaned not in skip_words and len(cleaned) > 2:
                key_words.append(cleaned)
        
        # Common patterns
        if 'check' in key_words and len(key_words) > 1:
            key_words.remove('check')
            return f"check_{'_'.join(key_words[:2])}"
        
        return '_'.join(key_words[:3]) if key_words else 'custom_tool'
    
    def _extract_best_keyword(self, request: str) -> str:
        """Extract the most relevant technical keyword."""
        # Look for technical terms (containing dots, underscores, etc.)
        technical_pattern = r'[\w]*[._-][\w]+|\.[\w]+'
        technical_matches = re.findall(technical_pattern, request)
        if technical_matches:
            # Prefer longer technical terms
            return max(technical_matches, key=len)
        
        # Look for quoted terms
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_matches = re.findall(quoted_pattern, request)
        if quoted_matches:
            return quoted_matches[0]
        
        # Extract longest meaningful word
        words = request.lower().split()
        skip_words = {'build', 'create', 'make', 'tool', 'that', 'which', 'should', 'would', 'could', 'have', 'with', 'from', 'check', 'verify'}
        
        meaningful_words = []
        for word in words:
            cleaned = re.sub(r'[^\w.-]', '', word)
            if cleaned and cleaned not in skip_words and len(cleaned) > 3:
                meaningful_words.append(cleaned)
        
        return max(meaningful_words, key=len) if meaningful_words else 'code'