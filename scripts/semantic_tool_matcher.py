#!/usr/bin/env python3
"""
Semantic Tool Matcher - Intelligent tool discovery using semantic similarity
Finds existing tools that can handle queries without creating duplicates
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from pathlib import Path
import re
import ast

# Avoid circular import - ToolCallingSystem will be passed in
from scripts.ai_brain import AIBrain


# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass


@dataclass
class ToolMatch:
    """Represents a matched tool with similarity score"""
    tool_name: str
    similarity_score: float
    capability_match: Dict[str, float]
    reason: str
    

@dataclass
class ToolCapability:
    """Represents the capabilities of a tool"""
    name: str
    description: str
    actions: List[str]
    inputs: List[str]
    outputs: List[str]
    keywords: List[str]
    examples: List[str]


class SemanticToolMatcher:
    """Matches queries to existing tools using semantic similarity"""
    
    def __init__(
        self,
        tool_system=None,
        cache_file: str = "tool_embeddings.json"
    ):
        self.tool_system = tool_system
        self.ai_brain = AIBrain()
        self.logger = logging.getLogger(__name__)
        self.cache_file = Path(cache_file)
        
        # Initialize NLP components
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Tool capabilities cache
        self.tool_capabilities: Dict[str, ToolCapability] = {}
        self.tool_embeddings: Dict[str, np.ndarray] = {}
        
        # Load or build tool index
        self._build_tool_index()
        
    def _build_tool_index(self):
        """Build semantic index of all available tools"""
        if not self.tool_system:
            return
            
        tools = self.tool_system.list_tools()
        
        # Extract capabilities for each tool
        for tool_name, tool_info in tools.items():
            capability = self._extract_tool_capability(tool_name, tool_info)
            self.tool_capabilities[tool_name] = capability
            
        # Build embeddings
        self._build_embeddings()
        
    def _extract_tool_capability(
        self,
        tool_name: str,
        tool_info: Dict[str, Any]
    ) -> ToolCapability:
        """Extract capabilities from tool metadata and code"""
        description = tool_info.get('description', '')
        
        # Get tool function
        tool = self.tool_system.get_tool(tool_name)
        
        # Extract from docstring
        docstring = tool.__doc__ if tool and tool.__doc__ else ""
        
        # Extract actions from function name and description
        actions = self._extract_actions(tool_name, description, docstring)
        
        # Extract inputs/outputs from parameters
        parameters = tool_info.get('parameters', {})
        inputs = list(parameters.keys()) if isinstance(parameters, dict) else []
        
        # Extract keywords
        keywords = self._extract_keywords(tool_name, description, docstring)
        
        # Generate examples
        examples = self._generate_tool_examples(tool_name, description)
        
        # Infer outputs from description
        outputs = self._infer_outputs(description, docstring)
        
        return ToolCapability(
            name=tool_name,
            description=description,
            actions=actions,
            inputs=inputs,
            outputs=outputs,
            keywords=keywords,
            examples=examples
        )
        
    def _extract_actions(
        self,
        tool_name: str,
        description: str,
        docstring: str
    ) -> List[str]:
        """Extract action verbs from tool information"""
        actions = []
        
        # Common action patterns
        action_patterns = [
            r'\b(create|generate|build|make)\b',
            r'\b(analyze|examine|inspect|check)\b',
            r'\b(find|search|locate|discover)\b',
            r'\b(update|modify|edit|change)\b',
            r'\b(delete|remove|clean|clear)\b',
            r'\b(optimize|improve|enhance|refine)\b',
            r'\b(validate|verify|test|check)\b',
            r'\b(convert|transform|translate)\b',
            r'\b(monitor|track|watch|observe)\b',
            r'\b(execute|run|perform|do)\b'
        ]
        
        combined_text = f"{tool_name} {description} {docstring}".lower()
        
        for pattern in action_patterns:
            matches = re.findall(pattern, combined_text)
            actions.extend(matches)
            
        # Extract from tool name
        name_parts = tool_name.split('_')
        for part in name_parts:
            if part in ['create', 'analyze', 'find', 'update', 'delete', 
                       'generate', 'search', 'build', 'get', 'set']:
                actions.append(part)
                
        return list(set(actions))
        
    def _extract_keywords(
        self,
        tool_name: str,
        description: str,
        docstring: str
    ) -> List[str]:
        """Extract relevant keywords from tool information"""
        # Combine all text
        text = f"{tool_name} {description} {docstring}".lower()
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', text)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        # Get top keywords
        top_keywords = [word for word, _ in word_freq.most_common(20)]
        
        return top_keywords
        
    def _generate_tool_examples(
        self,
        tool_name: str,
        description: str
    ) -> List[str]:
        """Generate example queries that this tool could handle"""
        examples = []
        
        # Generate based on tool name
        readable_name = tool_name.replace('_', ' ')
        examples.append(f"{readable_name}")
        examples.append(f"I need to {readable_name}")
        examples.append(f"Can you {readable_name}?")
        
        # Generate based on description
        if description:
            examples.append(description.lower())
            
            # Extract action and object
            action_match = re.search(r'^(\w+)\s+(.+)', description)
            if action_match:
                action = action_match.group(1)
                obj = action_match.group(2)
                examples.append(f"Please {action} {obj}")
                examples.append(f"{action} the {obj}")
                
        return examples
        
    def _infer_outputs(self, description: str, docstring: str) -> List[str]:
        """Infer what outputs a tool produces"""
        outputs = []
        
        combined_text = f"{description} {docstring}".lower()
        
        # Common output patterns
        output_patterns = [
            (r'return[s]?\s+(\w+)', 1),
            (r'generate[s]?\s+(\w+)', 1),
            (r'create[s]?\s+(\w+)', 1),
            (r'produce[s]?\s+(\w+)', 1),
            (r'output[s]?\s+(\w+)', 1),
            (r'result[s]?\s+in\s+(\w+)', 1),
        ]
        
        for pattern, group in output_patterns:
            matches = re.findall(pattern, combined_text)
            outputs.extend(matches)
            
        return list(set(outputs))
        
    def _build_embeddings(self):
        """Build vector embeddings for all tools"""
        if not self.tool_capabilities:
            return
            
        # Create text representations for each tool
        tool_texts = []
        tool_names = []
        
        for tool_name, capability in self.tool_capabilities.items():
            # Combine all capability information
            text_parts = [
                capability.name,
                capability.description,
                ' '.join(capability.actions),
                ' '.join(capability.keywords),
                ' '.join(capability.inputs),
                ' '.join(capability.outputs),
                ' '.join(capability.examples)
            ]
            
            tool_text = ' '.join(filter(None, text_parts))
            tool_texts.append(tool_text)
            tool_names.append(tool_name)
            
        # Fit vectorizer and transform texts
        if tool_texts:
            self.vectorizer.fit(tool_texts)
            embeddings = self.vectorizer.transform(tool_texts)
            
            # Store embeddings
            for i, tool_name in enumerate(tool_names):
                self.tool_embeddings[tool_name] = embeddings[i]
                
    async def find_similar_tools(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[ToolMatch]:
        """Find tools similar to the given query"""
        if not self.tool_embeddings:
            return []
            
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = []
        
        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = cosine_similarity(query_vector, tool_embedding)[0][0]
            
            if similarity >= threshold:
                # Calculate detailed capability match
                capability_match = await self._calculate_capability_match(
                    query, self.tool_capabilities[tool_name]
                )
                
                # Generate explanation
                reason = self._generate_match_reason(
                    query, self.tool_capabilities[tool_name], similarity
                )
                
                similarities.append(ToolMatch(
                    tool_name=tool_name,
                    similarity_score=similarity,
                    capability_match=capability_match,
                    reason=reason
                ))
                
        # Sort by similarity score
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similarities[:top_k]
        
    async def _calculate_capability_match(
        self,
        query: str,
        capability: ToolCapability
    ) -> Dict[str, float]:
        """Calculate detailed capability matching scores"""
        query_lower = query.lower()
        
        # Action match
        action_score = 0.0
        for action in capability.actions:
            if action in query_lower:
                action_score = 1.0
                break
                
        # Keyword match
        keyword_matches = sum(1 for k in capability.keywords if k in query_lower)
        keyword_score = min(keyword_matches / max(len(capability.keywords), 1), 1.0)
        
        # Input/output relevance
        input_score = 0.5  # Default moderate score
        output_score = 0.5
        
        return {
            "action_match": action_score,
            "keyword_match": keyword_score,
            "input_relevance": input_score,
            "output_relevance": output_score,
            "overall": (action_score + keyword_score + input_score + output_score) / 4
        }
        
    def _generate_match_reason(
        self,
        query: str,
        capability: ToolCapability,
        similarity: float
    ) -> str:
        """Generate human-readable reason for the match"""
        reasons = []
        
        query_lower = query.lower()
        
        # Check action matches
        matched_actions = [a for a in capability.actions if a in query_lower]
        if matched_actions:
            reasons.append(f"can {', '.join(matched_actions)}")
            
        # Check keyword matches
        matched_keywords = [k for k in capability.keywords[:5] if k in query_lower]
        if matched_keywords:
            reasons.append(f"handles {', '.join(matched_keywords)}")
            
        # Add similarity score
        reasons.append(f"{similarity:.0%} similarity")
        
        if reasons:
            return f"{capability.name} {' and '.join(reasons)}"
        else:
            return f"{capability.name} seems relevant ({similarity:.0%} match)"
            
    async def can_existing_tool_handle(
        self,
        query: str,
        confidence_threshold: float = 0.7
    ) -> Optional[str]:
        """Determine if an existing tool can handle the request"""
        matches = await self.find_similar_tools(query, top_k=1, threshold=confidence_threshold)
        
        if matches:
            best_match = matches[0]
            
            # Use AI to confirm the match
            confirmation_prompt = f"""
            Query: {query}
            
            Potential tool: {best_match.tool_name}
            Tool description: {self.tool_capabilities[best_match.tool_name].description}
            Similarity score: {best_match.similarity_score:.2f}
            
            Can this existing tool handle the query? Answer YES or NO.
            If YES, explain briefly how it would handle it.
            If NO, explain what's missing.
            """
            
            response = await self.ai_brain.generate_enhanced_response(confirmation_prompt)
            response_content = response.get('content', '') if isinstance(response, dict) else str(response)
            
            if "YES" in response_content.upper():
                return best_match.tool_name
                
        return None
        
    async def suggest_tool_composition(
        self,
        query: str,
        max_tools: int = 3
    ) -> Optional[List[str]]:
        """Suggest a composition of existing tools to handle a complex query"""
        # Find multiple relevant tools
        matches = await self.find_similar_tools(query, top_k=10, threshold=0.2)
        
        if len(matches) < 2:
            return None
            
        # Use AI to suggest composition
        tool_descriptions = []
        for match in matches[:5]:
            capability = self.tool_capabilities[match.tool_name]
            tool_descriptions.append(
                f"- {match.tool_name}: {capability.description}"
            )
            
        composition_prompt = f"""
        Query: {query}
        
        Available tools:
        {chr(10).join(tool_descriptions)}
        
        Can this query be handled by combining 2-3 of these existing tools?
        If yes, suggest which tools to use and in what order.
        Return as JSON: {{"possible": true/false, "tools": ["tool1", "tool2"], "explanation": "..."}}
        """
        
        response = await self.ai_brain.generate_enhanced_response(composition_prompt)
        response_content = response.get('content', '') if isinstance(response, dict) else str(response)
        
        try:
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("possible") and result.get("tools"):
                    return result["tools"][:max_tools]
        except:
            pass
            
        return None
        
    def add_tool_to_index(self, tool_name: str, tool_info: Dict[str, Any]):
        """Add a new tool to the semantic index"""
        capability = self._extract_tool_capability(tool_name, tool_info)
        self.tool_capabilities[tool_name] = capability
        
        # Rebuild embeddings
        self._build_embeddings()
        
    def update_tool_capability(
        self,
        tool_name: str,
        usage_examples: List[str] = None,
        successful_queries: List[str] = None
    ):
        """Update tool capability based on usage"""
        if tool_name not in self.tool_capabilities:
            return
            
        capability = self.tool_capabilities[tool_name]
        
        # Add usage examples
        if usage_examples:
            capability.examples.extend(usage_examples)
            capability.examples = list(set(capability.examples))  # Remove duplicates
            
        # Extract keywords from successful queries
        if successful_queries:
            for query in successful_queries:
                keywords = self._extract_keywords("", query, "")
                capability.keywords.extend(keywords)
            capability.keywords = list(set(capability.keywords))[:50]  # Limit size
            
        # Rebuild embeddings
        self._build_embeddings()
        
    async def prevent_duplicate_creation(
        self,
        tool_spec: str,
        similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """Check if a tool similar to the spec already exists"""
        existing_tool = await self.can_existing_tool_handle(tool_spec, similarity_threshold)
        
        if existing_tool:
            self.logger.info(
                f"Prevented duplicate tool creation. "
                f"Existing tool '{existing_tool}' can handle: {tool_spec}"
            )
            return existing_tool
            
        # Check for composition possibility
        composition = await self.suggest_tool_composition(tool_spec)
        if composition:
            self.logger.info(
                f"Prevented duplicate tool creation. "
                f"Can use composition: {' -> '.join(composition)}"
            )
            return f"composition:{','.join(composition)}"
            
        return None


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def demo():
        matcher = SemanticToolMatcher()
        
        # Test finding similar tools
        query = "analyze repository for security vulnerabilities"
        matches = await matcher.find_similar_tools(query)
        
        print(f"Query: {query}")
        print(f"Found {len(matches)} matching tools:\n")
        
        for match in matches:
            print(f"Tool: {match.tool_name}")
            print(f"Similarity: {match.similarity_score:.2%}")
            print(f"Reason: {match.reason}")
            print(f"Capability match: {match.capability_match}")
            print()
            
        # Test duplicate prevention
        new_tool_spec = "create a tool to scan code for security issues"
        existing = await matcher.prevent_duplicate_creation(new_tool_spec)
        
        if existing:
            print(f"\nDuplicate prevention: {existing}")
        else:
            print("\nNo existing tool found, safe to create new one")
            
    asyncio.run(demo())