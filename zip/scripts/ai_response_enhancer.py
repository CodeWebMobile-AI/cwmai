"""
AI Response Enhancer
====================

This module monitors AI responses and automatically improves them
when they don't meet quality standards.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager


class AIResponseEnhancer:
    """Enhances AI responses to ensure quality and usefulness."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
        self.state_manager = StateManager()
        
        # Response quality patterns
        self.poor_response_patterns = [
            r"Success:\s*True\s*Summary:\s*{.*}",  # Raw dict output
            r"^{.*}$",  # Just a JSON object
            r"^\[\]$",  # Empty list
            r"^None$",  # None response
            r"^Error:.*$",  # Just an error
            r"^\d+$",  # Just a number
        ]
        
        # Context awareness patterns
        self.context_questions = [
            r"tell me (?:more )?about (.+)",
            r"what is (.+)",
            r"describe (.+)",
            r"show me (.+)",
            r"list all (.+) with (.+)",
            r"what's the status of (.+)",
            r"how is (.+) doing",
        ]
        
    async def enhance_response(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> str:
        """Enhance an AI response if it's poor quality."""
        
        # Check if response needs enhancement
        if not self._needs_enhancement(ai_response):
            return ai_response
            
        print(f"Enhancing poor quality response for: {user_input}")
        
        # Determine what the user actually wanted
        intent = self._analyze_user_intent(user_input)
        
        # Check if we have the data in context
        context_data = self._extract_relevant_context(user_input, context)
        
        if context_data:
            # We have the data - format it nicely
            enhanced = await self._format_context_data(user_input, context_data, intent)
        else:
            # We need to interpret the raw response better
            enhanced = await self._enhance_raw_response(user_input, ai_response, intent)
            
        return enhanced
    
    def _needs_enhancement(self, response: str) -> bool:
        """Check if a response needs enhancement."""
        # Check against poor response patterns
        for pattern in self.poor_response_patterns:
            if re.match(pattern, response.strip(), re.IGNORECASE):
                return True
                
        # Check if it's too short for a meaningful response
        if len(response.strip()) < 50 and not response.strip().startswith("I "):
            return True
            
        # Check if it's just raw data without explanation
        if response.strip().startswith(("{", "[", "Success:")) and "I " not in response:
            return True
            
        return False
    
    def _analyze_user_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze what the user actually wants."""
        intent = {
            "type": "unknown",
            "target": None,
            "details_requested": []
        }
        
        # Check for repository/project questions
        repo_patterns = [
            r"(?:tell me (?:more )?about|describe|what is) ([a-zA-Z0-9-]+)",
            r"([\w-]+) (?:project|repo|repository)",
            r"status of ([a-zA-Z0-9-]+)",
        ]
        
        for pattern in repo_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                intent["type"] = "repository_info"
                intent["target"] = match.group(1)
                intent["details_requested"] = ["health_score", "description", "metrics", "activity"]
                break
                
        # Check for list requests
        if "list" in user_input.lower() and "health" in user_input.lower():
            intent["type"] = "list_with_scores"
            intent["details_requested"] = ["name", "health_score", "status"]
            
        return intent
    
    def _extract_relevant_context(self, user_input: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract relevant data from context."""
        # Look for project mentions
        project_pattern = r"([a-zA-Z0-9-]+)"
        matches = re.findall(project_pattern, user_input.lower())
        
        # Check if any match is a project in context
        if "projects" in context:
            for match in matches:
                if match in context["projects"]:
                    return context["projects"][match]
                    
        # Check formatted context
        if "formatted_context" in context:
            # Search in formatted context string
            for match in matches:
                if match in str(context["formatted_context"]):
                    # Try to extract the relevant section
                    return self._extract_project_from_formatted(match, context["formatted_context"])
                    
        return None
    
    def _extract_project_from_formatted(self, project_name: str, formatted_context: str) -> Optional[Dict[str, Any]]:
        """Extract project data from formatted context string."""
        # This is a simplified extraction - in reality would be more sophisticated
        lines = formatted_context.split('\n')
        capturing = False
        project_data = {}
        
        for line in lines:
            if project_name in line:
                capturing = True
            elif capturing and line.strip() == "":
                break
            elif capturing:
                # Try to parse key-value pairs
                if ":" in line:
                    key, value = line.split(":", 1)
                    project_data[key.strip()] = value.strip()
                    
        return project_data if project_data else None
    
    async def _format_context_data(self, user_input: str, data: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Format context data into a natural response."""
        prompt = f"""
        The user asked: "{user_input}"
        
        I have this data about {intent.get('target', 'the requested item')}:
        {json.dumps(data, indent=2)}
        
        Please create a natural, conversational response that:
        1. Directly answers their question
        2. Highlights the most important information first
        3. Uses a friendly, informative tone
        4. Formats numbers and data nicely
        5. Includes relevant emojis where appropriate
        
        Do not mention that you're formatting data or that this came from context.
        Just provide the information naturally.
        """
        
        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model='gemini'
        )
        
        if response.get('success'):
            return response.get('content', data)
        else:
            # Fallback formatting
            return self._fallback_format(data, intent)
    
    async def _enhance_raw_response(self, user_input: str, raw_response: str, intent: Dict[str, Any]) -> str:
        """Enhance a raw response from a tool."""
        prompt = f"""
        The user asked: "{user_input}"
        
        The system returned this raw data: {raw_response}
        
        Please reformat this into a natural, conversational response that:
        1. Explains what the data means in context of their question
        2. Highlights key information
        3. Uses a friendly tone
        4. Formats any numbers or technical data nicely
        5. If the data seems empty or unhelpful, acknowledge this and suggest alternatives
        
        Do not mention "raw data" or "the system returned" - just provide the information naturally.
        """
        
        response = await self.ai_client.generate_enhanced_response(
            prompt=prompt,
            model='gemini'
        )
        
        if response.get('success'):
            return response.get('content', raw_response)
        else:
            return self._fallback_format({"raw_response": raw_response}, intent)
    
    def _fallback_format(self, data: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Fallback formatting when AI enhancement fails."""
        if intent["type"] == "repository_info":
            name = data.get("name", intent.get("target", "Project"))
            health = data.get("health_score", "Unknown")
            desc = data.get("description", "No description available")
            
            return f"""**{name}**

📊 Health Score: {health}
📝 Description: {desc}

For more details, you can analyze this repository further."""
            
        elif intent["type"] == "list_with_scores":
            if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
                response = "Here are the projects with their health scores:\n\n"
                for name, info in data.items():
                    if isinstance(info, dict):
                        score = info.get("health_score", "N/A")
                        response += f"• **{name}**: {score}/100\n"
                return response
            
        # Generic fallback
        return f"Here's what I found:\n{json.dumps(data, indent=2)}"
    
    def save_enhancement_history(self, original: str, enhanced: str, user_input: str):
        """Save enhancement history for learning."""
        history_file = Path("enhancement_history.json")
        
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                
        history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "original_response": original,
            "enhanced_response": enhanced,
            "improvement_score": self._calculate_improvement_score(original, enhanced)
        })
        
        # Keep last 100 enhancements
        history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _calculate_improvement_score(self, original: str, enhanced: str) -> float:
        """Calculate how much better the enhanced response is."""
        score = 0.0
        
        # Length improvement (but not too long)
        if len(enhanced) > len(original) and len(enhanced) < len(original) * 3:
            score += 0.2
            
        # Contains explanation
        if any(word in enhanced.lower() for word in ["is a", "this", "it", "the"]):
            score += 0.2
            
        # Has formatting
        if any(char in enhanced for char in ["•", "**", "📊", "📝", "\n"]):
            score += 0.2
            
        # Not just data dump
        if not enhanced.strip().startswith("{") and not enhanced.strip().startswith("["):
            score += 0.2
            
        # Natural language
        if enhanced.count(" ") > 5:  # Has multiple words
            score += 0.2
            
        return min(score, 1.0)