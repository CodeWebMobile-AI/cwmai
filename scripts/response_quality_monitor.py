"""
Response Quality Monitor
========================

Monitors AI assistant responses and triggers improvements when quality issues are detected.
This ensures the system self-improves its communication quality over time.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import re

from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient


class ResponseQualityMonitor:
    """Monitors and improves AI response quality automatically."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.state_manager = StateManager()
        self.ai_client = HTTPAIClient()
        
        # Quality metrics
        self.quality_thresholds = {
            "min_response_length": 50,
            "max_raw_data_ratio": 0.7,  # Max 70% raw data
            "min_explanation_words": 10,
            "required_elements": ["context", "explanation", "natural_language"]
        }
        
        # Track recent responses
        self.response_history = []
        self.quality_scores = []
        
        # Pattern detection
        self.quality_patterns = {
            "good": [
                r"Here's what I found",
                r"Let me explain",
                r"Based on .+ data",
                r"The .+ shows",
                r"This means",
            ],
            "poor": [
                r"^Success: True",
                r"^{.*}$",
                r"^\[.*\]$",
                r"^Error:",
                r"^None$",
            ]
        }
        
    async def monitor_response(self, user_input: str, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor a response and return quality metrics."""
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response_length": len(response),
            "quality_score": 0.0,
            "issues": [],
            "improvements_needed": []
        }
        
        # Calculate quality score
        score = self._calculate_quality_score(response, user_input)
        quality_report["quality_score"] = score
        
        # Identify specific issues
        if score < 0.7:  # Below 70% is considered poor
            issues = self._identify_issues(response, user_input, context)
            quality_report["issues"] = issues
            quality_report["improvements_needed"] = self._suggest_improvements(issues)
            
            # Log for learning
            self._log_quality_issue(quality_report)
            
        # Update history
        self.response_history.append(quality_report)
        self.quality_scores.append(score)
        
        # Maintain rolling window of 100 responses
        if len(self.response_history) > 100:
            self.response_history.pop(0)
            self.quality_scores.pop(0)
            
        return quality_report
    
    def _calculate_quality_score(self, response: str, user_input: str) -> float:
        """Calculate quality score from 0 to 1."""
        score = 0.0
        
        # Length check (20%)
        if len(response) >= self.quality_thresholds["min_response_length"]:
            score += 0.2
        else:
            score += 0.2 * (len(response) / self.quality_thresholds["min_response_length"])
            
        # Natural language check (30%)
        word_count = len(response.split())
        if word_count >= self.quality_thresholds["min_explanation_words"]:
            score += 0.3
        else:
            score += 0.3 * (word_count / self.quality_thresholds["min_explanation_words"])
            
        # Not raw data dump (20%)
        if not any(re.match(pattern, response.strip()) for pattern in self.quality_patterns["poor"]):
            score += 0.2
            
        # Contains good patterns (20%)
        good_pattern_count = sum(1 for pattern in self.quality_patterns["good"] 
                                if re.search(pattern, response, re.IGNORECASE))
        score += min(0.2, good_pattern_count * 0.05)
        
        # Relevance to question (10%)
        if self._check_relevance(user_input, response):
            score += 0.1
            
        return min(score, 1.0)
    
    def _check_relevance(self, user_input: str, response: str) -> bool:
        """Check if response is relevant to the question."""
        # Extract key terms from user input
        key_terms = []
        
        # Look for project names
        project_pattern = r"([a-zA-Z0-9-]+)"
        matches = re.findall(project_pattern, user_input.lower())
        key_terms.extend(matches)
        
        # Check if any key terms appear in response
        response_lower = response.lower()
        return any(term in response_lower for term in key_terms if len(term) > 3)
    
    def _identify_issues(self, response: str, user_input: str, context: Dict[str, Any]) -> List[str]:
        """Identify specific quality issues."""
        issues = []
        
        # Too short
        if len(response) < self.quality_thresholds["min_response_length"]:
            issues.append("response_too_short")
            
        # Raw data dump
        if response.strip().startswith(("{", "[", "Success:")):
            issues.append("raw_data_dump")
            
        # No explanation
        if not any(word in response.lower() for word in ["is", "are", "the", "this", "it"]):
            issues.append("no_explanation")
            
        # Missing context usage
        if "tell me about" in user_input.lower() and len(response) < 100:
            issues.append("missing_context_usage")
            
        # Not conversational
        if not any(char in response for char in [" ", ".", ","]) or len(response.split()) < 5:
            issues.append("not_conversational")
            
        return issues
    
    def _suggest_improvements(self, issues: List[str]) -> List[str]:
        """Suggest specific improvements based on issues."""
        improvements = []
        
        improvement_map = {
            "response_too_short": "Expand response with more details and context",
            "raw_data_dump": "Format data into natural language explanation",
            "no_explanation": "Add explanation of what the data means",
            "missing_context_usage": "Use available context data instead of calling tools",
            "not_conversational": "Rewrite in a friendly, conversational tone"
        }
        
        for issue in issues:
            if issue in improvement_map:
                improvements.append(improvement_map[issue])
                
        return improvements
    
    def _log_quality_issue(self, report: Dict[str, Any]):
        """Log quality issues for learning."""
        log_file = Path("logs/response_quality_issues.json")
        log_file.parent.mkdir(exist_ok=True)
        
        issues = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    issues = json.load(f)
            except:
                pass
                
        issues.append(report)
        
        # Keep last 500 issues
        issues = issues[-500:]
        
        with open(log_file, 'w') as f:
            json.dump(issues, f, indent=2)
    
    async def generate_quality_report(self) -> Dict[str, Any]:
        """Generate a quality report for recent responses."""
        if not self.quality_scores:
            return {"status": "no_data"}
            
        avg_score = sum(self.quality_scores) / len(self.quality_scores)
        
        # Count issues
        issue_counts = {}
        for report in self.response_history:
            for issue in report.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
        return {
            "average_quality_score": round(avg_score, 2),
            "total_responses": len(self.quality_scores),
            "poor_responses": sum(1 for s in self.quality_scores if s < 0.7),
            "excellent_responses": sum(1 for s in self.quality_scores if s > 0.9),
            "common_issues": dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate quality trend."""
        if len(self.quality_scores) < 10:
            return "insufficient_data"
            
        # Compare first half to second half
        mid = len(self.quality_scores) // 2
        first_half_avg = sum(self.quality_scores[:mid]) / mid
        second_half_avg = sum(self.quality_scores[mid:]) / (len(self.quality_scores) - mid)
        
        if second_half_avg > first_half_avg + 0.1:
            return "improving"
        elif second_half_avg < first_half_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    async def trigger_system_improvement(self):
        """Trigger system-wide improvements based on quality issues."""
        report = await self.generate_quality_report()
        
        if report.get("average_quality_score", 1.0) < 0.7:
            self.logger.warning(f"Low average response quality: {report['average_quality_score']}")
            
            # Identify most common issues
            common_issues = report.get("common_issues", {})
            
            # Generate improvement prompt
            improvement_actions = []
            
            if "raw_data_dump" in common_issues:
                improvement_actions.append("Always format tool results into natural language")
                
            if "missing_context_usage" in common_issues:
                improvement_actions.append("Check context data before calling tools")
                
            if "not_conversational" in common_issues:
                improvement_actions.append("Use friendly, conversational tone in all responses")
                
            # Save improvement instructions
            self._save_improvement_instructions(improvement_actions)
            
            return {
                "improvements_triggered": True,
                "actions": improvement_actions
            }
            
        return {"improvements_triggered": False}
    
    def _save_improvement_instructions(self, actions: List[str]):
        """Save improvement instructions for the AI system."""
        instructions_file = Path("ai_improvement_instructions.json")
        
        instructions = {
            "timestamp": datetime.now().isoformat(),
            "quality_improvements": actions,
            "apply_to": "conversational_ai_assistant"
        }
        
        with open(instructions_file, 'w') as f:
            json.dump(instructions, f, indent=2)