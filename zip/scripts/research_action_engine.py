"""
Research Action Engine - Converts research insights into actionable improvements.

This module extracts actionable insights from research results and generates
specific implementation tasks that can improve CWMAI's performance.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib


class ResearchActionEngine:
    """Convert research insights into actionable improvements."""
    
    def __init__(self, task_generator=None, self_improver=None):
        self.task_generator = task_generator
        self.self_improver = self_improver
        self.implementation_history = []
        
        # Action pattern mappings for different research areas
        self.action_patterns = {
            "claude_interaction": {
                "formatting": {
                    "component": "IntelligentTaskGenerator",
                    "action_type": "UPDATE_TEMPLATE",
                    "priority": "HIGH"
                },
                "prompt_engineering": {
                    "component": "TaskManager", 
                    "action_type": "UPDATE_ISSUE_FORMAT",
                    "priority": "CRITICAL"
                },
                "acceptance_criteria": {
                    "component": "IntelligentTaskGenerator",
                    "action_type": "ADD_CRITERIA_FIELD",
                    "priority": "HIGH"
                }
            },
            "task_generation": {
                "decomposition": {
                    "component": "IntelligentTaskGenerator",
                    "action_type": "ADD_DECOMPOSITION_LOGIC",
                    "priority": "HIGH"
                },
                "complexity_scoring": {
                    "component": "IntelligentTaskGenerator", 
                    "action_type": "IMPLEMENT_COMPLEXITY_SCORING",
                    "priority": "MEDIUM"
                },
                "success_criteria": {
                    "component": "IntelligentTaskGenerator",
                    "action_type": "ENHANCE_CRITERIA_GENERATION",
                    "priority": "HIGH"
                }
            },
            "multi_agent_coordination": {
                "consensus": {
                    "component": "RealSwarmIntelligence",
                    "action_type": "UPDATE_CONSENSUS_ALGORITHM",
                    "priority": "MEDIUM"
                },
                "voting": {
                    "component": "RealSwarmIntelligence",
                    "action_type": "IMPLEMENT_WEIGHTED_VOTING",
                    "priority": "MEDIUM"
                },
                "specialization": {
                    "component": "RealSwarmIntelligence",
                    "action_type": "ADD_AGENT_SPECIALIZATION",
                    "priority": "LOW"
                }
            },
            "outcome_learning": {
                "pattern_recognition": {
                    "component": "OutcomeLearningSystem",
                    "action_type": "ENHANCE_PATTERN_DETECTION",
                    "priority": "MEDIUM"
                },
                "feedback_loops": {
                    "component": "OutcomeLearningSystem",
                    "action_type": "OPTIMIZE_FEEDBACK_LOOPS",
                    "priority": "HIGH"
                },
                "prediction": {
                    "component": "PredictiveTaskEngine",
                    "action_type": "IMPROVE_PREDICTION_MODEL",
                    "priority": "MEDIUM"
                }
            }
        }
        
        # Insight extraction patterns
        self.insight_patterns = {
            "implementation_steps": [
                r"(?:step|steps?)\s*(?:\d+[.)]?)\s*([^.!?]+)",
                r"(?:first|second|third|next|then|finally)[,:]?\s*([^.!?]+)",
                r"to\s+(?:implement|achieve|improve)[^,]+,?\s*([^.!?]+)"
            ],
            "best_practices": [
                r"best\s+practice[s]?[^.!?]*:\s*([^.!?]+)",
                r"(?:should|must|need\s+to)\s+([^.!?]+)",
                r"it\s+is\s+(?:important|crucial|essential)\s+to\s+([^.!?]+)"
            ],
            "problems_solutions": [
                r"(?:problem|issue|challenge)[^.!?]*:\s*([^.!?]+)",
                r"solution[^.!?]*:\s*([^.!?]+)",
                r"to\s+(?:fix|resolve|solve)[^,]+,?\s*([^.!?]+)"
            ],
            "metrics_improvements": [
                r"(?:improve|increase|boost|enhance)\s+([^.!?]+)",
                r"(?:reduce|decrease|minimize)\s+([^.!?]+)",
                r"(?:\d+%|\d+x)\s+(?:improvement|increase|boost)\s+in\s+([^.!?]+)"
            ]
        }
        
    def execute_research(self, research: Dict) -> List[Dict]:
        """
        Execute research by extracting actionable insights.
        This method is an alias for extract_actionable_insights for compatibility.
        
        Args:
            research: Research data containing content and metadata
            
        Returns:
            List of actionable insights
        """
        return self.extract_actionable_insights(research)
    
    def extract_actionable_insights(self, research: Dict) -> List[Dict]:
        """
        Extract actionable insights from research results.
        
        Args:
            research: Research results with content and metadata
            
        Returns:
            List of actionable insights
        """
        insights = []
        
        research_content = research.get("content", {})
        research_area = research.get("area", "general")
        research_topic = research.get("topic", "")
        
        # Extract different types of insights
        implementation_insights = self._extract_implementation_insights(research_content, research_area)
        problem_solution_insights = self._extract_problem_solutions(research_content, research_area)
        best_practice_insights = self._extract_best_practices(research_content, research_area)
        metric_insights = self._extract_metric_improvements(research_content, research_area)
        
        insights.extend(implementation_insights)
        insights.extend(problem_solution_insights)
        insights.extend(best_practice_insights)
        insights.extend(metric_insights)
        
        # Score and rank insights
        scored_insights = self._score_insights(insights, research)
        
        # Filter and deduplicate
        filtered_insights = self._filter_insights(scored_insights)
        
        return filtered_insights[:10]  # Return top 10 insights
    
    def generate_implementation_tasks(self, insights: List[Dict]) -> List[Dict]:
        """
        Generate specific implementation tasks from insights.
        
        Args:
            insights: List of actionable insights
            
        Returns:
            List of implementation tasks
        """
        tasks = []
        
        for insight in insights:
            task = self._create_implementation_task(insight)
            if task:
                tasks.append(task)
        
        # Group related tasks
        grouped_tasks = self._group_related_tasks(tasks)
        
        # Prioritize tasks
        prioritized_tasks = self._prioritize_tasks(grouped_tasks)
        
        return prioritized_tasks
    
    def create_immediate_action_plan(self, research: Dict) -> Dict:
        """
        Create an immediate action plan from critical research.
        
        Args:
            research: Critical research requiring immediate action
            
        Returns:
            Action plan with immediate steps
        """
        insights = self.extract_actionable_insights(research)
        critical_insights = [i for i in insights if i.get("urgency") == "critical"]
        
        if not critical_insights:
            critical_insights = insights[:3]  # Take top 3 if none marked critical
        
        action_plan = {
            "plan_id": f"action_{int(datetime.now().timestamp())}",
            "created_at": datetime.now().isoformat(),
            "research_id": research.get("id"),
            "area": research.get("area"),
            "urgency": "immediate",
            "estimated_impact": self._estimate_plan_impact(critical_insights),
            "implementation_steps": [],
            "rollback_plan": [],
            "success_metrics": []
        }
        
        # Create implementation steps
        for insight in critical_insights:
            step = self._create_implementation_step(insight)
            action_plan["implementation_steps"].append(step)
            
            # Create rollback step
            rollback_step = self._create_rollback_step(step)
            action_plan["rollback_plan"].append(rollback_step)
            
            # Add success metrics
            metrics = self._extract_success_metrics(insight)
            action_plan["success_metrics"].extend(metrics)
        
        return action_plan
    
    def _extract_implementation_insights(self, content: Dict, area: str) -> List[Dict]:
        """Extract implementation-focused insights."""
        insights = []
        
        # Convert content to searchable text
        text = self._content_to_text(content)
        
        # Look for implementation patterns
        for pattern in self.insight_patterns["implementation_steps"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                insight_text = match.group(1).strip()
                
                if len(insight_text) > 10:  # Filter out very short matches
                    insight = {
                        "type": "implementation",
                        "text": insight_text,
                        "area": area,
                        "confidence": 0.7,
                        "action_required": True,
                        "component": self._identify_component(insight_text, area),
                        "implementation_type": self._classify_implementation(insight_text, area)
                    }
                    insights.append(insight)
        
        return insights
    
    def _extract_problem_solutions(self, content: Dict, area: str) -> List[Dict]:
        """Extract problem-solution pairs."""
        insights = []
        text = self._content_to_text(content)
        
        for pattern in self.insight_patterns["problems_solutions"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                insight_text = match.group(1).strip()
                
                if len(insight_text) > 15:
                    insight = {
                        "type": "solution",
                        "text": insight_text,
                        "area": area,
                        "confidence": 0.8,
                        "action_required": True,
                        "urgency": self._assess_urgency(insight_text, area),
                        "component": self._identify_component(insight_text, area)
                    }
                    insights.append(insight)
        
        return insights
    
    def _extract_best_practices(self, content: Dict, area: str) -> List[Dict]:
        """Extract best practice recommendations."""
        insights = []
        text = self._content_to_text(content)
        
        for pattern in self.insight_patterns["best_practices"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                insight_text = match.group(1).strip()
                
                if len(insight_text) > 10:
                    insight = {
                        "type": "best_practice",
                        "text": insight_text,
                        "area": area,
                        "confidence": 0.6,
                        "action_required": self._requires_action(insight_text),
                        "component": self._identify_component(insight_text, area)
                    }
                    insights.append(insight)
        
        return insights
    
    def _extract_metric_improvements(self, content: Dict, area: str) -> List[Dict]:
        """Extract metrics and improvement targets."""
        insights = []
        text = self._content_to_text(content)
        
        for pattern in self.insight_patterns["metrics_improvements"]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                insight_text = match.group(1).strip()
                
                if len(insight_text) > 5:
                    insight = {
                        "type": "metric_improvement",
                        "text": insight_text,
                        "area": area,
                        "confidence": 0.9,
                        "action_required": True,
                        "metric_target": self._extract_metric_target(match.group(0)),
                        "component": self._identify_component(insight_text, area)
                    }
                    insights.append(insight)
        
        return insights
    
    def _create_implementation_task(self, insight: Dict) -> Optional[Dict]:
        """Create a specific implementation task from an insight."""
        if not insight.get("action_required"):
            return None
        
        area = insight.get("area", "general")
        component = insight.get("component", "Unknown")
        insight_text = insight.get("text", "")
        
        # Get action pattern for this area/component
        action_info = self._get_action_info(area, insight_text)
        
        task = {
            "type": "SYSTEM_IMPROVEMENT",
            "title": f"Implement: {insight_text[:50]}...",
            "description": self._create_task_description(insight, action_info),
            "component": component,
            "action_type": action_info.get("action_type", "UPDATE"),
            "priority": action_info.get("priority", "MEDIUM"),
            "estimated_effort": self._estimate_effort(insight),
            "success_criteria": self._create_success_criteria(insight),
            "rollback_plan": self._create_task_rollback_plan(insight, action_info),
            "metadata": {
                "generated_from": "research_insight",
                "insight_type": insight.get("type"),
                "area": area,
                "confidence": insight.get("confidence", 0.5),
                "created_at": datetime.now().isoformat()
            }
        }
        
        return task
    
    def _create_task_description(self, insight: Dict, action_info: Dict) -> str:
        """Create detailed task description."""
        insight_text = insight.get("text", "")
        area = insight.get("area", "")
        component = insight.get("component", "")
        
        description = f"""
## Research-Based System Improvement

**Insight:** {insight_text}

**Component:** {component}
**Area:** {area}
**Action Type:** {action_info.get('action_type', 'UPDATE')}

## Implementation Plan

{self._generate_implementation_steps(insight, action_info)}

## Expected Outcome

{self._generate_expected_outcome(insight)}

## Acceptance Criteria

{self._generate_acceptance_criteria(insight)}
"""
        
        return description.strip()
    
    def _generate_expected_outcome(self, insight: Dict) -> str:
        """Generate expected outcome description."""
        area = insight.get("area", "")
        text = insight.get("text", "")
        
        if area == "claude_interaction":
            return "Improved Claude response rate and task acceptance"
        elif area == "task_generation":
            return "Higher task completion rate and better task quality"
        elif area == "multi_agent_coordination":
            return "Better agent consensus and faster decision making"
        else:
            return f"Implementation of: {text[:50]}..."
    
    def _generate_acceptance_criteria(self, insight: Dict) -> str:
        """Generate acceptance criteria."""
        criteria = []
        area = insight.get("area", "")
        
        if area == "claude_interaction":
            criteria = [
                "- Claude response rate improves by at least 20%",
                "- Task acceptance rate increases",
                "- No regression in existing functionality"
            ]
        elif area == "task_generation":
            criteria = [
                "- Task completion rate improves by at least 15%", 
                "- Task quality scores increase",
                "- Implementation is backwards compatible"
            ]
        else:
            criteria = [
                "- Feature is implemented correctly",
                "- All tests pass",
                "- Performance is not degraded"
            ]
        
        return "\n".join(criteria)
    
    def _generate_implementation_steps(self, insight: Dict, action_info: Dict) -> str:
        """Generate specific implementation steps."""
        area = insight.get("area")
        action_type = action_info.get("action_type", "UPDATE")
        
        if area == "claude_interaction" and "template" in action_type.lower():
            return """
1. Analyze current task generation templates
2. Identify formatting issues preventing Claude responses
3. Update templates with improved structure and clarity
4. Add mandatory acceptance criteria fields
5. Test with sample tasks
6. Deploy updated templates
7. Monitor Claude response rate improvement
"""
        
        elif area == "task_generation" and "decomposition" in action_type.lower():
            return """
1. Review current task complexity assessment
2. Implement task decomposition algorithm
3. Add complexity scoring system
4. Create subtask generation logic
5. Test with complex tasks
6. Integrate with task generator
7. Monitor task completion rate improvement
"""
        
        else:
            return """
1. Analyze current implementation
2. Design improvement based on research insight
3. Implement changes with safety checks
4. Test thoroughly in development environment
5. Deploy with monitoring
6. Measure impact on relevant metrics
"""
    
    def _score_insights(self, insights: List[Dict], research: Dict) -> List[Dict]:
        """Score insights by potential impact and feasibility."""
        for insight in insights:
            score = 0.0
            
            # Base confidence score
            score += insight.get("confidence", 0.5) * 0.3
            
            # Area importance
            area = insight.get("area", "")
            area_weights = {
                "claude_interaction": 0.9,  # Critical for execution
                "task_generation": 0.8,     # Core functionality
                "outcome_learning": 0.7,    # Important for improvement
                "multi_agent_coordination": 0.6  # Quality enhancement
            }
            score += area_weights.get(area, 0.5) * 0.3
            
            # Urgency factor
            urgency = insight.get("urgency", "medium")
            urgency_weights = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
            score += urgency_weights.get(urgency, 0.6) * 0.2
            
            # Implementation feasibility
            if insight.get("component") != "Unknown":
                score += 0.1  # Known component = easier to implement
            
            if insight.get("action_required"):
                score += 0.1  # Clear action = higher value
            
            insight["impact_score"] = min(1.0, score)
        
        # Sort by score
        insights.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
        return insights
    
    def _filter_insights(self, insights: List[Dict]) -> List[Dict]:
        """Filter out low-quality or redundant insights."""
        filtered = []
        seen_texts = set()
        
        for insight in insights:
            text = insight.get("text", "").lower().strip()
            
            # Skip very short insights
            if len(text) < 10:
                continue
            
            # Skip duplicates
            if text in seen_texts:
                continue
            
            # Skip very low confidence
            if insight.get("confidence", 0) < 0.3:
                continue
            
            seen_texts.add(text)
            filtered.append(insight)
        
        return filtered
    
    def _group_related_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Group related tasks together."""
        # Simple grouping by component and area
        groups = {}
        
        for task in tasks:
            component = task.get("component", "Unknown")
            area = task.get("metadata", {}).get("area", "general")
            group_key = f"{component}_{area}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(task)
        
        # Create combined tasks for groups with multiple items
        combined_tasks = []
        
        for group_key, group_tasks in groups.items():
            if len(group_tasks) == 1:
                combined_tasks.append(group_tasks[0])
            else:
                # Create a combined task
                combined_task = self._combine_tasks(group_tasks)
                combined_tasks.append(combined_task)
        
        return combined_tasks
    
    def _prioritize_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Prioritize tasks by impact and urgency."""
        priority_scores = {
            "CRITICAL": 4,
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1
        }
        
        # Sort by priority and impact score
        tasks.sort(key=lambda t: (
            priority_scores.get(t.get("priority", "MEDIUM"), 2),
            t.get("metadata", {}).get("confidence", 0.5)
        ), reverse=True)
        
        return tasks
    
    def _content_to_text(self, content: Dict) -> str:
        """Convert research content to searchable text."""
        if isinstance(content, str):
            return content
        
        text_parts = []
        
        def extract_text(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item)
        
        extract_text(content)
        return " ".join(text_parts)
    
    def _identify_component(self, text: str, area: str) -> str:
        """Identify which component should be modified."""
        text_lower = text.lower()
        
        # Area-based component mapping
        area_components = {
            "claude_interaction": {
                "task": "IntelligentTaskGenerator",
                "issue": "TaskManager", 
                "format": "TaskManager",
                "prompt": "TaskManager"
            },
            "task_generation": {
                "task": "IntelligentTaskGenerator",
                "decomposition": "IntelligentTaskGenerator",
                "complexity": "IntelligentTaskGenerator"
            },
            "multi_agent_coordination": {
                "agent": "RealSwarmIntelligence",
                "consensus": "RealSwarmIntelligence",
                "swarm": "RealSwarmIntelligence"
            },
            "outcome_learning": {
                "learning": "OutcomeLearningSystem",
                "pattern": "OutcomeLearningSystem",
                "prediction": "PredictiveTaskEngine"
            }
        }
        
        if area in area_components:
            for keyword, component in area_components[area].items():
                if keyword in text_lower:
                    return component
        
        # Fallback mapping
        if "task" in text_lower:
            return "IntelligentTaskGenerator"
        elif "agent" in text_lower or "swarm" in text_lower:
            return "RealSwarmIntelligence"
        elif "learning" in text_lower:
            return "OutcomeLearningSystem"
        
        return "Unknown"
    
    def _classify_implementation(self, text: str, area: str) -> str:
        """Classify the type of implementation needed."""
        text_lower = text.lower()
        
        if "template" in text_lower or "format" in text_lower:
            return "template_update"
        elif "algorithm" in text_lower or "logic" in text_lower:
            return "algorithm_improvement"
        elif "add" in text_lower or "implement" in text_lower:
            return "feature_addition"
        elif "optimize" in text_lower or "improve" in text_lower:
            return "optimization"
        else:
            return "modification"
    
    def _assess_urgency(self, text: str, area: str) -> str:
        """Assess urgency of the insight."""
        text_lower = text.lower()
        
        urgent_keywords = ["critical", "urgent", "immediate", "emergency", "broken", "failing"]
        high_keywords = ["important", "should", "must", "need to", "fix"]
        
        if any(keyword in text_lower for keyword in urgent_keywords):
            return "critical"
        elif any(keyword in text_lower for keyword in high_keywords):
            return "high"
        elif area in ["claude_interaction", "task_generation"]:
            return "high"  # These areas are generally high priority
        else:
            return "medium"
    
    def _requires_action(self, text: str) -> bool:
        """Determine if insight requires action."""
        text_lower = text.lower()
        
        action_keywords = [
            "should", "must", "need to", "implement", "add", "create", 
            "update", "modify", "improve", "optimize", "fix", "change"
        ]
        
        return any(keyword in text_lower for keyword in action_keywords)
    
    def _get_action_info(self, area: str, text: str) -> Dict:
        """Get action information for area and text."""
        if area not in self.action_patterns:
            return {"action_type": "UPDATE", "priority": "MEDIUM"}
        
        text_lower = text.lower()
        area_patterns = self.action_patterns[area]
        
        for pattern_key, action_info in area_patterns.items():
            if pattern_key in text_lower:
                return action_info
        
        # Default for area
        return {"action_type": "UPDATE", "priority": "MEDIUM"}
    
    def _estimate_effort(self, insight: Dict) -> str:
        """Estimate implementation effort."""
        text = insight.get("text", "").lower()
        implementation_type = insight.get("implementation_type", "")
        
        if "template" in text or implementation_type == "template_update":
            return "LOW"
        elif "algorithm" in text or implementation_type == "algorithm_improvement":
            return "HIGH"
        elif "add" in text or implementation_type == "feature_addition":
            return "MEDIUM"
        else:
            return "MEDIUM"
    
    def _create_success_criteria(self, insight: Dict) -> List[str]:
        """Create success criteria for the insight."""
        area = insight.get("area", "")
        
        criteria = []
        
        if area == "claude_interaction":
            criteria.extend([
                "Claude response rate improves by at least 20%",
                "Task acceptance rate increases",
                "No regression in task quality"
            ])
        elif area == "task_generation":
            criteria.extend([
                "Task completion rate improves by at least 15%",
                "Task complexity scores are generated",
                "No increase in task failure rate"
            ])
        else:
            criteria.extend([
                "Implementation completes without errors",
                "Relevant metrics show improvement",
                "No performance degradation"
            ])
        
        return criteria
    
    def _extract_metric_target(self, text: str) -> Optional[str]:
        """Extract metric improvement target from text."""
        # Look for percentage improvements
        percentage_match = re.search(r'(\d+)%', text)
        if percentage_match:
            return f"{percentage_match.group(1)}% improvement"
        
        # Look for multiplier improvements
        multiplier_match = re.search(r'(\d+)x', text)
        if multiplier_match:
            return f"{multiplier_match.group(1)}x improvement"
        
        return None
    
    def get_implementation_summary(self) -> Dict:
        """Get summary of implementation activities."""
        return {
            "total_implementations": len(self.implementation_history),
            "recent_implementations": self.implementation_history[-10:],
            "success_rate": self._calculate_implementation_success_rate(),
            "most_common_areas": self._get_most_common_areas(),
            "average_impact": self._calculate_average_impact()
        }
    
    def record_implementation_outcome(self, task_id: str, success: bool, impact: float):
        """Record outcome of an implementation."""
        self.implementation_history.append({
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "impact": impact
        })
        
        # Keep history manageable
        if len(self.implementation_history) > 500:
            self.implementation_history = self.implementation_history[-500:]
    
    def _create_task_rollback_plan(self, insight: Dict, action_info: Dict) -> Dict:
        """Create rollback plan for task implementation."""
        return {
            "backup_required": True,
            "rollback_steps": [
                "Create backup of current implementation",
                "Test rollback procedure",
                "Monitor system after rollback"
            ],
            "validation_criteria": [
                "System functionality is restored",
                "No data loss occurred",
                "Performance is back to baseline"
            ]
        }
    
    def _combine_tasks(self, tasks: List[Dict]) -> Dict:
        """Combine multiple related tasks into one."""
        if len(tasks) == 1:
            return tasks[0]
        
        # Create combined task
        combined = {
            "type": "SYSTEM_IMPROVEMENT_BATCH",
            "title": f"Combined implementation: {len(tasks)} related improvements",
            "description": "Multiple related improvements combined for efficiency:\n\n" + 
                          "\n".join([f"- {task.get('title', 'Unknown task')}" for task in tasks]),
            "subtasks": tasks,
            "priority": max([task.get("priority", "MEDIUM") for task in tasks], 
                          key=lambda p: {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(p, 2)),
            "estimated_effort": "HIGH",  # Combined tasks are more complex
            "metadata": {
                "combined_task": True,
                "task_count": len(tasks),
                "areas": list(set([task.get("metadata", {}).get("area") for task in tasks]))
            }
        }
        
        return combined
    
    def _calculate_implementation_success_rate(self) -> float:
        """Calculate implementation success rate."""
        if not self.implementation_history:
            return 0.0
        
        successful = len([h for h in self.implementation_history if h["success"]])
        return successful / len(self.implementation_history)
    
    def _get_most_common_areas(self) -> List[str]:
        """Get most common implementation areas."""
        from collections import Counter
        areas = [h.get("area", "unknown") for h in self.implementation_history]
        common_areas = Counter(areas).most_common(5)
        return [area for area, count in common_areas]
    
    def _calculate_average_impact(self) -> float:
        """Calculate average impact of implementations."""
        if not self.implementation_history:
            return 0.0
        
        impacts = [h.get("impact", 0) for h in self.implementation_history]
        return sum(impacts) / len(impacts)
    
    def _assess_research_quality(self, research: Dict) -> str:
        """Assess quality of research."""
        quality_score = research.get("quality_score", 0.5)
        
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "average"
        else:
            return "poor"
    
    def _assess_implementation_success(self, cycle_results: Dict) -> str:
        """Assess implementation success."""
        implementations = cycle_results.get("implementations", [])
        
        if not implementations:
            return "not_attempted"
        
        successful = len([i for i in implementations if "implemented" in i.get("status", "")])
        
        if successful == len(implementations):
            return "fully_implemented"
        elif successful > 0:
            return "partially_implemented"
        else:
            return "failed"
    
    def _assess_performance_impact(self, cycle_results: Dict) -> str:
        """Assess performance impact."""
        changes = cycle_results.get("performance_changes", {})
        
        if not changes:
            return "no_change"
        
        positive_changes = len([c for c in changes.values() 
                              if isinstance(c, dict) and c.get("change", 0) > 0])
        
        total_changes = len(changes)
        
        if positive_changes == 0:
            return "regression"
        elif positive_changes / total_changes >= 0.8:
            return "high_improvement"
        elif positive_changes / total_changes >= 0.5:
            return "moderate_improvement"
        else:
            return "small_improvement"
    
    def _calculate_value_delivered(self, cycle_results: Dict) -> float:
        """Calculate value delivered by research cycle."""
        # Simple value calculation based on implementations and improvements
        implementations = len(cycle_results.get("implementations", []))
        performance_changes = cycle_results.get("performance_changes", {})
        
        positive_changes = len([c for c in performance_changes.values() 
                              if isinstance(c, dict) and c.get("change", 0) > 0])
        
        return min(10.0, implementations * 2 + positive_changes)
    
    def _calculate_research_effectiveness(self, research: Dict, cycle_results: Dict) -> float:
        """Calculate effectiveness of specific research."""
        quality_score = research.get("quality_score", 0.5)
        
        # Check if research led to implementations
        implementations = len(cycle_results.get("implementations", []))
        implementation_bonus = min(0.3, implementations * 0.1)
        
        # Check performance improvements
        performance_changes = cycle_results.get("performance_changes", {})
        positive_changes = len([c for c in performance_changes.values() 
                              if isinstance(c, dict) and c.get("change", 0) > 0])
        performance_bonus = min(0.2, positive_changes * 0.05)
        
        return min(1.0, quality_score + implementation_bonus + performance_bonus)