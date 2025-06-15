"""
Smart CLI Plugins - Extended Intelligence

Additional capabilities that can be plugged into the Smart CLI for
even more intelligent behavior.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PluginResult:
    """Result from a plugin execution."""
    success: bool
    data: Any
    suggestions: List[str] = None
    visualizations: List[Dict[str, Any]] = None


class SmartPlugin(ABC):
    """Base class for smart CLI plugins."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"plugin.{name}")
    
    @abstractmethod
    async def can_handle(self, intent: Dict[str, Any]) -> bool:
        """Check if this plugin can handle the intent."""
        pass
    
    @abstractmethod
    async def execute(self, intent: Dict[str, Any], context: Dict[str, Any]) -> PluginResult:
        """Execute the plugin logic."""
        pass


class AutomationPlugin(SmartPlugin):
    """Plugin for creating and managing automated workflows."""
    
    def __init__(self):
        super().__init__(
            "automation",
            "Create and manage automated workflows from natural language"
        )
        self.workflows = {}
    
    async def can_handle(self, intent: Dict[str, Any]) -> bool:
        """Check if intent is about automation."""
        action = intent.get('action', '')
        text = intent.get('raw_text', '').lower()
        
        automation_keywords = [
            'automate', 'workflow', 'schedule', 'every day', 'every week',
            'when', 'if', 'trigger', 'automatically'
        ]
        
        return any(keyword in text for keyword in automation_keywords)
    
    async def execute(self, intent: Dict[str, Any], context: Dict[str, Any]) -> PluginResult:
        """Create or manage automation."""
        text = intent.get('raw_text', '')
        
        # Parse automation intent
        automation_type = self._parse_automation_type(text)
        
        if automation_type == 'scheduled':
            workflow = await self._create_scheduled_workflow(text, context)
        elif automation_type == 'conditional':
            workflow = await self._create_conditional_workflow(text, context)
        elif automation_type == 'reactive':
            workflow = await self._create_reactive_workflow(text, context)
        else:
            workflow = await self._create_simple_workflow(text, context)
        
        # Save workflow
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        self.workflows[workflow_id] = workflow
        
        return PluginResult(
            success=True,
            data={
                'workflow_id': workflow_id,
                'workflow': workflow,
                'type': automation_type
            },
            suggestions=[
                f"Test the workflow with: test workflow {workflow_id}",
                "View all workflows with: show my automations",
                "Schedule this workflow with: schedule {workflow_id} daily at 9am"
            ]
        )
    
    def _parse_automation_type(self, text: str) -> str:
        """Determine the type of automation."""
        if any(word in text.lower() for word in ['every', 'daily', 'weekly', 'schedule']):
            return 'scheduled'
        elif any(word in text.lower() for word in ['if', 'when', 'whenever']):
            return 'conditional'
        elif any(word in text.lower() for word in ['on', 'after', 'trigger']):
            return 'reactive'
        return 'simple'
    
    async def _create_scheduled_workflow(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a scheduled workflow."""
        # Extract schedule pattern
        schedule_patterns = {
            r'every day at (\d{1,2}(?::\d{2})?\s*(?:am|pm)?)': 'daily',
            r'every week on (\w+)': 'weekly',
            r'every (\d+) hours?': 'interval',
            r'every morning': 'daily_morning',
            r'every evening': 'daily_evening'
        }
        
        schedule = None
        for pattern, schedule_type in schedule_patterns.items():
            match = re.search(pattern, text, re.I)
            if match:
                schedule = {
                    'type': schedule_type,
                    'value': match.group(1) if match.groups() else None,
                    'pattern': pattern
                }
                break
        
        # Extract actions
        action_text = re.sub(r'every.*?(?:at \d{1,2}(?::\d{2})?\s*(?:am|pm)?)?', '', text, flags=re.I).strip()
        
        return {
            'type': 'scheduled',
            'schedule': schedule or {'type': 'daily', 'value': '9:00 am'},
            'actions': self._parse_actions(action_text),
            'description': f"Scheduled workflow: {text}",
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """Parse actions from text."""
        # Split by conjunctions
        action_parts = re.split(r'\s+(?:and|then|also|,)\s+', text, flags=re.I)
        
        actions = []
        for part in action_parts:
            if part.strip():
                actions.append({
                    'type': 'command',
                    'value': part.strip(),
                    'original': part
                })
        
        return actions
    
    async def _create_conditional_workflow(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a conditional workflow."""
        # Extract condition
        condition_match = re.search(r'(?:if|when|whenever)\s+(.+?)(?:then|,)\s*(.+)', text, re.I)
        
        if condition_match:
            condition = condition_match.group(1).strip()
            actions = condition_match.group(2).strip()
        else:
            condition = "unspecified condition"
            actions = text
        
        return {
            'type': 'conditional',
            'condition': {
                'text': condition,
                'type': self._classify_condition(condition)
            },
            'actions': self._parse_actions(actions),
            'description': f"Conditional workflow: {text}",
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _classify_condition(self, condition: str) -> str:
        """Classify the type of condition."""
        if any(word in condition.lower() for word in ['error', 'fail', 'crash']):
            return 'error_condition'
        elif any(word in condition.lower() for word in ['complete', 'finish', 'done']):
            return 'completion_condition'
        elif any(word in condition.lower() for word in ['new', 'create', 'add']):
            return 'creation_condition'
        return 'custom_condition'
    
    async def _create_reactive_workflow(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a reactive workflow."""
        return {
            'type': 'reactive',
            'trigger': self._extract_trigger(text),
            'actions': self._parse_actions(text),
            'description': f"Reactive workflow: {text}",
            'created_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _extract_trigger(self, text: str) -> Dict[str, Any]:
        """Extract trigger from text."""
        trigger_patterns = {
            'issue_created': r'(?:on|when)\s+(?:new\s+)?issue',
            'pr_opened': r'(?:on|when)\s+(?:new\s+)?(?:pr|pull request)',
            'task_completed': r'(?:on|when)\s+task.*?complete',
            'error_occurred': r'(?:on|when)\s+error'
        }
        
        for trigger_type, pattern in trigger_patterns.items():
            if re.search(pattern, text, re.I):
                return {'type': trigger_type, 'pattern': pattern}
        
        return {'type': 'custom', 'text': text}
    
    async def _create_simple_workflow(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple workflow."""
        return {
            'type': 'simple',
            'actions': self._parse_actions(text),
            'description': f"Simple workflow: {text}",
            'created_at': datetime.now(timezone.utc).isoformat()
        }


class VisualizationPlugin(SmartPlugin):
    """Plugin for creating visual representations of data."""
    
    def __init__(self):
        super().__init__(
            "visualization",
            "Create charts and visualizations from natural language"
        )
    
    async def can_handle(self, intent: Dict[str, Any]) -> bool:
        """Check if intent is about visualization."""
        text = intent.get('raw_text', '').lower()
        
        viz_keywords = [
            'chart', 'graph', 'plot', 'visualize', 'show me',
            'diagram', 'timeline', 'distribution', 'trend'
        ]
        
        return any(keyword in text for keyword in viz_keywords)
    
    async def execute(self, intent: Dict[str, Any], context: Dict[str, Any]) -> PluginResult:
        """Create visualization."""
        text = intent.get('raw_text', '')
        
        # Determine visualization type
        viz_type = self._determine_viz_type(text)
        
        # Get data context
        data_source = self._extract_data_source(text, context)
        
        # Create visualization spec
        viz_spec = await self._create_viz_spec(viz_type, data_source, text)
        
        return PluginResult(
            success=True,
            data={
                'type': viz_type,
                'spec': viz_spec,
                'source': data_source
            },
            visualizations=[viz_spec],
            suggestions=[
                "Export this chart as PNG",
                "Add more data to the visualization",
                "Change chart type to " + self._suggest_alternative_viz(viz_type)
            ]
        )
    
    def _determine_viz_type(self, text: str) -> str:
        """Determine the type of visualization."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['line', 'trend', 'over time']):
            return 'line_chart'
        elif any(word in text_lower for word in ['bar', 'compare', 'comparison']):
            return 'bar_chart'
        elif any(word in text_lower for word in ['pie', 'distribution', 'breakdown']):
            return 'pie_chart'
        elif any(word in text_lower for word in ['scatter', 'correlation']):
            return 'scatter_plot'
        elif any(word in text_lower for word in ['timeline', 'gantt']):
            return 'timeline'
        elif any(word in text_lower for word in ['flow', 'diagram', 'architecture']):
            return 'diagram'
        
        return 'auto'  # Let the system decide
    
    def _extract_data_source(self, text: str, context: Dict[str, Any]) -> str:
        """Extract what data to visualize."""
        # Check for specific data mentions
        if 'task' in text.lower():
            return 'tasks'
        elif 'issue' in text.lower():
            return 'issues'
        elif 'performance' in text.lower():
            return 'performance_metrics'
        elif 'project' in text.lower():
            return 'projects'
        
        # Default to recent activity
        return 'recent_activity'
    
    async def _create_viz_spec(self, viz_type: str, data_source: str, text: str) -> Dict[str, Any]:
        """Create visualization specification."""
        # This would normally fetch real data
        # For now, return a sample spec
        
        if viz_type == 'line_chart':
            return {
                'type': 'line',
                'title': f"{data_source.replace('_', ' ').title()} Over Time",
                'data': {
                    'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
                    'datasets': [{
                        'label': data_source,
                        'data': [12, 19, 15, 25, 22]
                    }]
                },
                'options': {
                    'responsive': True,
                    'animation': True
                }
            }
        elif viz_type == 'bar_chart':
            return {
                'type': 'bar',
                'title': f"{data_source.replace('_', ' ').title()} Comparison",
                'data': {
                    'labels': ['Project A', 'Project B', 'Project C'],
                    'datasets': [{
                        'label': 'Count',
                        'data': [15, 25, 10]
                    }]
                }
            }
        elif viz_type == 'pie_chart':
            return {
                'type': 'pie',
                'title': f"{data_source.replace('_', ' ').title()} Distribution",
                'data': {
                    'labels': ['Completed', 'In Progress', 'Pending'],
                    'datasets': [{
                        'data': [60, 30, 10]
                    }]
                }
            }
        
        return {
            'type': 'auto',
            'title': 'Data Visualization',
            'description': f"Visualization for: {text}"
        }
    
    def _suggest_alternative_viz(self, current_type: str) -> str:
        """Suggest alternative visualization type."""
        alternatives = {
            'line_chart': 'bar chart',
            'bar_chart': 'line chart',
            'pie_chart': 'bar chart',
            'scatter_plot': 'line chart',
            'timeline': 'gantt chart'
        }
        return alternatives.get(current_type, 'different chart')


class ExplanationPlugin(SmartPlugin):
    """Plugin for explaining code, systems, and decisions."""
    
    def __init__(self, ai_brain):
        super().__init__(
            "explanation",
            "Provide detailed explanations of code and system behavior"
        )
        self.ai_brain = ai_brain
    
    async def can_handle(self, intent: Dict[str, Any]) -> bool:
        """Check if intent is about explanation."""
        text = intent.get('raw_text', '').lower()
        
        explain_keywords = [
            'explain', 'why', 'how does', 'what is', 'tell me about',
            'understand', 'clarify', 'describe', 'help me understand'
        ]
        
        return any(keyword in text for keyword in explain_keywords)
    
    async def execute(self, intent: Dict[str, Any], context: Dict[str, Any]) -> PluginResult:
        """Provide explanation."""
        text = intent.get('raw_text', '')
        
        # Determine what needs explanation
        subject = self._extract_subject(text)
        
        # Generate explanation
        explanation = await self._generate_explanation(subject, context)
        
        # Create learning resources
        resources = self._suggest_resources(subject)
        
        return PluginResult(
            success=True,
            data={
                'subject': subject,
                'explanation': explanation,
                'resources': resources,
                'complexity_level': self._assess_complexity(subject)
            },
            suggestions=[
                f"Show me examples of {subject}",
                f"How do I implement {subject}?",
                "Explain in simpler terms"
            ]
        )
    
    def _extract_subject(self, text: str) -> str:
        """Extract what needs to be explained."""
        # Remove explanation keywords
        cleaned = re.sub(
            r'(explain|tell me about|what is|how does|help me understand)\s+',
            '',
            text,
            flags=re.I
        ).strip()
        
        return cleaned or "the system"
    
    async def _generate_explanation(self, subject: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed explanation."""
        prompt = f"""Explain {subject} in the context of a software development system.

Provide:
1. A simple overview
2. How it works
3. Why it's useful
4. Common use cases
5. Best practices

Make the explanation clear and practical."""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        return {
            'overview': self._extract_section(response.get('result', ''), 'overview'),
            'how_it_works': self._extract_section(response.get('result', ''), 'how it works'),
            'benefits': self._extract_section(response.get('result', ''), 'useful'),
            'use_cases': self._extract_section(response.get('result', ''), 'use cases'),
            'best_practices': self._extract_section(response.get('result', ''), 'best practices')
        }
    
    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from explanation text."""
        # Simple extraction - in production would use better parsing
        lines = text.split('\n')
        capturing = False
        section_text = []
        
        for line in lines:
            if section.lower() in line.lower():
                capturing = True
                continue
            elif capturing and line.strip() and not line.startswith(' '):
                break
            elif capturing:
                section_text.append(line)
        
        return '\n'.join(section_text).strip() or f"Information about {section}"
    
    def _suggest_resources(self, subject: str) -> List[Dict[str, str]]:
        """Suggest learning resources."""
        # In production, would search actual resources
        return [
            {
                'type': 'documentation',
                'title': f"Official {subject} Documentation",
                'url': f"https://docs.example.com/{subject.replace(' ', '-')}"
            },
            {
                'type': 'tutorial',
                'title': f"Interactive {subject} Tutorial",
                'url': f"https://learn.example.com/{subject.replace(' ', '-')}"
            },
            {
                'type': 'example',
                'title': f"{subject} Examples",
                'url': f"https://github.com/examples/{subject.replace(' ', '-')}"
            }
        ]
    
    def _assess_complexity(self, subject: str) -> str:
        """Assess complexity level of subject."""
        complex_topics = ['architecture', 'distributed', 'microservices', 'machine learning']
        simple_topics = ['variable', 'function', 'loop', 'array']
        
        subject_lower = subject.lower()
        
        if any(topic in subject_lower for topic in complex_topics):
            return 'advanced'
        elif any(topic in subject_lower for topic in simple_topics):
            return 'beginner'
        
        return 'intermediate'


class SmartSuggestionsPlugin(SmartPlugin):
    """Plugin that provides intelligent, context-aware suggestions."""
    
    def __init__(self, ai_brain):
        super().__init__(
            "smart_suggestions",
            "Provide intelligent suggestions based on context and patterns"
        )
        self.ai_brain = ai_brain
    
    async def can_handle(self, intent: Dict[str, Any]) -> bool:
        """This plugin can always provide suggestions."""
        return True
    
    async def execute(self, intent: Dict[str, Any], context: Dict[str, Any]) -> PluginResult:
        """Generate smart suggestions."""
        # Analyze current context
        analysis = await self._analyze_context(intent, context)
        
        # Generate suggestions based on multiple factors
        suggestions = await self._generate_suggestions(analysis)
        
        # Rank suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, context)
        
        return PluginResult(
            success=True,
            data={
                'suggestions': ranked_suggestions[:5],
                'context_factors': analysis,
                'confidence_scores': [s['confidence'] for s in ranked_suggestions[:5]]
            },
            suggestions=[s['text'] for s in ranked_suggestions[:3]]
        )
    
    async def _analyze_context(self, intent: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current context for suggestion generation."""
        return {
            'recent_actions': self._get_recent_actions(context),
            'current_focus': self._determine_focus(context),
            'time_context': self._get_time_context(),
            'user_patterns': self._analyze_user_patterns(context),
            'system_state': self._analyze_system_state(context)
        }
    
    def _get_recent_actions(self, context: Dict[str, Any]) -> List[str]:
        """Get recent user actions."""
        history = context.get('history', [])
        return [h.get('action', 'unknown') for h in history[-5:]]
    
    def _determine_focus(self, context: Dict[str, Any]) -> str:
        """Determine current user focus."""
        recent_projects = []
        for h in context.get('history', [])[-10:]:
            if 'repository' in h.get('result', {}):
                recent_projects.append(h['result']['repository'])
        
        if recent_projects:
            # Most mentioned project
            return max(set(recent_projects), key=recent_projects.count)
        
        return 'general'
    
    def _get_time_context(self) -> Dict[str, Any]:
        """Get time-based context."""
        now = datetime.now()
        
        return {
            'time_of_day': 'morning' if now.hour < 12 else 'afternoon' if now.hour < 18 else 'evening',
            'day_of_week': now.strftime('%A'),
            'is_weekend': now.weekday() >= 5,
            'is_month_end': now.day > 25
        }
    
    def _analyze_user_patterns(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        patterns = context.get('command_patterns', {})
        
        return {
            'most_common_action': max(patterns, key=patterns.get) if patterns else None,
            'command_frequency': sum(patterns.values()) if patterns else 0,
            'diversity_score': len(patterns) / max(sum(patterns.values()), 1) if patterns else 0
        }
    
    def _analyze_system_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state."""
        return {
            'has_active_tasks': bool(context.get('active_tasks', [])),
            'has_pending_issues': bool(context.get('pending_issues', [])),
            'last_success': context.get('last_success', True)
        }
    
    async def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on analysis."""
        suggestions = []
        
        # Time-based suggestions
        time_ctx = analysis['time_context']
        if time_ctx['time_of_day'] == 'morning':
            suggestions.append({
                'text': "Review overnight issues and automated task results",
                'type': 'routine',
                'confidence': 0.8
            })
        elif time_ctx['is_month_end']:
            suggestions.append({
                'text': "Generate monthly performance report",
                'type': 'periodic',
                'confidence': 0.9
            })
        
        # Pattern-based suggestions
        patterns = analysis['user_patterns']
        if patterns['most_common_action']:
            suggestions.append({
                'text': f"Run your common {patterns['most_common_action']} workflow",
                'type': 'pattern',
                'confidence': 0.7
            })
        
        # State-based suggestions
        state = analysis['system_state']
        if state['has_active_tasks']:
            suggestions.append({
                'text': "Check progress on active tasks",
                'type': 'followup',
                'confidence': 0.85
            })
        
        # Focus-based suggestions
        focus = analysis['current_focus']
        if focus != 'general':
            suggestions.extend([
                {
                    'text': f"Generate architecture improvements for {focus}",
                    'type': 'improvement',
                    'confidence': 0.75
                },
                {
                    'text': f"Search for similar projects to {focus} for inspiration",
                    'type': 'research',
                    'confidence': 0.7
                }
            ])
        
        # Recent action continuations
        recent = analysis['recent_actions']
        if 'create_issue' in recent:
            suggestions.append({
                'text': "Create tasks for the issues you just created",
                'type': 'continuation',
                'confidence': 0.8
            })
        
        return suggestions
    
    def _rank_suggestions(self, suggestions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank suggestions by relevance."""
        # Add context scoring
        for suggestion in suggestions:
            # Boost score based on type and context
            if suggestion['type'] == 'routine' and self._is_routine_time(context):
                suggestion['confidence'] *= 1.2
            elif suggestion['type'] == 'continuation' and self._has_unfinished_work(context):
                suggestion['confidence'] *= 1.3
            
            # Cap at 1.0
            suggestion['confidence'] = min(suggestion['confidence'], 1.0)
        
        # Sort by confidence
        return sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
    
    def _is_routine_time(self, context: Dict[str, Any]) -> bool:
        """Check if it's typical routine time."""
        # Simple check - could be more sophisticated
        hour = datetime.now().hour
        return hour in [9, 10, 14, 15]  # Common work routine times
    
    def _has_unfinished_work(self, context: Dict[str, Any]) -> bool:
        """Check for unfinished work."""
        return bool(context.get('active_tasks', [])) or bool(context.get('pending_issues', []))


# Plugin registry
AVAILABLE_PLUGINS = {
    'automation': AutomationPlugin,
    'visualization': VisualizationPlugin,
    'explanation': ExplanationPlugin,
    'smart_suggestions': SmartSuggestionsPlugin
}


class PluginManager:
    """Manages smart CLI plugins."""
    
    def __init__(self, ai_brain=None):
        self.plugins: Dict[str, SmartPlugin] = {}
        self.ai_brain = ai_brain
        self.logger = logging.getLogger("PluginManager")
    
    def register_plugin(self, plugin_name: str, plugin_class: type = None):
        """Register a plugin."""
        if plugin_class is None and plugin_name in AVAILABLE_PLUGINS:
            plugin_class = AVAILABLE_PLUGINS[plugin_name]
        
        if plugin_class:
            # Initialize plugin with AI brain if needed
            if plugin_name in ['explanation', 'smart_suggestions']:
                plugin = plugin_class(self.ai_brain)
            else:
                plugin = plugin_class()
            
            self.plugins[plugin_name] = plugin
            self.logger.info(f"Registered plugin: {plugin_name}")
    
    def register_all_plugins(self):
        """Register all available plugins."""
        for name in AVAILABLE_PLUGINS:
            self.register_plugin(name)
    
    async def process_with_plugins(self, intent: Dict[str, Any], context: Dict[str, Any]) -> List[PluginResult]:
        """Process intent through all applicable plugins."""
        results = []
        
        for name, plugin in self.plugins.items():
            try:
                if await plugin.can_handle(intent):
                    self.logger.info(f"Plugin {name} handling intent")
                    result = await plugin.execute(intent, context)
                    results.append((name, result))
            except Exception as e:
                self.logger.error(f"Plugin {name} error: {e}")
        
        return results
    
    def get_plugin(self, name: str) -> Optional[SmartPlugin]:
        """Get a specific plugin."""
        return self.plugins.get(name)
    
    def list_plugins(self) -> List[Dict[str, str]]:
        """List all registered plugins."""
        return [
            {
                'name': name,
                'description': plugin.description
            }
            for name, plugin in self.plugins.items()
        ]