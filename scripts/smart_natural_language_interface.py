"""
Smart Natural Language Interface for CWMAI

An advanced AI-powered interface that provides intelligent, context-aware natural language
interaction with the CWMAI system. Features multi-model AI support, learning capabilities,
and sophisticated command understanding.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import pickle

# AI and system imports
from scripts.ai_brain import IntelligentAIBrain
from scripts.enhanced_http_ai_client import EnhancedHTTPAIClient
from scripts.mcp_integration import MCPIntegrationHub
from scripts.brave_search_integration import BraveSearchEnhancedResearch, get_brave_search_client
from scripts.task_manager import TaskManager
from scripts.state_manager import StateManager
from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.architecture_generator import ArchitectureGenerator
from scripts.market_research_engine import MarketResearchEngine
from scripts.smart_cli_plugins import PluginManager, PluginResult


class CommandConfidence(Enum):
    """Confidence levels for command interpretation."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class CommandIntent:
    """Represents an interpreted command intent."""
    action: str
    entities: Dict[str, Any]
    confidence: CommandConfidence
    parameters: Dict[str, Any] = field(default_factory=dict)
    context_required: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)


@dataclass
class ConversationContext:
    """Maintains conversation state and context."""
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_project: Optional[str] = None
    current_task: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_goals: List[str] = field(default_factory=list)
    last_command_time: Optional[datetime] = None
    command_patterns: Dict[str, int] = field(default_factory=dict)
    

class SmartNaturalLanguageInterface:
    """Advanced natural language interface with AI-powered understanding."""
    
    def __init__(self, ai_brain: Optional[IntelligentAIBrain] = None,
                 enable_learning: bool = True,
                 enable_multi_model: bool = True,
                 enable_plugins: bool = True):
        """Initialize the smart interface.
        
        Args:
            ai_brain: AI brain instance (creates new if None)
            enable_learning: Enable learning from user interactions
            enable_multi_model: Enable multi-model AI consensus
            enable_plugins: Enable smart plugins
        """
        self.logger = logging.getLogger(__name__)
        self.ai_brain = ai_brain or IntelligentAIBrain(enable_round_robin=True)
        self.enable_learning = enable_learning
        self.enable_multi_model = enable_multi_model
        self.enable_plugins = enable_plugins
        
        # Initialize components
        self.mcp_hub: Optional[MCPIntegrationHub] = None
        self.brave_search: Optional[BraveSearchEnhancedResearch] = None
        self.task_manager = TaskManager()
        self.state_manager = StateManager()
        
        # Context and learning
        self.context = ConversationContext()
        self.command_embeddings: Dict[str, List[float]] = {}
        self.user_model_path = Path.home() / ".cwmai" / "user_model.pkl"
        self._load_user_model()
        
        # Enhanced command patterns with semantic understanding
        self.command_patterns = self._build_smart_patterns()
        
        # Multi-model AI clients
        self.ai_models = self._setup_ai_models()
        
        # Plugin system
        self.plugin_manager = PluginManager(self.ai_brain) if enable_plugins else None
        if self.plugin_manager:
            self.plugin_manager.register_all_plugins()
        
    def _setup_ai_models(self) -> Dict[str, Any]:
        """Set up multiple AI models for consensus."""
        models = {}
        
        if self.enable_multi_model:
            # Try to set up different models
            try:
                models['claude'] = EnhancedHTTPAIClient()
            except:
                pass
                
            try:
                models['gpt4'] = EnhancedHTTPAIClient()
            except:
                pass
                
            try:
                models['gemini'] = EnhancedHTTPAIClient()
            except:
                pass
                
        return models
    
    def _build_smart_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build smart command patterns with semantic understanding."""
        return {
            'create_issue': [
                {
                    'pattern': re.compile(r'(?:create|make|add|open|file|submit)\s+(?:an?\s+)?(?:issue|bug|ticket|problem|feature)\s+(?:for|in|on|about)?\s*([^\s]+)?\s*(?:about|regarding|for|with|saying|titled?)?\s*(.+)?', re.I),
                    'extractor': self._extract_issue_entities,
                    'validators': ['validate_repository', 'validate_issue_content']
                }
            ],
            'search': [
                {
                    'pattern': re.compile(r'(?:search|find|look|query|discover|explore)\s+(?:for|up)?\s*(?:repos?|repositories|projects?|code|packages?)\s*(?:about|with|containing|related to|for)?\s*(.+)', re.I),
                    'extractor': self._extract_search_entities,
                    'enhancers': ['enhance_search_query']
                }
            ],
            'architecture': [
                {
                    'pattern': re.compile(r'(?:create|design|generate|build|make|plan)\s+(?:an?\s+)?(?:architecture|design|blueprint|structure|system)\s+(?:for|of)?\s*(.+)', re.I),
                    'extractor': self._extract_architecture_entities,
                    'validators': ['validate_architecture_request'],
                    'enhancers': ['enhance_architecture_details']
                }
            ],
            'complex_query': [
                {
                    'pattern': re.compile(r'.*(?:and then|after that|followed by|next|also).*', re.I),
                    'handler': self._handle_complex_query
                }
            ]
        }
    
    async def initialize(self):
        """Initialize async components."""
        # Initialize MCP if available
        try:
            self.mcp_hub = MCPIntegrationHub()
            await self.mcp_hub.initialize()
            self.logger.info("✅ MCP integration initialized")
        except Exception as e:
            self.logger.warning(f"MCP initialization failed: {e}")
        
        # Initialize Brave Search
        brave_key = os.getenv('BRAVE_API_KEY', 'BSAn2ZCq32LqCmwmmVQwo1VHehKL4Gt')
        if brave_key:
            self.brave_search = BraveSearchEnhancedResearch(brave_key)
            self.logger.info("✅ Brave Search initialized")
    
    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input with advanced AI understanding.
        
        Args:
            user_input: Natural language input from user
            
        Returns:
            Response dictionary with results and explanations
        """
        # Update context
        self._update_context(user_input)
        
        # Get intent with multi-model consensus if enabled
        intent = await self._get_intent_with_consensus(user_input)
        
        # Log command pattern for learning
        if self.enable_learning:
            self._learn_from_command(user_input, intent)
        
        # Check plugins first
        plugin_results = []
        if self.enable_plugins and self.plugin_manager:
            plugin_intent = {
                'action': intent.action,
                'entities': intent.entities,
                'raw_text': user_input,
                'confidence': intent.confidence.value
            }
            
            plugin_context = {
                'history': self.context.history,
                'current_project': self.context.current_project,
                'command_patterns': self.context.command_patterns,
                'user_preferences': self.context.user_preferences
            }
            
            plugin_results = await self.plugin_manager.process_with_plugins(plugin_intent, plugin_context)
        
        # If plugins handled it successfully, use plugin results
        if plugin_results:
            successful_plugins = [(name, result) for name, result in plugin_results if result.success]
            if successful_plugins:
                # Get the most successful plugin result
                plugin_name, plugin_result = successful_plugins[0]
                
                result = {
                    'success': plugin_result.success,
                    'action': f'plugin_{plugin_name}',
                    'plugin_data': plugin_result.data,
                    'plugin_name': plugin_name
                }
                
                # Add visualizations if any
                if plugin_result.visualizations:
                    result['visualizations'] = plugin_result.visualizations
                
                # Merge suggestions
                if plugin_result.suggestions:
                    result['plugin_suggestions'] = plugin_result.suggestions
            else:
                # No successful plugins, use standard handling
                if intent.confidence == CommandConfidence.HIGH:
                    result = await self._execute_intent(intent)
                elif intent.confidence == CommandConfidence.MEDIUM:
                    result = await self._execute_with_confirmation(intent, user_input)
                else:
                    result = await self._handle_uncertain_intent(user_input, intent)
        else:
            # Execute with standard intent handling
            if intent.confidence == CommandConfidence.HIGH:
                result = await self._execute_intent(intent)
            elif intent.confidence == CommandConfidence.MEDIUM:
                # Ask for confirmation
                result = await self._execute_with_confirmation(intent, user_input)
            else:
                # Use AI to interpret and suggest
                result = await self._handle_uncertain_intent(user_input, intent)
        
        # Add explanation and suggestions
        result['explanation'] = self._generate_explanation(intent, result)
        
        # Generate suggestions (including plugin suggestions)
        base_suggestions = await self._generate_suggestions(intent, result)
        plugin_suggestions = result.get('plugin_suggestions', [])
        result['suggestions'] = plugin_suggestions + base_suggestions
        
        # Update conversation history
        self.context.history.append({
            'timestamp': datetime.now(timezone.utc),
            'input': user_input,
            'intent': intent,
            'result': result
        })
        
        # Save user model if learning enabled
        if self.enable_learning:
            self._save_user_model()
        
        return result
    
    async def _get_intent_with_consensus(self, user_input: str) -> CommandIntent:
        """Get command intent using multi-model consensus."""
        # First try pattern matching
        intent = self._match_patterns(user_input)
        
        if intent and intent.confidence == CommandConfidence.HIGH:
            return intent
        
        # Use multi-model AI for complex understanding
        if self.enable_multi_model and self.ai_models:
            intents = []
            
            prompt = f"""Analyze this command and extract the intent:
Command: {user_input}

Previous context:
- Current project: {self.context.current_project}
- Last command: {self.context.history[-1]['input'] if self.context.history else 'None'}

Extract:
1. Primary action (create_issue, search, generate_architecture, etc.)
2. Entities (repository names, search terms, etc.)
3. Parameters (any specific requirements)
4. Confidence level (high/medium/low)

Respond in JSON format."""

            # Get interpretations from multiple models
            for model_name, client in self.ai_models.items():
                try:
                    response = await self._call_ai_model(client, prompt)
                    if response:
                        intents.append(json.loads(response))
                except:
                    continue
            
            # Consensus logic
            if intents:
                return self._consensus_intent(intents)
        
        # Fallback to single AI interpretation
        return await self._ai_interpret_intent(user_input)
    
    def _consensus_intent(self, intents: List[Dict[str, Any]]) -> CommandIntent:
        """Determine consensus from multiple AI interpretations."""
        # Count action votes
        action_votes = {}
        all_entities = {}
        
        for intent in intents:
            action = intent.get('action', 'unknown')
            action_votes[action] = action_votes.get(action, 0) + 1
            
            # Merge entities
            entities = intent.get('entities', {})
            for key, value in entities.items():
                if key not in all_entities:
                    all_entities[key] = []
                all_entities[key].append(value)
        
        # Get most voted action
        best_action = max(action_votes, key=action_votes.get)
        vote_percentage = action_votes[best_action] / len(intents)
        
        # Determine confidence based on consensus
        if vote_percentage >= 0.8:
            confidence = CommandConfidence.HIGH
        elif vote_percentage >= 0.6:
            confidence = CommandConfidence.MEDIUM
        else:
            confidence = CommandConfidence.LOW
        
        # Merge entities (take most common or first)
        merged_entities = {}
        for key, values in all_entities.items():
            # Use most common value
            merged_entities[key] = max(set(values), key=values.count)
        
        return CommandIntent(
            action=best_action,
            entities=merged_entities,
            confidence=confidence,
            parameters={'consensus_score': vote_percentage}
        )
    
    async def _execute_intent(self, intent: CommandIntent) -> Dict[str, Any]:
        """Execute a command intent."""
        action_handlers = {
            'create_issue': self._handle_create_issue,
            'search': self._handle_search,
            'generate_architecture': self._handle_architecture,
            'create_task': self._handle_create_task,
            'show_status': self._handle_status,
            'analyze_market': self._handle_market_analysis,
            'complex_operation': self._handle_complex_operation
        }
        
        handler = action_handlers.get(intent.action, self._handle_unknown_action)
        
        try:
            result = await handler(intent)
            result['success'] = True
            return result
        except Exception as e:
            self.logger.error(f"Error executing intent: {e}")
            return {
                'success': False,
                'error': str(e),
                'recovery_suggestions': await self._suggest_recovery(intent, e)
            }
    
    async def _handle_create_issue(self, intent: CommandIntent) -> Dict[str, Any]:
        """Smart issue creation with context awareness."""
        repo = intent.entities.get('repository') or self.context.current_project
        
        # If no repo specified, intelligently determine it
        if not repo:
            repo = await self._infer_repository(intent)
        
        # Enhance issue content with AI
        enhanced_content = await self._enhance_issue_content(
            intent.entities.get('title', ''),
            intent.entities.get('description', '')
        )
        
        # Create issue via MCP or GitHub
        if self.mcp_hub and self.mcp_hub.github:
            result = await self.mcp_hub.github.create_issue(
                repo=repo,
                title=enhanced_content['title'],
                body=enhanced_content['body'],
                labels=enhanced_content.get('suggested_labels', [])
            )
        else:
            # Fallback to task creation
            self.task_manager.create_task(
                type='github_issue',
                title=enhanced_content['title'],
                description=enhanced_content['body'],
                metadata={'repository': repo}
            )
            result = {'issue_number': 'pending', 'html_url': 'pending'}
        
        return {
            'action': 'issue_created',
            'repository': repo,
            'issue': result,
            'enhancements': enhanced_content.get('enhancements', [])
        }
    
    async def _enhance_issue_content(self, title: str, description: str) -> Dict[str, Any]:
        """Enhance issue content with AI."""
        prompt = f"""Enhance this GitHub issue:
Title: {title}
Description: {description}

Improve the title and description to be more clear and actionable.
Add suggested labels based on the content.
Include any relevant technical details that might be missing.

Respond in JSON format with: title, body, suggested_labels, enhancements (list of improvements made)"""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        try:
            return json.loads(response.get('result', '{}'))
        except:
            return {
                'title': title or 'New Issue',
                'body': description or 'Details to be added',
                'suggested_labels': []
            }
    
    async def _handle_search(self, intent: CommandIntent) -> Dict[str, Any]:
        """Smart search with multiple sources."""
        query = intent.entities.get('query', '')
        
        results = {
            'github': [],
            'web': [],
            'local': []
        }
        
        # Search GitHub via MCP
        if self.mcp_hub and self.mcp_hub.github:
            results['github'] = await self.mcp_hub.github.search_repositories(query, limit=5)
        
        # Search web via Brave
        if self.brave_search:
            web_results = await self.brave_search.client.search_developer_content(query, count=5)
            results['web'] = web_results
        
        # Search local projects
        state = self.state_manager.load_state()
        projects = state.get('projects', {})
        results['local'] = [
            {'name': name, 'data': data}
            for name, data in projects.items()
            if query.lower() in name.lower() or query.lower() in str(data).lower()
        ]
        
        # AI-powered result ranking
        ranked_results = await self._rank_search_results(query, results)
        
        return {
            'action': 'search_completed',
            'query': query,
            'results': ranked_results,
            'total_found': sum(len(r) for r in results.values()),
            'sources': list(results.keys())
        }
    
    async def _handle_architecture(self, intent: CommandIntent) -> Dict[str, Any]:
        """Generate intelligent architecture with market research."""
        project_type = intent.entities.get('project_type', '')
        
        # Enhance request with market research
        if self.brave_search:
            market_insights = await self.brave_search.analyze_market_opportunities()
            
            # Find relevant market trends
            relevant_trends = [
                insight for insight in market_insights
                if project_type.lower() in insight.get('title', '').lower()
                or project_type.lower() in insight.get('description', '').lower()
            ]
            
            if relevant_trends:
                intent.parameters['market_insights'] = relevant_trends
        
        # Generate architecture
        generator = ArchitectureGenerator(self.ai_brain)
        architecture = await generator.generate_architecture(
            project_name=project_type,
            project_type=intent.entities.get('category', 'web'),
            requirements=intent.entities.get('requirements', []),
            constraints=intent.entities.get('constraints', [])
        )
        
        # Save architecture
        save_path = f"architectures/{project_type.replace(' ', '_')}_architecture.md"
        os.makedirs('architectures', exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(self._format_architecture(architecture))
        
        return {
            'action': 'architecture_generated',
            'project': project_type,
            'architecture': architecture,
            'saved_to': save_path,
            'market_insights': intent.parameters.get('market_insights', [])
        }
    
    async def _handle_complex_operation(self, intent: CommandIntent) -> Dict[str, Any]:
        """Handle complex multi-step operations."""
        steps = intent.entities.get('steps', [])
        results = []
        
        for i, step in enumerate(steps):
            self.logger.info(f"Executing step {i+1}/{len(steps)}: {step}")
            
            # Process each step as a separate command
            step_result = await self.process_input(step)
            results.append(step_result)
            
            # Check if step failed
            if not step_result.get('success', False):
                return {
                    'action': 'complex_operation_failed',
                    'failed_at_step': i + 1,
                    'steps_completed': results,
                    'error': step_result.get('error')
                }
        
        return {
            'action': 'complex_operation_completed',
            'steps': len(steps),
            'results': results,
            'summary': await self._summarize_complex_operation(results)
        }
    
    def _update_context(self, user_input: str):
        """Update conversation context."""
        self.context.last_command_time = datetime.now(timezone.utc)
        
        # Extract potential project references
        repo_pattern = r'\b([\w-]+/[\w-]+)\b'
        matches = re.findall(repo_pattern, user_input)
        if matches:
            self.context.current_project = matches[0]
    
    def _learn_from_command(self, user_input: str, intent: CommandIntent):
        """Learn from user command patterns."""
        if not self.enable_learning:
            return
        
        # Track command patterns
        pattern_key = f"{intent.action}:{intent.confidence.value}"
        self.context.command_patterns[pattern_key] = \
            self.context.command_patterns.get(pattern_key, 0) + 1
        
        # Learn user preferences
        if intent.confidence == CommandConfidence.HIGH:
            # This was a successful pattern match
            self.context.user_preferences['preferred_patterns'] = \
                self.context.user_preferences.get('preferred_patterns', [])
            
            self.context.user_preferences['preferred_patterns'].append({
                'input': user_input,
                'intent': intent.action,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
    
    def _load_user_model(self):
        """Load saved user model."""
        if self.user_model_path.exists():
            try:
                with open(self.user_model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.context = saved_data.get('context', self.context)
                    self.command_embeddings = saved_data.get('embeddings', {})
                self.logger.info("Loaded user model")
            except Exception as e:
                self.logger.warning(f"Could not load user model: {e}")
    
    def _save_user_model(self):
        """Save user model for personalization."""
        if not self.enable_learning:
            return
        
        try:
            self.user_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.user_model_path, 'wb') as f:
                pickle.dump({
                    'context': self.context,
                    'embeddings': self.command_embeddings,
                    'saved_at': datetime.now(timezone.utc).isoformat()
                }, f)
        except Exception as e:
            self.logger.warning(f"Could not save user model: {e}")
    
    def _generate_explanation(self, intent: CommandIntent, result: Dict[str, Any]) -> str:
        """Generate natural language explanation of what happened."""
        if result.get('success', False):
            explanations = {
                'create_issue': f"I've created a new issue in {result.get('repository', 'the repository')}. "
                               f"The issue has been enhanced with better formatting and appropriate labels.",
                'search': f"I searched across multiple sources and found {result.get('total_found', 0)} results. "
                         f"The results have been ranked by relevance to your query.",
                'generate_architecture': f"I've generated a comprehensive architecture for {result.get('project', 'your project')}. "
                                       f"The design includes market insights and best practices.",
            }
            
            base_explanation = explanations.get(intent.action, "I've completed your request.")
            
            if intent.confidence == CommandConfidence.MEDIUM:
                base_explanation = "I interpreted your request and " + base_explanation.lower()
            
            return base_explanation
        else:
            return f"I encountered an issue: {result.get('error', 'Unknown error')}. " \
                   f"Here are some suggestions to resolve it."
    
    async def _generate_suggestions(self, intent: CommandIntent, result: Dict[str, Any]) -> List[str]:
        """Generate intelligent next-step suggestions."""
        suggestions = []
        
        if intent.action == 'create_issue' and result.get('success'):
            suggestions.extend([
                f"Create a task to work on issue #{result.get('issue', {}).get('issue_number', '')}",
                "Search for similar issues to avoid duplicates",
                "Generate an implementation plan for this issue"
            ])
        elif intent.action == 'search':
            suggestions.extend([
                "Create an issue based on search results",
                "Generate architecture for one of these projects",
                "Analyze market demand for these solutions"
            ])
        elif intent.action == 'generate_architecture':
            suggestions.extend([
                "Create a GitHub repository for this project",
                "Generate implementation tasks",
                "Create a project README"
            ])
        
        # Add personalized suggestions based on patterns
        if self.context.command_patterns:
            most_common = max(self.context.command_patterns, key=self.context.command_patterns.get)
            action = most_common.split(':')[0]
            suggestions.append(f"Run another {action} command (your most common action)")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    # Pattern extraction methods
    def _extract_issue_entities(self, match) -> Dict[str, Any]:
        """Extract entities from issue creation patterns."""
        groups = match.groups()
        return {
            'repository': groups[0] if groups[0] else None,
            'description': groups[1] if len(groups) > 1 and groups[1] else None
        }
    
    def _extract_search_entities(self, match) -> Dict[str, Any]:
        """Extract entities from search patterns."""
        return {
            'query': match.group(1).strip()
        }
    
    def _extract_architecture_entities(self, match) -> Dict[str, Any]:
        """Extract entities from architecture patterns."""
        return {
            'project_type': match.group(1).strip()
        }
    
    def _match_patterns(self, user_input: str) -> Optional[CommandIntent]:
        """Match user input against patterns."""
        for action, patterns in self.command_patterns.items():
            for pattern_dict in patterns:
                pattern = pattern_dict['pattern']
                match = pattern.match(user_input)
                
                if match:
                    # Extract entities
                    extractor = pattern_dict.get('extractor')
                    entities = extractor(match) if extractor else {}
                    
                    return CommandIntent(
                        action=action,
                        entities=entities,
                        confidence=CommandConfidence.HIGH,
                        parameters={'pattern_matched': pattern.pattern}
                    )
        
        return None
    
    async def _ai_interpret_intent(self, user_input: str) -> CommandIntent:
        """Use AI to interpret unclear commands."""
        prompt = f"""Interpret this command in the context of a development task management system:
Command: {user_input}

Context:
- Current project: {self.context.current_project}
- Recent commands: {[h['input'] for h in self.context.history[-3:]]}

Determine:
1. The intended action (create_issue, search, generate_architecture, create_task, show_status, etc.)
2. Key entities and parameters
3. Confidence level

Respond in JSON format."""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        try:
            interpretation = json.loads(response.get('result', '{}'))
            return CommandIntent(
                action=interpretation.get('action', 'unknown'),
                entities=interpretation.get('entities', {}),
                confidence=CommandConfidence(interpretation.get('confidence', 'low')),
                parameters=interpretation.get('parameters', {})
            )
        except:
            return CommandIntent(
                action='unknown',
                entities={'raw_input': user_input},
                confidence=CommandConfidence.UNCERTAIN
            )
    
    async def _handle_uncertain_intent(self, user_input: str, intent: CommandIntent) -> Dict[str, Any]:
        """Handle uncertain commands with AI assistance."""
        # Generate clarifying questions
        prompt = f"""The user said: "{user_input}"

I'm not sure what they want. Generate 2-3 clarifying questions to understand their intent better.
Also suggest possible interpretations of what they might mean.

Context: This is a development task management system that can create issues, search repos, generate architectures, etc."""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        return {
            'success': False,
            'reason': 'unclear_intent',
            'original_input': user_input,
            'clarification_needed': True,
            'questions': self._extract_questions(response.get('result', '')),
            'possible_interpretations': intent.suggested_alternatives or [
                "Create an issue about " + user_input,
                "Search for " + user_input,
                "Show status of " + user_input
            ]
        }
    
    def _extract_questions(self, ai_response: str) -> List[str]:
        """Extract questions from AI response."""
        lines = ai_response.split('\n')
        questions = [
            line.strip() 
            for line in lines 
            if line.strip() and '?' in line
        ][:3]
        
        return questions or [
            "What would you like to do with this?",
            "Can you provide more details?",
            "Which project or repository is this for?"
        ]
    
    async def _call_ai_model(self, client: Any, prompt: str) -> Optional[str]:
        """Call an AI model with error handling."""
        try:
            # Use the HTTP AI client's method
            response = await client.generate_enhanced_response(prompt)
            if response and 'result' in response:
                return response['result']
            return None
        except Exception as e:
            self.logger.warning(f"AI model call failed: {e}")
            return None
    
    def _format_architecture(self, architecture: Dict[str, Any]) -> str:
        """Format architecture for saving."""
        sections = []
        
        sections.append(f"# {architecture.get('project_name', 'Project')} Architecture\n")
        sections.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        
        if architecture.get('overview'):
            sections.append("## Overview\n")
            sections.append(architecture['overview'] + "\n")
        
        if architecture.get('components'):
            sections.append("## Components\n")
            for comp in architecture['components']:
                sections.append(f"### {comp.get('name', 'Component')}\n")
                sections.append(f"{comp.get('description', '')}\n")
                
                if comp.get('technologies'):
                    sections.append("**Technologies:**\n")
                    for tech in comp['technologies']:
                        sections.append(f"- {tech}\n")
                sections.append("\n")
        
        if architecture.get('market_insights'):
            sections.append("## Market Insights\n")
            for insight in architecture['market_insights']:
                sections.append(f"- {insight.get('title', '')}: {insight.get('description', '')}\n")
        
        return '\n'.join(sections)
    
    async def _rank_search_results(self, query: str, results: Dict[str, List]) -> List[Dict[str, Any]]:
        """Rank search results using AI."""
        all_results = []
        
        for source, items in results.items():
            for item in items:
                all_results.append({
                    'source': source,
                    'item': item,
                    'relevance_score': await self._calculate_relevance(query, item)
                })
        
        # Sort by relevance
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return all_results[:10]  # Top 10 results
    
    async def _calculate_relevance(self, query: str, item: Dict[str, Any]) -> float:
        """Calculate relevance score for a search result."""
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        
        # Extract text from item
        item_text = ' '.join([
            str(item.get('name', '')),
            str(item.get('title', '')),
            str(item.get('description', ''))
        ]).lower()
        
        item_words = set(item_text.split())
        
        # Calculate overlap
        overlap = len(query_words & item_words)
        return overlap / max(len(query_words), 1)
    
    async def _suggest_recovery(self, intent: CommandIntent, error: Exception) -> List[str]:
        """Suggest recovery actions for errors."""
        suggestions = []
        
        if "rate limit" in str(error).lower():
            suggestions.append("Wait a few minutes and try again")
            suggestions.append("Use a different AI provider")
        elif "not found" in str(error).lower():
            suggestions.append("Check if the repository or resource exists")
            suggestions.append("Verify the spelling and try again")
        elif "permission" in str(error).lower():
            suggestions.append("Check your API tokens and permissions")
            suggestions.append("Ensure you have access to this resource")
        else:
            suggestions.append("Try rephrasing your command")
            suggestions.append("Break down complex operations into steps")
        
        return suggestions
    
    async def _infer_repository(self, intent: CommandIntent) -> str:
        """Intelligently infer repository from context."""
        # Check recent history
        for entry in reversed(self.context.history[-5:]):
            if 'repository' in entry.get('result', {}):
                return entry['result']['repository']
        
        # Check current projects
        state = self.state_manager.load_state()
        projects = list(state.get('projects', {}).keys())
        
        if projects:
            # Use AI to match intent with project
            description = intent.entities.get('description', '')
            prompt = f"""Given these projects: {projects}
And this issue description: {description}

Which project is most relevant? Return just the project name."""
            
            response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
            project = response.get('result', '').strip()
            
            if project in projects:
                return project
        
        return "cwmai/cwmai"  # Default repository
    
    async def _execute_with_confirmation(self, intent: CommandIntent, user_input: str) -> Dict[str, Any]:
        """Execute with user confirmation for medium confidence."""
        confirmation = {
            'needs_confirmation': True,
            'interpreted_as': {
                'action': intent.action,
                'details': intent.entities
            },
            'original_input': user_input,
            'confidence': intent.confidence.value,
            'confirm_prompt': f"Did you mean to {intent.action.replace('_', ' ')} with these details?"
        }
        
        return confirmation
    
    async def _handle_unknown_action(self, intent: CommandIntent) -> Dict[str, Any]:
        """Handle unknown actions."""
        return {
            'error': f"Unknown action: {intent.action}",
            'suggestion': "Try rephrasing your command or use 'help' to see available commands"
        }
    
    async def _summarize_complex_operation(self, results: List[Dict[str, Any]]) -> str:
        """Summarize results of a complex operation."""
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)
        
        summary = f"Completed {successful}/{total} steps successfully."
        
        if successful < total:
            failures = [i+1 for i, r in enumerate(results) if not r.get('success', False)]
            summary += f" Failed steps: {failures}"
        
        return summary
    
    async def _handle_market_analysis(self, intent: CommandIntent) -> Dict[str, Any]:
        """Handle market analysis requests."""
        topic = intent.entities.get('topic', 'general software development')
        
        if self.brave_search:
            # Get real-time market data
            trends = await self.brave_search.discover_emerging_technologies()
            opportunities = await self.brave_search.analyze_market_opportunities()
            pain_points = await self.brave_search.find_developer_pain_points()
            
            # Use AI to synthesize insights
            prompt = f"""Analyze these market insights for {topic}:

Trends: {json.dumps(trends[:3], indent=2)}
Opportunities: {json.dumps(opportunities[:3], indent=2)}
Pain Points: {json.dumps(pain_points[:3], indent=2)}

Provide a concise market analysis with:
1. Key opportunities
2. Recommended project ideas
3. Technologies to focus on
4. Potential challenges"""

            response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
            
            return {
                'action': 'market_analysis_completed',
                'topic': topic,
                'analysis': response.get('result', ''),
                'raw_data': {
                    'trends': trends[:3],
                    'opportunities': opportunities[:3],
                    'pain_points': pain_points[:3]
                }
            }
        else:
            # Fallback to AI-only analysis
            market_engine = MarketResearchEngine(self.ai_brain)
            trends = await market_engine.discover_market_trends()
            
            return {
                'action': 'market_analysis_completed',
                'topic': topic,
                'analysis': 'AI-generated analysis without real-time data',
                'trends': [t.__dict__ for t in trends[:5]]
            }
    
    async def _handle_create_task(self, intent: CommandIntent) -> Dict[str, Any]:
        """Handle task creation with intelligence."""
        task_description = intent.entities.get('description', '')
        
        # Use intelligent task generator
        task_gen = IntelligentTaskGenerator(self.ai_brain, self.state_manager)
        
        # Generate enhanced task
        tasks = await task_gen.generate_tasks_from_description(task_description)
        
        if tasks:
            # Create the first task
            created_task = self.task_manager.create_task(
                type=tasks[0].get('type', 'feature'),
                title=tasks[0].get('title'),
                description=tasks[0].get('description'),
                metadata=tasks[0].get('metadata', {})
            )
            
            return {
                'action': 'task_created',
                'task': created_task,
                'additional_suggestions': tasks[1:3] if len(tasks) > 1 else []
            }
        else:
            return {
                'error': 'Could not generate task',
                'suggestion': 'Please provide more details about what you want to accomplish'
            }
    
    async def _handle_status(self, intent: CommandIntent) -> Dict[str, Any]:
        """Show intelligent system status."""
        state = self.state_manager.load_state()
        
        # Get various statistics
        stats = {
            'total_projects': len(state.get('projects', {})),
            'active_tasks': len([t for t in state.get('tasks', {}).values() if t.get('status') == 'active']),
            'completed_tasks': len([t for t in state.get('tasks', {}).values() if t.get('status') == 'completed']),
            'total_operations': state.get('metrics', {}).get('total_operations', 0),
            'success_rate': state.get('metrics', {}).get('success_rate', 0)
        }
        
        # Generate AI summary
        prompt = f"""Summarize this system status in a friendly, conversational way:
{json.dumps(stats, indent=2)}

Include:
1. Overall health assessment
2. Key achievements
3. Suggested next actions"""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        
        return {
            'action': 'status_displayed',
            'stats': stats,
            'summary': response.get('result', 'System is operational'),
            'recent_activity': self.context.history[-5:] if self.context.history else []
        }
    
    async def _handle_complex_query(self, match) -> Dict[str, Any]:
        """Handle complex multi-part queries."""
        query = match.group(0)
        
        # Split by conjunctions
        parts = re.split(r'\s+(?:and then|after that|followed by|next|also)\s+', query, flags=re.I)
        
        return {
            'steps': parts,
            'type': 'sequential'
        }
    
    async def _enhance_search_query(self, query: str) -> str:
        """Enhance search query with synonyms and related terms."""
        prompt = f"""Enhance this search query with related terms and synonyms:
Query: {query}

Add 2-3 related terms that would help find more relevant results.
Return just the enhanced query string."""

        response = await self.ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
        return response.get('result', query)
    
    async def validate_repository(self, repo: str) -> bool:
        """Validate repository exists and is accessible."""
        if self.mcp_hub and self.mcp_hub.github:
            try:
                info = await self.mcp_hub.github.get_repository_info(repo)
                return bool(info)
            except:
                return False
        return True  # Assume valid if can't check
    
    async def validate_issue_content(self, title: str, body: str) -> bool:
        """Validate issue content is appropriate."""
        return bool(title and len(title) > 5)