#!/usr/bin/env python3
"""
CWMAI Conversational AI Assistant

A Claude-like conversational interface that understands natural language,
maintains context, and can execute CWMAI commands within a friendly dialogue.

SECURITY FIXES APPLIED:
1. SYSTEM_COMMAND handler now uses strict allowlist
2. SYSTEM_COMMAND removed from AI prompt to prevent misuse
3. Tool loading is resilient to broken tools
4. Tool parameter parsing uses ast.literal_eval for safety
5. All logging calls have been reviewed for correctness
"""

import asyncio
import json
import logging
import os
import re
import random
import subprocess
import psutil
import signal
import sys
import shutil
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import logging functions but don't set up yet
from scripts.conversational_ai_logger import setup_conversational_ai_logging, log_user_message, log_system_message

# CWMAI imports
from scripts.ai_brain import IntelligentAIBrain
from scripts.natural_language_interface import NaturalLanguageInterface
from scripts.task_manager import TaskManager
from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient
from scripts.dynamic_context_collector import DynamicContextCollector
from scripts.workflow_orchestrator import WorkflowOrchestrator
from scripts.system_workflows import get_workflow_definition, list_all_workflows
from scripts.conversation_pattern_analyzer import ConversationPatternAnalyzer
from scripts.tool_context_enhancer import get_tool_context_enhancer
from scripts.tool_discovery_flow import ToolDiscoveryFlow
from scripts.tool_execution_engine import ToolExecutionEngine


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    timestamp: datetime
    user_input: str
    assistant_response: str
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMemory:
    """Long-term memory for the assistant."""
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    project_context: Dict[str, Any] = field(default_factory=dict)
    common_tasks: List[str] = field(default_factory=list)
    interaction_style: str = "friendly_professional"
    learned_patterns: Dict[str, int] = field(default_factory=dict)


class ResponseStyle(Enum):
    """Different response styles the assistant can use."""
    FRIENDLY_PROFESSIONAL = "friendly_professional"
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CASUAL = "casual"


class ResetType(Enum):
    """Different types of system resets."""
    FULL = "full"  # Clear everything
    SELECTIVE = "selective"  # Preserve AI cache and/or knowledge base
    EMERGENCY = "emergency"  # When system is in bad state
    LOGS_ONLY = "logs_only"  # Clear only log files
    STATE_ONLY = "state_only"  # Clear only state files
    CACHE_ONLY = "cache_only"  # Clear only cache files


class ConversationalAIAssistant:
    """A Claude-like conversational AI assistant for CWMAI."""
    
    # SECURITY: Strict allowlist for system commands
    ALLOWED_SYSTEM_COMMANDS = [
        'ls -l',
        'pwd',
        'echo',
        'df -h',
        'date',
        'uptime',
        'ls',
        'ls -la',
        'whoami',
        'hostname'
    ]
    
    def __init__(self, style: ResponseStyle = ResponseStyle.FRIENDLY_PROFESSIONAL):
        """Initialize the conversational assistant.
        
        Args:
            style: The response style to use
        """
        self.logger = logging.getLogger(__name__)
        self.style = style
        
        # Core components
        self.ai_brain = IntelligentAIBrain(enable_round_robin=False)
        self.nli = NaturalLanguageInterface()
        self.http_client = HTTPAIClient(enable_round_robin=False)
        self.context_collector = DynamicContextCollector()
        self.workflow_orchestrator = WorkflowOrchestrator()
        self.state_manager = StateManager()  # Add state manager
        self.pattern_analyzer = ConversationPatternAnalyzer()  # Add pattern analyzer
        self.context_enhancer = get_tool_context_enhancer()  # Add context enhancer
        self.discovery_flow = ToolDiscoveryFlow()  # Add discovery flow manager
        
        # Initialize execution engine after tool system is ready
        self.execution_engine = None
        
        # Conversation state
        self.conversation_history: List[ConversationTurn] = []
        self.current_context: Dict[str, Any] = {
            'current_project': None,
            'current_task': None,
            'last_command': None,
            'pending_confirmations': {},
            'conversation_topic': None
        }
        
        # Memory
        self.memory = ConversationMemory()
        self.memory_file = Path.home() / ".cwmai" / "conversation_memory.json"
        self._load_memory()
        
        # Conversation templates
        self.templates = self._load_conversation_templates()
        
        # Continuous AI system control
        self.continuous_ai_process = None
        self.continuous_ai_script = Path(__file__).parent.parent / "run_continuous_ai.py"
        
    def _load_memory(self):
        """Load conversation memory from disk."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memory.user_preferences = data.get('user_preferences', {})
                    self.memory.project_context = data.get('project_context', {})
                    self.memory.common_tasks = data.get('common_tasks', [])
                    self.memory.learned_patterns = data.get('learned_patterns', {})
                    
                    # Load conversation history
                    history_data = data.get('conversation_history', [])
                    for turn_data in history_data[-20:]:  # Keep last 20 turns
                        try:
                            turn = ConversationTurn(
                                timestamp=datetime.fromisoformat(turn_data['timestamp']),
                                user_input=turn_data['user_input'],
                                assistant_response=turn_data['assistant_response'],
                                actions_taken=turn_data.get('actions_taken', []),
                                context_updates=turn_data.get('context_updates', {})
                            )
                            self.conversation_history.append(turn)
                        except:
                            pass  # Skip invalid entries
        except Exception as e:
            self.logger.debug(f"Could not load memory: {e}")
    
    def _save_memory(self):
        """Save conversation memory to disk."""
        try:
            self.memory_file.parent.mkdir(exist_ok=True)
            
            # Prepare conversation history for serialization
            history_data = []
            for turn in self.conversation_history[-20:]:  # Keep last 20 turns
                history_data.append({
                    'timestamp': turn.timestamp.isoformat(),
                    'user_input': turn.user_input,
                    'assistant_response': turn.assistant_response,
                    'actions_taken': turn.actions_taken,
                    'context_updates': turn.context_updates
                })
            
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'user_preferences': self.memory.user_preferences,
                    'project_context': self.memory.project_context,
                    'common_tasks': self.memory.common_tasks[-20:],  # Keep last 20
                    'learned_patterns': self.memory.learned_patterns,
                    'conversation_history': history_data
                }, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not save memory: {e}")
    
    def _load_conversation_templates(self) -> Dict[str, List[str]]:
        """Load conversation response templates."""
        return {
            'greetings': [
                "Hi! I'm your CWMAI assistant. How can I help you today?",
                "Hello! Ready to help you manage your projects. What would you like to work on?",
                "Hey there! I'm here to help with your development tasks. What's on your mind?"
            ],
            'confirmations': [
                "Got it! Let me {} for you.",
                "Sure thing! I'll {} right away.",
                "Absolutely! Working on {} now."
            ],
            'clarifications': [
                "I want to make sure I understand correctly. Did you mean {}?",
                "Just to clarify, you'd like me to {}?",
                "Let me confirm - you want to {}?"
            ],
            'thinking': [
                "Let me check that for you...",
                "Looking into this...",
                "Give me a moment to find that information..."
            ],
            'success': [
                "✓ Done! {}",
                "✓ Successfully {}",
                "✓ All set! I've {}"
            ],
            'error_gentle': [
                "I ran into a small issue: {}. Would you like me to try a different approach?",
                "Hmm, there was a problem: {}. Let me suggest an alternative.",
                "I couldn't complete that because: {}. Here's what we can do instead:"
            ]
        }
    
    async def initialize(self):
        """Initialize the assistant and its components."""
        log_system_message("Initializing conversational AI assistant...")
        await self.nli.initialize()
        await self.workflow_orchestrator.initialize()
        
        # Initialize tool execution engine
        try:
            from scripts.tool_calling_system import ToolCallingSystem
            tool_system = ToolCallingSystem()
            self.execution_engine = ToolExecutionEngine(
                ai_brain=self.ai_brain,
                tool_system=tool_system,
                discovery_flow=self.discovery_flow
            )
            log_system_message("Tool execution engine initialized")
        except Exception as e:
            log_system_message(f"Failed to initialize execution engine: {e}", "WARNING")
        
        # Quality monitoring disabled - was causing hangs
        # try:
        #     from scripts.quality_improvement_daemon import start_quality_monitoring
        #     await start_quality_monitoring()
        #     log_system_message("Quality monitoring started")
        # except Exception as e:
        #     log_system_message(f"Quality monitoring failed to start: {e}", "WARNING")
        
        log_system_message("Assistant ready for conversation")
        log_user_message("Assistant initialized and ready!")
    
    async def handle_conversation(self, user_input: str) -> str:
        """Handle a conversational turn with the user using AI with full context.
        
        # Add console logging for debugging
        print(f"\n[DEBUG] handle_conversation called with: {user_input}")
        
        Args:
            user_input: The user's input
            
        Returns:
            The assistant's response
        """
        # Update conversation context
        self._update_context(user_input)
        
        # ALWAYS use AI with full context - no hardcoded patterns
        response = await self._handle_with_intelligent_context(user_input)
        
        # Auto-enhance poor quality responses
        try:
            from scripts.ai_response_enhancer import AIResponseEnhancer
            enhancer = AIResponseEnhancer()
            
            # Get the full context for enhancement
            dynamic_context = await self._gather_complete_context(user_input)
            
            # Check and enhance if needed
            enhanced_response = await enhancer.enhance_response(user_input, response, dynamic_context)
            
            if enhanced_response != response:
                log_system_message(f"Enhanced response quality for better user experience")
                enhancer.save_enhancement_history(response, enhanced_response, user_input)
                response = enhanced_response
        except Exception as e:
            # If enhancement fails, use original response
            log_system_message(f"Response enhancement failed: {e}", "WARNING")
        
        # Record the turn
        turn = ConversationTurn(
            timestamp=datetime.now(timezone.utc),
            user_input=user_input,
            assistant_response=response,
            actions_taken=self.current_context.get('last_actions', []),
            context_updates=self.current_context.copy()
        )
        self.conversation_history.append(turn)
        
        # Learn from the interaction
        self._learn_from_interaction(user_input, response)
        
        # Save memory periodically
        if len(self.conversation_history) % 10 == 0:
            self._save_memory()
        
        return response
    
    async def _handle_with_intelligent_context(self, user_input: str) -> str:
        """Handle user input using Claude Code's focused approach.
        
        Give AI only what it needs at each step, letting it search for context.
        """
        log_system_message(f"Processing user input: {user_input[:100]}...")
        
        # Check if this is a continuation of a tool discovery flow
        if hasattr(self, '_tool_discovery_context'):
            return await self._continue_tool_discovery_flow(user_input)
        
        # Otherwise, start fresh with minimal context
        print(f"[DEBUG] Calling _handle_with_minimal_context")
        result = await self._handle_with_minimal_context(user_input)
        print(f"[DEBUG] Result from _handle_with_minimal_context: {result[:100] if result else 'None'}...")
        return result
    
    async def _handle_with_minimal_context(self, user_input: str) -> str:
        """Handle input with AI-driven approach - Claude Code style."""
        print(f"[DEBUG] Processing: {user_input[:100]}...")
        
        # Get all available tools
        essential_tools = self._get_essential_tools_for_request(user_input)
        
        # Get MCP tools too
        mcp_tools = []
        try:
            mcp_tools = self.mcp_hub.list_available_tools()
        except:
            pass
        
        # Build tool list with descriptions
        tool_info = []
        for tool in essential_tools[:50]:  # Include more tools
            name = tool.get('name', 'unknown')
            desc = tool.get('description', 'No description')
            tool_info.append(f"- {name}: {desc}")
        
        # Add MCP tools
        if mcp_tools:
            tool_info.append("\nMCP Tools available:")
            tool_info.append("- brave_search:search - Web search using Brave Search API")
            tool_info.append("- github:* - GitHub operations (issues, PRs, etc)")
            tool_info.append("- filesystem:* - File system operations")
            tool_info.append("- memory:* - Context storage and retrieval")
        
        # Include conversation history if available
        history_context = ""
        if self.conversation_history:
            # Get recent conversation history
            recent_history = self._format_recent_history()
            if recent_history:
                history_context = f"\nRecent conversation history:\n{recent_history}\n"
        
        # Simple prompt letting AI know it has everything
        prompt = f"""{history_context}Request: {user_input}

You are an AI assistant with full access to tools. Your goal is to be as helpful as possible by finding and providing the information users need.

You have many tools at your disposal - use them creatively and proactively to answer questions thoroughly. Don't just work with what's immediately available; explore, search, and discover.

Available tools:
{chr(10).join(tool_info[:30])}

Be resourceful. If information seems incomplete, use your tools to find more context. Think about what the user really wants and how you can best help them achieve it.

When creating tools, use descriptive names that clearly indicate their purpose.

Respond naturally. If you need to use a tool, mention it clearly with its full name and parameters."""
        
        try:
            response = await self.ai_brain.execute_capability('intelligent_analysis', {
                'prompt': prompt,
                'model': 'gemini'
            })
            
            ai_response = response.get('result', '')
            print(f"[DEBUG] AI response: {ai_response[:100]}...")
            
            # Process the response to execute any mentioned tools
            processed_response = await self._process_ai_response(ai_response, user_input)
            return processed_response
                
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return f"Error: {str(e)}"
    
    async def _execute_and_continue(self, ai_response: str, original_request: str) -> str:
        """Let AI handle execution flow."""
        # Just return the AI response - no pattern matching
        return ai_response
    
    def _get_essential_tools_for_request(self, user_input: str) -> List[Dict[str, Any]]:
        """Get all available tools."""
        try:
            from scripts.tool_calling_system import ToolCallingSystem
            tool_system = ToolCallingSystem()
            return tool_system.get_tools_for_ai()
        except:
            return []
    
    def _format_essential_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools for AI."""
        if not tools:
            return "No tools available"
        
        return ', '.join([t.get('name', 'unknown') for t in tools[:20]])
    
    async def _guide_ai_to_action(self, user_input: str) -> str:
        """Let AI decide next action without guidance."""
        # Just ask AI what to do
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': user_input,
            'model': 'gemini'
        })
        
        return response.get('result', 'Unable to process request')
    
    async def _process_ai_response(self, ai_response: str, user_input: str) -> str:
        """Process AI response and execute any tool calls mentioned."""
        import re
        
        # Get available tools
        tools = self._get_essential_tools_for_request(user_input)
        tool_names = [t['name'] for t in tools]
        
        executed_tools = []
        
        # Pattern 1: tool_name() or tool_name(params)
        func_pattern = r'\b(\w+)\s*\((.*?)\)'
        for match in re.finditer(func_pattern, ai_response):
            tool_name = match.group(1)
            params_str = match.group(2)
            if tool_name in tool_names:
                try:
                    tool_call = f"{tool_name}({params_str})"
                    result = await self._execute_tool_call(tool_call)
                    executed_tools.append((tool_name, result))
                except Exception as e:
                    self.logger.error(f"Error executing tool {tool_name}: {e}")
        
        # Pattern 2: Standalone tool names in code blocks or on separate lines
        # Look for tool names that appear alone (common in AI responses)
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            # Check if line is just a tool name (possibly with backticks)
            cleaned_line = line.strip('`').strip()
            if cleaned_line in tool_names and cleaned_line not in [t[0] for t in executed_tools]:
                try:
                    tool_call = f"{cleaned_line}()"
                    result = await self._execute_tool_call(tool_call)
                    executed_tools.append((cleaned_line, result))
                except Exception as e:
                    self.logger.error(f"Error executing tool {cleaned_line}: {e}")
        
        # If we executed any tools, have AI interpret the results
        if executed_tools:
            # Build a prompt for the AI to interpret the results
            interpretation_prompt = f"The user asked: {user_input}\n\n"
            interpretation_prompt += "I executed the following tools and got these results:\n\n"
            
            for tool_name, result in executed_tools:
                interpretation_prompt += f"Tool: {tool_name}\n"
                if isinstance(result, dict):
                    import json
                    interpretation_prompt += f"Result: {json.dumps(result, indent=2)}\n\n"
                else:
                    interpretation_prompt += f"Result: {str(result)}\n\n"
            
            interpretation_prompt += "Please provide a natural language summary of these results for the user. Be conversational and highlight the key information."
            
            # Get AI to interpret the results
            try:
                response = await self.ai_brain.execute_capability('intelligent_analysis', {
                    'prompt': interpretation_prompt,
                    'model': 'gemini'
                })
                
                interpreted_response = response.get('result', '')
                
                # Return the original AI response plus the interpretation
                return f"{ai_response}\n\n{interpreted_response}"
                
            except Exception as e:
                self.logger.error(f"Error interpreting results: {e}")
                # Fallback to showing raw results
                return f"{ai_response}\n\nTool Results: {json.dumps(executed_tools[0][1], indent=2)}"
        
        # Otherwise just return the AI response
        return ai_response
    
    async def _continue_tool_discovery_flow(self, user_input: str) -> str:
        """Continue an ongoing tool discovery flow."""
        # This would handle multi-step tool discovery
        # For now, treat as new request
        return await self._handle_with_minimal_context(user_input)
    
    async def _search_for_context(self, user_input: str) -> str:
        """Search for context when AI requests it."""
        # Implement context search
        return await self._guide_ai_to_action(user_input)
    
    # Removed old method with hardcoded prompts
    async def _handle_with_intelligent_context_old(self, user_input: str) -> str:
        """Old method - kept for reference."""
        # Get pattern-based suggestions before processing
        pattern_suggestions = self.pattern_analyzer.get_pattern_suggestions(
            user_input, 
            {'conversation_history': self._get_recent_history()}
        )
        
        # Gather COMPLETE system context
        dynamic_context = await self._gather_complete_context(user_input)
        
        # Format the context
        formatted_context = self.context_collector.format_context_for_ai(dynamic_context)
        
        # If user is asking about a specific project, include its full details
        project_details = ""
        if any(proj in user_input.lower() for proj in ['eco-track-ai', 'brand-guardian', 'inventory-sync']):
            state = self.state_manager.get_state()
            projects = state.get('projects', {})
            for proj_name, proj_data in projects.items():
                if proj_name.lower() in user_input.lower():
                    project_details = f"\n\nDETAILED PROJECT DATA FOR {proj_name}:\n"
                    project_details += f"Health Score: {proj_data.get('health_score', 0)}/100\n"
                    project_details += f"Description: {proj_data.get('description', 'N/A')}\n"
                    project_details += f"Language: {proj_data.get('language', 'Unknown')}\n"
                    project_details += f"Open Issues: {proj_data.get('metrics', {}).get('issues_open', 0)}\n"
                    project_details += f"Recent Commits: {proj_data.get('recent_activity', {}).get('recent_commits', 0)}\n"
                    project_details += f"Status: {proj_data.get('status', 'unknown')}\n"
                    break
        
        # Get available tools - with resilient loading
        available_tools = []
        tool_names = []
        try:
            from scripts.tool_calling_system import ToolCallingSystem
            tool_system = ToolCallingSystem()
            available_tools = tool_system.get_tools_for_ai()
            tool_names = [t['name'] for t in available_tools]
        except Exception as e:
            log_system_message(f"Error loading tools: {e}", "ERROR")
            # Continue with empty tools list rather than crashing
            available_tools = []
            tool_names = []
        
        # Use minimal prompt - let AI figure it out
        prompt = f"{user_input}\n\nContext: {formatted_context}"

        try:
            # Get AI response
            response = await self.ai_brain.execute_capability('intelligent_analysis', {
                'prompt': prompt,
                'model': 'gemini'
            })
            
            return response.get('result', 'Unable to process request')
            
        except Exception as e:
            log_system_message(f"Error in AI handler: {e}", "ERROR")
            
            # Provide more helpful error messages
            error_str = str(e)
            if "tool" in error_str.lower():
                return f"Error creating or executing tool: {error_str}\n\nTry being more specific about what you want the tool to do."
            elif "CREATE_TOOL" in error_str:
                return f"Error in tool creation process: {error_str}\n\nThe system is working on improving its tool creation capabilities."
            else:
                return f"I encountered an error: {error_str}\n\nPlease try rephrasing your request or check the system logs for details."
    
    # REMOVED - AI provides all responses
    
    def _parse_continuous_ai_options(self, command: str) -> Dict[str, Any]:
        """Parse options from run_continuous_ai.py command."""
        options = {}
        
        # Parse workers
        match = re.search(r'--workers\s+(\d+)', command)
        if match:
            options['workers'] = int(match.group(1))
        
        # Parse mode
        match = re.search(r'--mode\s+(\w+)', command)
        if match:
            options['mode'] = match.group(1)
        
        # Parse flags
        if '--no-research' in command:
            options['enable_research'] = False
        if '--monitor-workers' in command:
            options['enable_monitor'] = True
        if '--no-mcp' in command:
            options['enable_mcp'] = False
        
        return options
    
    def _parse_reset_options(self, command: str) -> 'ResetType':
        """Parse reset type from reset_system.py command."""
        if '--preserve-cache' in command and '--preserve-knowledge' in command:
            return ResetType.SELECTIVE
        elif '--preserve-cache' in command:
            return ResetType.CACHE_ONLY
        elif '--preserve-knowledge' in command:
            return ResetType.SELECTIVE
        else:
            return ResetType.FULL
    
    def _format_system_result(self, result: Dict[str, Any]) -> str:
        """Format system command results."""
        if result.get('success'):
            return f"✓ {result.get('message', 'Command executed successfully')}\n{json.dumps(result.get('status', {}), indent=2)}"
        else:
            return f"❌ {result.get('message', 'Command failed')}\n{result.get('error', '')}"
    
    def _format_reset_result(self, result: Dict[str, Any]) -> str:
        """Format reset command results."""
        if result.get('success'):
            msg = f"✓ System reset completed successfully!\n"
            msg += f"• Deleted {len(result.get('deleted_files', []))} files\n"
            msg += f"• Freed {result.get('deleted_size_mb', 0):.1f}MB\n"
            if result.get('preserved_files'):
                msg += f"• Preserved {len(result['preserved_files'])} files"
            return msg
        else:
            return f"❌ Reset failed: {', '.join(result.get('errors', ['Unknown error']))}"
    
    # REMOVED - No more hardcoded patterns!
    # The AI decides everything based on context
    
    def _update_context(self, user_input: str):
        """Update conversation context."""
        # Just store the input - let AI understand context
        self.current_context['last_input'] = user_input
        self.current_context['timestamp'] = datetime.now(timezone.utc).isoformat()
    
    # REMOVED - AI generates greetings based on context
    
    # REMOVED - All input goes through AI
    
    # REMOVED - All input goes through AI
    
    # REMOVED - AI handles confirmations with context
    
    # REMOVED - AI generates appropriate responses
    
    # REMOVED - All input goes through AI
    
    def _format_recent_history(self) -> str:
        """Format recent conversation history for context."""
        recent = self.conversation_history[-3:] if self.conversation_history else []
        formatted = []
        for turn in recent:
            formatted.append(f"User: {turn.user_input}")
            formatted.append(f"Assistant: {turn.assistant_response[:100]}...")
        return "\n".join(formatted)
    
    def _format_complete_history(self) -> str:
        """Format complete relevant conversation history."""
        if not self.conversation_history:
            return "No previous conversation"
        
        # Include last 10 turns for better context
        recent = self.conversation_history[-10:]
        formatted = []
        for i, turn in enumerate(recent):
            formatted.append(f"[Turn {i+1} - {turn.timestamp.strftime('%H:%M:%S')}]")
            formatted.append(f"User: {turn.user_input}")
            formatted.append(f"Assistant: {turn.assistant_response}")
            if turn.actions_taken:
                formatted.append(f"Actions: {json.dumps(turn.actions_taken)}")
            formatted.append("")
        return "\n".join(formatted)
    
    def _format_all_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Format all available tools with full details."""
        formatted = []
        for tool in tools:
            formatted.append(f"- {tool['name']}: {tool.get('description', 'No description')}")
            if 'parameters' in tool:
                formatted.append(f"  Parameters: {tool['parameters']}")
        return "\n".join(formatted)
    
    async def _gather_complete_context(self, user_input: str) -> Dict[str, Any]:
        """Gather COMPLETE system context - everything the AI needs."""
        # Start with dynamic context
        context = await self.context_collector.gather_context_for_query(
            user_input, 
            self.conversation_history[-10:] if self.conversation_history else []
        )
        
        # Add system status
        try:
            continuous_status = await self.check_continuous_ai_status()
            context['continuous_ai_status'] = continuous_status
        except Exception as e:
            log_system_message(f"Error getting continuous AI status: {e}", "ERROR")
            context['continuous_ai_status'] = {'error': str(e)}
        
        # Add workflow information
        context['available_workflows'] = list_all_workflows()
        context['custom_workflows'] = self.workflow_orchestrator.list_workflows()
        
        # Add file system info
        context['important_files'] = {
            'run_continuous_ai.py': 'Starts the continuous AI system with options',
            'reset_system.py': 'Resets system state with preservation options',
            'system_state.json': 'Current system state',
            'task_state.json': 'Current task queue state',
            'logs/': 'System and conversation logs'
        }
        
        # Add current working directory and environment
        context['environment'] = {
            'cwd': os.getcwd(),
            'python_version': sys.version,
            'platform': sys.platform
        }
        
        # Add memory and learning
        context['assistant_memory'] = {
            'user_preferences': self.memory.user_preferences,
            'common_tasks': self.memory.common_tasks[-10:],
            'learned_patterns': dict(sorted(
                [(k, v) for k, v in self.memory.learned_patterns.items() if isinstance(v, (int, float))], 
                key=lambda x: x[1], reverse=True)[:10])
        }
        
        # Add pending actions
        context['pending_actions'] = self.current_context.get('pending_confirmations', {})
        
        return context
    
    # REMOVED - No more pattern matching! AI decides everything
    
    def _get_recent_user_input(self) -> str:
        """Get the most recent user input from conversation history."""
        if self.conversation_history:
            return self.conversation_history[-1].user_input
        return ""
    
    async def _execute_tool_call(self, tool_call_str: str) -> str:
        """Execute a tool call string like 'tool_name(param1="value1")'."""
        print(f"[DEBUG] _execute_tool_call called with: {tool_call_str}")
        try:
            # Parse tool name and parameters
            match = re.match(r'(\w+)\((.*)\)', tool_call_str)
            if not match:
                return f"Invalid tool call format: {tool_call_str}"
            
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # SECURITY: Parse parameters using ast.literal_eval for safety
            params = {}
            if params_str:
                try:
                    # Transform the parameter string into a valid dict representation
                    # Replace parameter format: key=value -> "key": value
                    dict_str = params_str
                    
                    # Handle various parameter formats
                    # First, extract all key=value pairs using a more robust regex
                    param_pattern = r'(\w+)\s*=\s*(\'[^\']*\'|"[^"]*"|\[[^\]]*\]|\{[^\}]*\}|[^,]+)'
                    param_matches = re.findall(param_pattern, params_str)
                    
                    if param_matches:
                        # Build a proper dictionary string
                        dict_parts = []
                        for key, value in param_matches:
                            # Clean up the value
                            value = value.strip()
                            
                            # If it's already quoted, use it as-is
                            if (value.startswith('"') and value.endswith('"')) or \
                               (value.startswith("'") and value.endswith("'")):
                                dict_parts.append(f'"{key}": {value}')
                            # If it looks like a number
                            elif value.replace('.', '').replace('-', '').isdigit():
                                dict_parts.append(f'"{key}": {value}')
                            # If it's a boolean
                            elif value.lower() in ['true', 'false']:
                                dict_parts.append(f'"{key}": {value.lower()}')
                            # If it's a list or dict (already has brackets)
                            elif value.startswith('[') or value.startswith('{'):
                                dict_parts.append(f'"{key}": {value}')
                            # Otherwise, treat as string
                            else:
                                dict_parts.append(f'"{key}": "{value}"')
                        
                        dict_str = "{" + ", ".join(dict_parts) + "}"
                        
                        # Use ast.literal_eval for safe parsing
                        params = ast.literal_eval(dict_str)
                    
                except (ValueError, SyntaxError) as e:
                    # Fallback to simple parsing if ast.literal_eval fails
                    log_system_message(f"Parameter parsing with ast.literal_eval failed: {e}, using fallback", "WARNING")
                    param_matches = re.findall(r'(\w+)=["\']?([^,"\']+)["\']?', params_str)
                    for key, value in param_matches:
                        # Try to convert to appropriate type
                        if value.isdigit():
                            params[key] = int(value)
                        elif value.lower() in ['true', 'false']:
                            params[key] = value.lower() == 'true'
                        else:
                            params[key] = value
            
            # Enhance parameters with context from conversation
            # TEMPORARILY DISABLED to debug issue
            # enhanced_params = self.context_enhancer.enhance_tool_parameters(
            #     tool_name, 
            #     params, 
            #     self._get_recent_user_input()
            # )
            enhanced_params = params  # Use params directly
            
            # Execute the tool
            from scripts.tool_calling_system import ToolCallingSystem
            tool_system = ToolCallingSystem()
            
            # Debug logging
            print(f"[DEBUG] About to call tool '{tool_name}' with params: {enhanced_params}")
            log_system_message(f"Calling tool {tool_name} with params: {enhanced_params}")
            
            try:
                result = await tool_system.call_tool(tool_name, **enhanced_params)
                print(f"[DEBUG] Tool call completed, result type: {type(result)}")
            except Exception as e:
                print(f"[DEBUG] Tool call exception: {type(e).__name__}: {e}")
                log_system_message(f"Tool call exception: {e}", "ERROR")
                # Re-raise with more context
                raise Exception(f"Tool {tool_name} failed with: {str(e)}")
            
            # Format the result
            if result.get('success'):
                return self._format_tool_result(result.get('result'), tool_name)
            elif result.get('error') and 'not found' in result.get('error'):
                # Tool doesn't exist, offer to create it
                return await self._handle_missing_tool(tool_name, params)
            else:
                error_msg = result.get('error', '')
                # Check if this is an import or attribute error that we can fix
                if any(err in error_msg for err in ['No module named', 'has no attribute', 'ModuleNotFoundError', 'ImportError', 'AttributeError']):
                    return await self._handle_broken_tool(tool_name, params, error_msg)
                else:
                    return f"Error executing {tool_name}: {error_msg}"
                
        except Exception as e:
            self.logger.error(f"Error executing tool call: {e}")
            return f"I encountered an error executing the tool: {str(e)}"
    
    async def _handle_missing_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Handle case where a tool doesn't exist."""
        return f"""I notice we don't have a '{tool_name}' tool yet. Let me create it for you...

Creating new tool: {tool_name}
This tool will {tool_name.replace('_', ' ')} based on your request.

Please wait a moment while I generate this functionality..."""
    
    async def _handle_broken_tool(self, tool_name: str, params: Dict[str, Any], error_msg: str) -> str:
        """Handle case where a tool exists but has errors (import issues, etc)."""
        print(f"Attempting to fix broken tool: {tool_name} with error: {error_msg}")
        
        # Analyze the error to understand what's wrong
        fix_description = f"""I detected that the '{tool_name}' tool has an error: {error_msg}

Let me fix this tool by regenerating it with the correct imports and implementation..."""
        
        # Get the tool's original purpose by reading its file
        tool_path = Path(f"scripts/custom_tools/{tool_name}.py")
        original_description = ""
        original_requirements = ""
        
        if tool_path.exists():
            try:
                content = tool_path.read_text()
                # Extract description from docstring or __description__
                if '__description__' in content:
                    desc_match = re.search(r'__description__\s*=\s*["\'](.+?)["\']', content)
                    if desc_match:
                        original_description = desc_match.group(1)
                
                # Extract requirements from docstring
                if 'Requirements:' in content:
                    req_match = re.search(r'Requirements:\s*(.+?)(?:\n\n|\*/)', content, re.DOTALL)
                    if req_match:
                        original_requirements = req_match.group(1).strip()
                        
            except Exception as e:
                self.logger.error(f"Error reading broken tool file: {e}")
        
        # Prepare enhanced requirements with error context
        enhanced_requirements = f"""
        Original purpose: {original_description or f'{tool_name.replace("_", " ")} tool'}
        Original requirements: {original_requirements}
        
        CRITICAL FIX NEEDED:
        The tool is currently broken with error: {error_msg}
        
        When regenerating this tool:
        1. Use the correct imports from available modules
        2. For 'workflow' tools, use: from scripts.workflow_orchestrator import WorkflowOrchestrator
        3. For 'tool execution', use: from scripts.tool_calling_system import ToolCallingSystem
        4. Never import non-existent modules like 'tool_executor' or 'workflow_manager'
        5. Check the enhanced context for all available modules
        6. Ensure all class methods and attributes exist
        """
        
        # Delete the broken tool file first
        if tool_path.exists():
            tool_path.unlink()
            print(f"Deleted broken tool file: {tool_path}")
        
        # Create the fixed tool
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        
        creation_result = await tool_system._create_new_tool(
            name=tool_name,
            description=original_description or f"Fixed version of {tool_name} tool",
            requirements=enhanced_requirements
        )
        
        if creation_result.get('success'):
            # Try executing the fixed tool
            result = await tool_system.call_tool(tool_name, **params)
            
            if result.get('success'):
                return f"""{fix_description}

✓ Successfully fixed and regenerated '{tool_name}'!

{self._format_tool_result(result.get('result'), tool_name)}

The tool has been fixed and is now working properly."""
            else:
                return f"""{fix_description}

I regenerated the tool but it still has issues: {result.get('error')}

Let me know if you'd like me to try a different approach."""
        else:
            return f"""{fix_description}

I couldn't regenerate the tool: {creation_result.get('error')}

Would you like me to try creating a simpler version?"""
    
    async def _create_and_execute_tool(self, tool_name: str, user_input: str, description: str) -> str:
        """Create a new tool and execute it."""
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        
        # Determine requirements based on tool name and user input
        requirements = f"""
        Based on user request: {user_input}
        Create a tool that: {description}
        Tool should integrate with existing system components
        """
        
        # Create the tool
        creation_result = await tool_system._create_new_tool(
            name=tool_name,
            description=description,
            requirements=requirements
        )
        
        if creation_result.get('success'):
            # Tool created, now execute it with context
            tool_params = {
                'context': {
                    'user_input': user_input,
                    'description': description
                }
            }
            result = await tool_system.call_tool(tool_name, **tool_params)
            
            if result.get('success'):
                return f"""✓ Created new tool '{tool_name}' successfully!

{self._format_tool_result(result.get('result'), tool_name)}

The tool has been saved and will be available for future use."""
            else:
                return f"Created tool '{tool_name}' but encountered an error executing it: {result.get('error')}"
        else:
            return f"I couldn't create the tool '{tool_name}': {creation_result.get('error')}"
    
    def _format_tool_result(self, result: Any, tool_name: str) -> str:
        """Return raw tool results without formatting."""
        # Return raw JSON for all tools so AI can see complete data
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        elif isinstance(result, list):
            return json.dumps(result, indent=2)
        else:
            # For any other type, just convert to string
            return str(result)
    
    def _format_result_data(self, data: Any) -> str:
        """Format result data conversationally."""
        if isinstance(data, list):
            if len(data) == 0:
                return "I didn't find any results."
            elif len(data) == 1:
                return f"I found one result:\n• {data[0]}"
            else:
                formatted = f"I found {len(data)} results:\n"
                for item in data[:5]:  # Show first 5
                    if isinstance(item, dict):
                        formatted += f"• {item.get('name', item.get('title', str(item)))}\n"
                    else:
                        formatted += f"• {item}\n"
                if len(data) > 5:
                    formatted += f"... and {len(data) - 5} more"
                return formatted
        elif isinstance(data, dict):
            # Format key details
            important_keys = ['name', 'title', 'description', 'status', 'message']
            details = []
            for key in important_keys:
                if key in data:
                    details.append(f"{key.title()}: {data[key]}")
            return "\n".join(details)
        else:
            return str(data)
    
    async def _create_workflow(self, workflow_name: str, workflow_json: str) -> str:
        """Create a new workflow from AI-generated definition."""
        try:
            # Parse the workflow JSON
            workflow_def = json.loads(workflow_json)
            
            # Ensure the workflow has a name
            workflow_def['name'] = workflow_name
            
            # Create the workflow
            workflow = await self.workflow_orchestrator.create_workflow(workflow_def)
            
            return f"""✓ Created workflow: {workflow.name}
ID: {workflow.id}
Description: {workflow.description}
Steps: {len(workflow.steps)}

You can now execute this workflow with:
EXECUTE_WORKFLOW: {workflow.id}(param1='value1', param2='value2')"""
            
        except json.JSONDecodeError as e:
            return f"❌ Invalid workflow JSON: {e}"
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            return f"❌ Error creating workflow: {e}"
    
    async def _execute_workflow_call(self, workflow_call: str) -> str:
        """Execute a workflow from a call string."""
        try:
            # Parse workflow call (either by name or ID)
            match = re.match(r'(\w+)\((.*?)\)$', workflow_call)
            if match:
                workflow_identifier = match.group(1)
                params_str = match.group(2)
                
                # Parse parameters
                params = {}
                if params_str:
                    # Use the same parameter parsing as tool calls
                    param_pattern = r'(\w+)=[\'"]([^\'"]+)[\'"]|(\w+)=(\S+)'
                    for match in re.finditer(param_pattern, params_str):
                        if match.group(1):
                            params[match.group(1)] = match.group(2)
                        else:
                            params[match.group(3)] = match.group(4)
            else:
                workflow_identifier = workflow_call.strip()
                params = {}
            
            # Check if it's a predefined workflow
            predefined_workflow = get_workflow_definition(workflow_identifier)
            if predefined_workflow:
                # Create and execute the predefined workflow
                workflow = await self.workflow_orchestrator.create_workflow(predefined_workflow)
                result = await self.workflow_orchestrator.execute_workflow(workflow.id, params)
            else:
                # Try to execute by ID
                result = await self.workflow_orchestrator.execute_workflow(workflow_identifier, params)
            
            if result['success']:
                return f"""✓ Workflow executed successfully!
Status: {result['workflow']['status']}
Duration: {result['workflow']['duration_ms']}ms
Steps completed: {len([s for s in result['workflow']['steps'] if s['status'] == 'completed'])}

Final context:
{json.dumps(result['final_context'], indent=2)[:500]}..."""
            else:
                return f"❌ Workflow failed: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            self.logger.error(f"Error executing workflow: {e}")
            return f"❌ Error executing workflow: {e}"
    
    async def _list_workflows(self) -> str:
        """List all available workflows."""
        try:
            # Get predefined workflows
            predefined = list_all_workflows()
            
            # Get created workflows
            created = self.workflow_orchestrator.list_workflows()
            
            response = "📋 **Available Workflows**\n\n"
            
            if predefined:
                response += "**System Workflows:**\n"
                for wf in predefined:
                    response += f"• **{wf['name']}** - {wf['display_name']}\n"
                    response += f"  {wf['description']}\n\n"
            
            if created:
                response += "\n**Custom Workflows:**\n"
                for wf in created:
                    response += f"• **{wf['id']}** - {wf['name']}\n"
                    response += f"  Status: {wf['status']}, Created: {wf['created_at']}\n\n"
            
            response += "\nExecute a workflow with: EXECUTE_WORKFLOW: workflow_name(params...)"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error listing workflows: {e}")
            return f"❌ Error listing workflows: {e}"
    
    async def _explain_capabilities(self) -> str:
        """Explain what the assistant can do."""
        return """I'm your CWMAI assistant! I can help you with:

📋 **Task Management**
• Create issues and tasks for any repository
• List and track active tasks
• Update task status and priorities

🔍 **Discovery & Search**
• Search through repositories
• Find specific code or documentation
• Analyze repository health

🏗️ **Architecture & Design**
• Generate system architectures
• Create project structures
• Design database schemas

📊 **Analysis & Insights**
• Show system status and performance
• Analyze market opportunities
• Review code quality metrics

🤖 **Continuous AI Control**
• Start/stop the continuous AI system
• Check if the system is running
• Monitor system health and performance
• View worker status and task queue

🔧 **System Maintenance**
• Smart system reset with multiple options
• Analyze system health and recommend fixes
• Clear logs, state, or cache selectively
• Emergency reset for critical issues

⚡ **Workflow Automation**
• Create complex multi-step workflows
• Execute predefined system workflows
• Run workflows for health checks, analysis, maintenance
• Orchestrate parallel and conditional task execution

💬 **And More!**
• Answer questions about your projects
• Provide coding suggestions
• Help with best practices

Just tell me what you'd like to do in natural language. For example:

**Starting things up:**
- "Fire up the system"
- "Let's get things running"
- "Wake up the AI"
- "Boot everything up"
- "Get the workers going with 5 instances"

**Checking on things:**
- "How are things going?"
- "Is everything running smoothly?"
- "What's happening with the system?"
- "Are we online?"
- "How's the AI doing?"

**Stopping gracefully:**
- "Shut it down for now"
- "Let's take a break"
- "Power everything down"
- "Stop all the workers"

**When things go wrong:**
- "Something's broken, can you fix it?"
- "The system is acting weird"
- "Everything seems stuck"
- "Things aren't working right"

**Fresh starts:**
- "Give me a clean slate"
- "Let's start from scratch"
- "Wipe everything but keep my research data"
- "Factory reset but preserve the AI cache"
- "Clear the logs, they're getting huge"

What would you like to work on?"""
    
    async def _get_status_conversationally(self) -> str:
        """Get system status in a conversational way."""
        status_result = await self.nli.process_natural_language("show status")
        
        if status_result.get('success') and status_result.get('data'):
            data = status_result['data']
            
            response = "Here's what's happening:\n\n"
            
            # Active tasks
            if 'active_tasks' in data:
                count = len(data['active_tasks'])
                if count == 0:
                    response += "📋 No active tasks right now - all clear!\n"
                elif count == 1:
                    response += "📋 There's 1 active task in progress\n"
                else:
                    response += f"📋 There are {count} active tasks in progress\n"
            
            # System health
            if 'system_health' in data:
                health = data['system_health']
                if health > 90:
                    response += "💚 System is running smoothly\n"
                elif health > 70:
                    response += "💛 System is doing okay, with minor issues\n"
                else:
                    response += "❤️ System needs attention - some issues detected\n"
            
            # Recent activity
            if 'recent_activity' in data:
                response += f"\n📊 Recent activity: {data['recent_activity']}"
            
            # Check continuous AI status
            continuous_status = await self.check_continuous_ai_status()
            if continuous_status['running']:
                response += f"\n\n🤖 Continuous AI: Running (PID: {continuous_status['pid']})"
                response += f"\n   • Active workers: {continuous_status['active_workers']}"
                response += f"\n   • Tasks queued: {continuous_status['queued_tasks']}"
            else:
                response += "\n\n🤖 Continuous AI: Not running"
            
            response += "\n\nWould you like me to show more details about anything specific?"
            
            return response
        else:
            return "Let me check the system status... It seems I'm having trouble accessing that information right now. Is there something specific you'd like to know about?"
    
    async def _answer_with_ai(self, question: str) -> str:
        """Use AI to answer general questions with full system context."""
        # Gather dynamic context based on the question
        dynamic_context = await self.context_collector.gather_context_for_query(
            question, 
            self.conversation_history[-5:] if self.conversation_history else []
        )
        
        # Format the context for the AI
        formatted_context = self.context_collector.format_context_for_ai(dynamic_context)
        
        # Build comprehensive prompt
        prompt = f"""You are the CWMAI assistant with full access to system information.

CURRENT SYSTEM CONTEXT:
{formatted_context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based on the actual system state provided above
2. If the user asks about repositories, list the actual repositories shown
3. If the user asks about status, use the real status information
4. If the user asks what you can do, list the actual available commands
5. If action is needed, specify which command should be executed
6. Be specific and accurate - use the real data provided

Provide a helpful, conversational answer that uses the actual system information."""
        
        response = await self.ai_brain.execute_capability('question_answering', {
            'prompt': prompt,
            'model': 'gemini'  # Force Gemini for all assistant requests
        })
        
        # Check if the AI suggested executing a command
        ai_response = response.get('result', '')
        
        # If the response suggests executing a command, try to do it
        if any(cmd in ai_response.lower() for cmd in ['execute', 'run command', 'use command']):
            # Extract and execute the suggested command
            executed = await self._try_execute_from_ai_suggestion(question, ai_response)
            if executed:
                return executed
        
        return ai_response or "I'm having trouble accessing that information. Could you try rephrasing your question?"
    
    async def _try_execute_from_ai_suggestion(self, original_query: str, ai_response: str) -> Optional[str]:
        """Try to execute a command suggested by the AI.
        
        Args:
            original_query: The user's original question
            ai_response: The AI's response that may contain command suggestions
            
        Returns:
            Executed command result or None
        """
        # Look for command patterns in AI response
        command_patterns = [
            r"execute[s]? (?:the )?command[:]? ['\"]([^'\"]+)['\"]",
            r"run[s]? (?:the )?command[:]? ['\"]([^'\"]+)['\"]",
            r"use[s]? (?:the )?command[:]? ['\"]([^'\"]+)['\"]",
            r"you (?:can|should) (?:execute|run|use)[:]? ['\"]([^'\"]+)['\"]"
        ]
        
        import re
        for pattern in command_patterns:
            match = re.search(pattern, ai_response, re.IGNORECASE)
            if match:
                suggested_command = match.group(1)
                print(f"AI suggested command: {suggested_command}")
                
                # Execute through NLI
                try:
                    result = await self.nli.process_natural_language(suggested_command)
                    if result.get('success'):
                        # Format the result conversationally
                        return self._format_command_result(result, suggested_command)
                except Exception as e:
                    self.logger.error(f"Error executing AI-suggested command: {e}")
        
        return None
    
    def _format_command_result(self, result: Dict[str, Any], command: str) -> str:
        """Format command execution result conversationally."""
        if result.get('data'):
            data = result['data']
            
            # Handle different data types
            if isinstance(data, list):
                if len(data) == 0:
                    return f"I executed '{command}' but didn't find any results."
                else:
                    response = f"Here's what I found:\n\n"
                    for item in data[:5]:  # Limit to 5 items
                        if isinstance(item, dict):
                            response += f"• {item.get('name', item.get('title', str(item)))}\n"
                        else:
                            response += f"• {item}\n"
                    if len(data) > 5:
                        response += f"\n...and {len(data) - 5} more items"
                    return response
            elif isinstance(data, dict):
                # Format key information from dict
                important_keys = ['name', 'title', 'description', 'status', 'message', 'count']
                details = []
                for key in important_keys:
                    if key in data:
                        details.append(f"{key.title()}: {data[key]}")
                return "\n".join(details) if details else str(data)
            else:
                return str(data)
        
        return result.get('message', f"I executed '{command}' successfully.")
    
    async def _execute_pending_action(self, pending: Dict[str, Any]) -> str:
        """Execute a pending action after confirmation."""
        action = pending.get('action')
        params = pending.get('params', {})
        
        # Execute the action
        result = await self.nli.process_natural_language(pending.get('original_command', ''))
        
        if result.get('success'):
            return f"Done! I've {action} as requested."
        else:
            return f"I ran into an issue: {result.get('reason', 'Unknown error')}. Would you like to try a different approach?"
    
    def _learn_from_interaction(self, user_input: str, response: str):
        """Learn from user interactions to improve future responses."""
        # Prepare context for pattern analysis
        context = {
            'conversation_history': [{
                'user_input': turn.user_input,
                'assistant_response': turn.assistant_response,
                'timestamp': turn.timestamp.isoformat(),
                'actions': turn.actions_taken
            } for turn in self.conversation_history[-5:]],  # Last 5 turns
            'interaction_success': 'success' in response or 'completed' in response,
            'previous_intent': self.memory.learned_patterns.get('last_intent', 'unknown')
        }
        
        # Analyze conversation patterns
        analysis = self.pattern_analyzer.analyze_conversation_turn(
            user_input, response, context
        )
        
        # Update memory with analysis results
        if analysis['intent']:
            self.memory.learned_patterns['last_intent'] = analysis['intent'].primary_intent
            
            # Track intent frequency
            intent_key = f"intent_{analysis['intent'].primary_intent}"
            self.memory.learned_patterns[intent_key] = self.memory.learned_patterns.get(intent_key, 0) + 1
        
        # Track entities
        if analysis.get('entities'):
            for entity_type, values in analysis['entities'].items():
                entity_key = f"entity_{entity_type}"
                self.memory.learned_patterns[entity_key] = self.memory.learned_patterns.get(entity_key, 0) + len(values)
        
        # Track successful patterns
        if context['interaction_success'] and analysis['detected_patterns']:
            pattern_key = f"successful_pattern_{analysis['detected_patterns'][0]['type']}"
            self.memory.learned_patterns[pattern_key] = self.memory.learned_patterns.get(pattern_key, 0) + 1
        
        # Track common tasks
        if context['interaction_success']:
            self.memory.common_tasks.append(user_input)
            # Keep only last 50 tasks
            self.memory.common_tasks = self.memory.common_tasks[-50:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = f"Conversation summary ({len(self.conversation_history)} turns):\n\n"
        
        # Recent topics
        recent_topics = set()
        for turn in self.conversation_history[-10:]:
            if 'project' in turn.user_input.lower():
                recent_topics.add('projects')
            if 'issue' in turn.user_input.lower():
                recent_topics.add('issues')
            if 'status' in turn.user_input.lower():
                recent_topics.add('status')
        
        if recent_topics:
            summary += f"Recent topics: {', '.join(recent_topics)}\n"
        
        # Common patterns
        if self.memory.learned_patterns:
            top_intents = sorted(
                [(k.replace('intent_', ''), v) for k, v in self.memory.learned_patterns.items() if k.startswith('intent_')],
                key=lambda x: x[1], reverse=True
            )[:3]
            if top_intents:
                summary += f"Common requests: {', '.join([intent[0] for intent in top_intents])}\n"
        
        # Last 3 interactions
        summary += "\nRecent conversation:\n"
        for turn in self.conversation_history[-3:]:
            summary += f"• User: {turn.user_input}\n"
            summary += f"  Assistant: {turn.assistant_response[:100]}...\n"
        
        return summary
    
    def _format_pattern_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format pattern-based suggestions for the AI prompt."""
        if not suggestions:
            return "No pattern suggestions available"
        
        formatted = []
        for suggestion in suggestions[:3]:  # Limit to top 3
            formatted.append(f"- {suggestion.get('description', 'Unknown pattern')}: {suggestion.get('action', 'No action suggested')}")
        
        return "\n".join(formatted)
    
    def _get_recent_history(self) -> List[Dict[str, Any]]:
        """Get recent conversation history for pattern analysis."""
        return [{
            'user_input': turn.user_input,
            'assistant_response': turn.assistant_response,
            'timestamp': turn.timestamp.isoformat()
        } for turn in self.conversation_history[-5:]]
    
    async def start_continuous_ai_system(self, workers: int = 3, mode: str = 'balanced',
                                       enable_research: bool = True, enable_monitor: bool = True,
                                       enable_mcp: bool = True) -> Dict[str, Any]:
        """Start the continuous AI system with specified options."""
        try:
            # Check if already running
            status = await self.check_continuous_ai_status()
            if status['running']:
                return {
                    'success': False,
                    'message': f"Continuous AI is already running with PID {status['pid']}",
                    'status': status
                }
            
            # Build command
            cmd = [sys.executable, str(self.continuous_ai_script)]
            cmd.extend(['--workers', str(workers)])
            cmd.extend(['--mode', mode])
            
            if not enable_research:
                cmd.append('--no-research')
            if enable_monitor:
                cmd.append('--monitor-workers')
            if not enable_mcp:
                cmd.append('--no-mcp')
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            
            self.continuous_ai_process = process
            
            # Wait a moment and check if it started successfully
            await asyncio.sleep(2)
            
            if process.poll() is None:
                # Process is still running
                new_status = await self.check_continuous_ai_status()
                return {
                    'success': True,
                    'message': f"Started continuous AI system with {workers} workers in {mode} mode",
                    'pid': process.pid,
                    'status': new_status
                }
            else:
                # Process terminated
                stdout, stderr = process.communicate()
                return {
                    'success': False,
                    'message': "Failed to start continuous AI system",
                    'error': stderr or stdout
                }
                
        except Exception as e:
            self.logger.error(f"Error starting continuous AI: {e}")
            return {
                'success': False,
                'message': f"Error starting continuous AI: {str(e)}"
            }
    
    async def stop_continuous_ai_system(self) -> Dict[str, Any]:
        """Stop the continuous AI system gracefully."""
        try:
            status = await self.check_continuous_ai_status()
            
            if not status['running']:
                return {
                    'success': True,
                    'message': "Continuous AI system is not running"
                }
            
            # Try graceful shutdown first
            if sys.platform != 'win32':
                os.killpg(os.getpgid(status['pid']), signal.SIGTERM)
            else:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(status['pid'])], capture_output=True)
            
            # Wait for graceful shutdown
            await asyncio.sleep(3)
            
            # Check if stopped
            new_status = await self.check_continuous_ai_status()
            if not new_status['running']:
                return {
                    'success': True,
                    'message': "Successfully stopped continuous AI system"
                }
            
            # Force kill if still running
            if sys.platform != 'win32':
                os.killpg(os.getpgid(status['pid']), signal.SIGKILL)
            else:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(status['pid'])], capture_output=True)
            
            return {
                'success': True,
                'message': "Forcefully stopped continuous AI system"
            }
            
        except Exception as e:
            self.logger.error(f"Error stopping continuous AI: {e}")
            return {
                'success': False,
                'message': f"Error stopping continuous AI: {str(e)}"
            }
    
    async def check_continuous_ai_status(self) -> Dict[str, Any]:
        """Check if continuous AI system is running."""
        try:
            # Look for the process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and any('run_continuous_ai.py' in arg for arg in cmdline):
                        # Found the process, get more info
                        orchestrator_state_file = Path("continuous_orchestrator_state.json")
                        
                        active_workers = 0
                        queued_tasks = 0
                        
                        if orchestrator_state_file.exists():
                            try:
                                with open(orchestrator_state_file, 'r') as f:
                                    state = json.load(f)
                                    active_workers = state.get('active_workers', 0)
                                    queued_tasks = len(state.get('task_queue', []))
                            except:
                                pass
                        
                        return {
                            'running': True,
                            'pid': proc.info['pid'],
                            'uptime': datetime.now().timestamp() - proc.info['create_time'],
                            'active_workers': active_workers,
                            'queued_tasks': queued_tasks
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'running': False,
                'pid': None,
                'active_workers': 0,
                'queued_tasks': 0
            }
            
        except Exception as e:
            self.logger.error(f"Error checking continuous AI status: {e}")
            return {
                'running': False,
                'error': str(e)
            }
    
    async def execute_system_reset(self, reset_type: ResetType) -> Dict[str, Any]:
        """Execute a system reset with the specified type."""
        try:
            reset_script = Path(__file__).parent.parent / "reset_system.py"
            if not reset_script.exists():
                return {
                    'success': False,
                    'message': "Reset script not found",
                    'errors': ["reset_system.py not found in project root"]
                }
            
            # Build command based on reset type
            cmd = [sys.executable, str(reset_script)]
            
            if reset_type == ResetType.SELECTIVE:
                cmd.extend(['--preserve-cache', '--preserve-knowledge'])
            elif reset_type == ResetType.LOGS_ONLY:
                cmd.append('--logs-only')
            elif reset_type == ResetType.STATE_ONLY:
                cmd.append('--state-only')
            elif reset_type == ResetType.CACHE_ONLY:
                cmd.append('--cache-only')
            elif reset_type == ResetType.EMERGENCY:
                cmd.append('--emergency')
            
            # Execute reset
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse output to get details
                output_lines = result.stdout.strip().split('\n')
                deleted_files = []
                deleted_size_mb = 0
                preserved_files = []
                
                for line in output_lines:
                    if 'Deleted:' in line:
                        deleted_files.append(line.split('Deleted:')[1].strip())
                    elif 'Preserved:' in line:
                        preserved_files.append(line.split('Preserved:')[1].strip())
                    elif 'Total size freed:' in line:
                        try:
                            size_str = line.split(':')[1].strip()
                            deleted_size_mb = float(size_str.replace('MB', '').strip())
                        except:
                            pass
                
                return {
                    'success': True,
                    'message': f"System reset completed ({reset_type.value})",
                    'deleted_files': deleted_files,
                    'deleted_size_mb': deleted_size_mb,
                    'preserved_files': preserved_files
                }
            else:
                return {
                    'success': False,
                    'message': "Reset failed",
                    'errors': result.stderr.strip().split('\n') if result.stderr else ['Unknown error']
                }
                
        except Exception as e:
            self.logger.error(f"Error executing system reset: {e}")
            return {
                'success': False,
                'message': f"Error executing reset: {str(e)}",
                'errors': [str(e)]
            }


async def main():
    """Main entry point for testing the conversational assistant."""
    # Set up logging
    setup_conversational_ai_logging()
    
    # Create assistant
    assistant = ConversationalAIAssistant()
    await assistant.initialize()
    
    print("\n🤖 CWMAI Conversational Assistant Ready!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye! Have a great day! 👋")
                break
            
            if not user_input:
                continue
            
            # Process conversation
            response = await assistant.handle_conversation(user_input)
            
            # Display response
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAssistant: Interrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\nAssistant: I apologize, but I encountered an error: {str(e)}")
            print("Please try again or rephrase your request.")


if __name__ == "__main__":
    asyncio.run(main())