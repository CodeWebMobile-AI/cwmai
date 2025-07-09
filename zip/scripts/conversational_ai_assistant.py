#!/usr/bin/env python3
"""
CWMAI Conversational AI Assistant

A Claude-like conversational interface that understands natural language,
maintains context, and can execute CWMAI commands within a friendly dialogue.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Set up custom logging for conversational AI
from scripts.conversational_ai_logger import setup_conversational_ai_logging, log_user_message, log_system_message
setup_conversational_ai_logging()

# CWMAI imports
from scripts.ai_brain import IntelligentAIBrain
from scripts.natural_language_interface import NaturalLanguageInterface
from scripts.task_manager import TaskManager
from scripts.state_manager import StateManager
from scripts.http_ai_client import HTTPAIClient
from scripts.dynamic_context_collector import DynamicContextCollector


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
    
    def __init__(self, style: ResponseStyle = ResponseStyle.FRIENDLY_PROFESSIONAL):
        """Initialize the conversational assistant.
        
        Args:
            style: The response style to use
        """
        self.logger = logging.getLogger(__name__)
        self.style = style
        
        # Core components
        self.ai_brain = IntelligentAIBrain(enable_round_robin=True)
        self.nli = NaturalLanguageInterface()
        self.http_client = HTTPAIClient(enable_round_robin=True)
        self.context_collector = DynamicContextCollector()
        
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
        except Exception as e:
            self.logger.debug(f"Could not load memory: {e}")
    
    def _save_memory(self):
        """Save conversation memory to disk."""
        try:
            self.memory_file.parent.mkdir(exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'user_preferences': self.memory.user_preferences,
                    'project_context': self.memory.project_context,
                    'common_tasks': self.memory.common_tasks[-20:],  # Keep last 20
                    'learned_patterns': self.memory.learned_patterns
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
        log_system_message("Assistant ready for conversation")
        log_user_message("Assistant initialized and ready!")
    
    async def handle_conversation(self, user_input: str) -> str:
        """Handle a conversational turn with the user using AI with full context.
        
        Args:
            user_input: The user's input
            
        Returns:
            The assistant's response
        """
        # Update conversation context
        self._update_context(user_input)
        
        # ALWAYS use AI with full context - no hardcoded patterns
        response = await self._handle_with_intelligent_context(user_input)
        
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
        """Handle any user input intelligently using AI with full system context.
        
        This is the ONLY handler - no hardcoded patterns.
        The AI has full context and decides everything.
        """
        log_system_message(f"Processing user input: {user_input[:100]}...")
        
        # Gather COMPLETE system context
        dynamic_context = await self._gather_complete_context(user_input)
        
        # Format the context
        formatted_context = self.context_collector.format_context_for_ai(dynamic_context)
        
        # Get available tools
        from scripts.tool_calling_system import ToolCallingSystem
        tool_system = ToolCallingSystem()
        available_tools = tool_system.get_tools_for_ai()
        tool_names = [t['name'] for t in available_tools]
        
        # Build comprehensive prompt for the AI with EVERYTHING
        prompt = f"""You are the CWMAI assistant with COMPLETE control over the system.

COMPLETE SYSTEM STATE:
{formatted_context}

ALL AVAILABLE TOOLS ({len(available_tools)} total):
{self._format_all_tools(available_tools)}

CONVERSATION HISTORY:
{self._format_complete_history()}

CURRENT CONTEXT:
- Time: {datetime.now(timezone.utc).isoformat()}
- Memory: {json.dumps(self.memory.__dict__, indent=2)}
- Pending confirmations: {json.dumps(self.current_context.get('pending_confirmations', {}))}
- System files: run_continuous_ai.py (starts system), reset_system.py (resets system)

USER INPUT: {user_input}

YOUR CAPABILITIES:
1. You can execute ANY tool by responding: EXECUTE_TOOL: tool_name(param1='value1', param2='value2')
2. You can create new tools: CREATE_TOOL: tool_name // description
3. You can run system commands: SYSTEM_COMMAND: command
4. You can start/stop the continuous AI: SYSTEM_COMMAND: python run_continuous_ai.py [options]
5. You can reset the system: SYSTEM_COMMAND: python reset_system.py [options]
6. You understand ALL natural language - "fire up", "how's it going", "shut down", etc.
7. For confirmations (yes/no), check pending_confirmations and execute the pending action

DECISION PROCESS:
- Analyze the user's intent from their natural language
- Check if there's a pending confirmation to handle
- Decide the best action: tool execution, system command, or informative response
- Use the context to provide accurate, real-time information
- Be conversational and natural in responses

RESPOND WITH THE APPROPRIATE ACTION OR HELPFUL INFORMATION."""

        try:
            # Get AI response
            response = await self.ai_brain.execute_capability('intelligent_analysis', {
                'prompt': prompt,
                'max_length': 600
            })
            
            ai_response = response.get('result', '')
            
            # Check for various AI directives
            if 'EXECUTE_TOOL:' in ai_response:
                tool_match = re.search(r'EXECUTE_TOOL:\s*(.+?)(?:\n|$)', ai_response)
                if tool_match:
                    tool_call = tool_match.group(1).strip()
                    log_system_message(f"AI executing tool: {tool_call}")
                    result = await self._execute_tool_call(tool_call)
                    return result
            
            elif 'CREATE_TOOL:' in ai_response:
                match = re.search(r'CREATE_TOOL:\s*(\w+)\s*//\s*(.+?)(?:\n|$)', ai_response)
                if match:
                    tool_name = match.group(1).strip()
                    description = match.group(2).strip()
                    log_system_message(f"AI creating tool: {tool_name}")
                    result = await self._create_and_execute_tool(tool_name, user_input, description)
                    return result
            
            elif 'SYSTEM_COMMAND:' in ai_response:
                match = re.search(r'SYSTEM_COMMAND:\s*(.+?)(?:\n|$)', ai_response)
                if match:
                    command = match.group(1).strip()
                    log_system_message(f"AI executing system command: {command}")
                    
                    # Handle special system commands
                    if 'run_continuous_ai.py' in command:
                        # Parse options from command
                        options = self._parse_continuous_ai_options(command)
                        result = await self.start_continuous_ai_system(**options)
                        return self._format_system_result(result)
                    
                    elif 'reset_system.py' in command:
                        # Parse reset options
                        reset_type = self._parse_reset_options(command)
                        result = await self.execute_system_reset(reset_type)
                        return self._format_reset_result(result)
                    
                    else:
                        # General command execution
                        try:
                            result = subprocess.run(command, shell=True, capture_output=True, text=True)
                            if result.returncode == 0:
                                return f"✓ Command executed successfully:\n{result.stdout}"
                            else:
                                return f"❌ Command failed:\n{result.stderr}"
                        except Exception as e:
                            return f"❌ Error executing command: {e}"
            
            
            # Return the AI's response
            return ai_response
            
        except Exception as e:
            log_system_message(f"Error in AI handler: {e}", "ERROR")
            return f"I encountered an error processing your request. Please try rephrasing or check the system logs."
    
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
        """Update conversation context based on user input."""
        # Extract potential project/repo mentions
        repo_pattern = r'\b([a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)?)\b'
        potential_repos = re.findall(repo_pattern, user_input)
        
        # Update topic if discussing something specific
        if 'issue' in user_input.lower():
            self.current_context['conversation_topic'] = 'issues'
        elif 'architecture' in user_input.lower():
            self.current_context['conversation_topic'] = 'architecture'
        elif 'status' in user_input.lower() or 'progress' in user_input.lower():
            self.current_context['conversation_topic'] = 'status'
    
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
            'learned_patterns': dict(sorted(self.memory.learned_patterns.items(), 
                                          key=lambda x: x[1], reverse=True)[:10])
        }
        
        # Add pending actions
        context['pending_actions'] = self.current_context.get('pending_confirmations', {})
        
        return context
    
    # REMOVED - No more pattern matching! AI decides everything
    
    async def _execute_tool_call(self, tool_call_str: str) -> str:
        """Execute a tool call string like 'tool_name(param1="value1")'."""
        try:
            # Parse tool name and parameters
            match = re.match(r'(\w+)\((.*)\)', tool_call_str)
            if not match:
                return f"Invalid tool call format: {tool_call_str}"
            
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters (simple implementation)
            params = {}
            if params_str:
                # Handle simple key=value pairs
                param_matches = re.findall(r'(\w+)=["\']?([^,"\']+)["\']?', params_str)
                for key, value in param_matches:
                    # Try to convert to appropriate type
                    if value.isdigit():
                        params[key] = int(value)
                    elif value.lower() in ['true', 'false']:
                        params[key] = value.lower() == 'true'
                    else:
                        params[key] = value
            
            # Execute the tool
            from scripts.tool_calling_system import ToolCallingSystem
            tool_system = ToolCallingSystem()
            result = await tool_system.call_tool(tool_name, **params)
            
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
        self.logger.info(f"Attempting to fix broken tool: {tool_name} with error: {error_msg}")
        
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
            self.logger.info(f"Deleted broken tool file: {tool_path}")
        
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
            # Tool created, now execute it
            result = await tool_system.call_tool(tool_name)
            
            if result.get('success'):
                return f"""✓ Created new tool '{tool_name}' successfully!

{self._format_tool_result(result.get('result'), tool_name)}

The tool has been saved and will be available for future use."""
            else:
                return f"Created tool '{tool_name}' but encountered an error executing it: {result.get('error')}"
        else:
            return f"I couldn't create the tool '{tool_name}': {creation_result.get('error')}"
    
    def _format_tool_result(self, result: Any, tool_name: str) -> str:
        """Format tool execution results nicely."""
        if isinstance(result, dict):
            # Special formatting for list_available_commands
            if 'by_category' in result and 'total_tools' in result:
                output = [f"📋 Available Commands ({result.get('total_tools', 0)} total)\n"]
                categories = result.get('by_category', {})
                for category, tools in categories.items():
                    if tools:
                        output.append(f"\n{category.upper()} ({len(tools)} commands):")
                        for tool in tools[:5]:  # Show first 5 in each category
                            name = tool.get('name', 'unknown')
                            desc = tool.get('description', '')
                            if tool.get('ai_created'):
                                output.append(f"  • {name} - {desc} [AI-CREATED]")
                            else:
                                output.append(f"  • {name} - {desc}")
                        if len(tools) > 5:
                            output.append(f"  ... and {len(tools) - 5} more")
                return '\n'.join(output)
            # Count-type results
            elif 'total' in result and 'summary' in result:
                return result.get('summary', str(result))
            # Action results
            elif 'success' in result and 'message' in result:
                return result.get('message')
            # Health check results
            elif 'health_breakdown' in result:
                breakdown = result.get('health_breakdown', {})
                return f"Health Check Complete: {breakdown.get('healthy', 0)} healthy, {breakdown.get('warning', 0)} warnings, {breakdown.get('critical', 0)} critical issues"
            else:
                # Generic dict - format key points
                key_points = []
                for key, value in result.items():
                    if key not in ['raw_data', 'metadata', 'detailed_stats', 'by_category']:
                        if isinstance(value, (str, int, float, bool)):
                            key_points.append(f"{key.replace('_', ' ').title()}: {value}")
                        elif isinstance(value, dict) and len(str(value)) < 100:
                            key_points.append(f"{key.replace('_', ' ').title()}: {value}")
                return '\n'.join(key_points) if key_points else str(result)
        elif isinstance(result, list):
            return self._format_result_data(result)
        else:
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
            'max_length': 500
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
                self.logger.info(f"AI suggested command: {suggested_command}")
                
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
        # Track command patterns
        command_verbs = ['create', 'show', 'search', 'list', 'analyze', 'generate']
        for verb in command_verbs:
            if verb in user_input.lower():
                self.memory.learned_patterns[verb] = self.memory.learned_patterns.get(verb, 0) + 1
        
        # Track common tasks
        if 'success' in response:
            self.memory.common_tasks.append(user_input)
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = f"Conversation Summary ({len(self.conversation_history)} turns):\n"
        summary += f"Started: {self.conversation_history[0].timestamp}\n"
        
        # Count action types
        action_counts = {}
        for turn in self.conversation_history:
            for action in turn.actions_taken:
                action_type = action.get('type', 'unknown')
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        if action_counts:
            summary += "\nActions taken:\n"
            for action_type, count in action_counts.items():
                summary += f"• {action_type}: {count}\n"
        
        return summary
    
    # Continuous AI System Control Methods
    
    async def check_continuous_ai_status(self) -> Dict[str, Any]:
        """Check if the continuous AI system is running and get its status.
        
        Returns:
            Dict containing status information
        """
        status = {
            'running': False,
            'pid': None,
            'uptime': None,
            'state_file_exists': False,
            'last_activity': None,
            'system_health': None,
            'active_workers': 0,
            'queued_tasks': 0,
            'completed_tasks': 0
        }
        
        # Check if process is running
        if self.continuous_ai_process and self.continuous_ai_process.poll() is None:
            status['running'] = True
            status['pid'] = self.continuous_ai_process.pid
        else:
            # Check for running process by name
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'run_continuous_ai.py' in ' '.join(cmdline):
                        status['running'] = True
                        status['pid'] = proc.info['pid']
                        # Calculate uptime
                        create_time = proc.create_time()
                        uptime = datetime.now().timestamp() - create_time
                        status['uptime'] = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        # Check state file
        state_file = Path("continuous_orchestrator_state.json")
        if state_file.exists():
            status['state_file_exists'] = True
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                    status['last_activity'] = state_data.get('last_updated', 'Unknown')
                    
                    # Extract metrics
                    metrics = state_data.get('metrics', {})
                    status['system_health'] = metrics.get('system_health', 0)
                    status['active_workers'] = metrics.get('active_workers', 0)
                    status['queued_tasks'] = metrics.get('tasks_queued', 0)
                    status['completed_tasks'] = metrics.get('tasks_completed', 0)
            except Exception as e:
                self.logger.error(f"Error reading state file: {e}")
        
        # Check system logs
        log_file = Path("continuous_ai.log")
        if log_file.exists():
            try:
                # Get last few lines from log
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Look for recent worker status updates
                        for line in reversed(lines[-20:]):
                            if "Worker Status:" in line:
                                # Extract worker info from log line
                                import re
                                match = re.search(r'(\d+)/(\d+) active.*Queue: (\d+) tasks.*Completed: (\d+)', line)
                                if match:
                                    status['active_workers'] = int(match.group(1))
                                    status['queued_tasks'] = int(match.group(3))
                                    status['completed_tasks'] = int(match.group(4))
                                    break
            except Exception as e:
                self.logger.debug(f"Could not read log file: {e}")
        
        return status
    
    async def start_continuous_ai_system(self, **kwargs) -> Dict[str, Any]:
        """Start the continuous AI system with specified options.
        
        Args:
            **kwargs: Options to pass to the continuous AI system
                - workers: Number of parallel workers (default: 3)
                - mode: Execution mode (production/development/test)
                - enable_research: Enable research engine (default: True)
                - enable_monitor: Enable worker monitoring (default: True)
                - enable_mcp: Enable MCP integration (default: True)
                
        Returns:
            Dict with success status and message
        """
        # Check if already running
        status = await self.check_continuous_ai_status()
        if status['running']:
            return {
                'success': False,
                'message': f"Continuous AI system is already running (PID: {status['pid']})",
                'status': status
            }
        
        # Build command
        cmd = [sys.executable, str(self.continuous_ai_script)]
        
        # Add options
        if 'workers' in kwargs:
            cmd.extend(['--workers', str(kwargs['workers'])])
        if 'mode' in kwargs:
            cmd.extend(['--mode', kwargs['mode']])
        if kwargs.get('enable_research', True):
            cmd.append('--no-research' if not kwargs['enable_research'] else '')
        if kwargs.get('enable_monitor', True):
            cmd.append('--monitor-workers')
        if 'enable_mcp' in kwargs:
            cmd.append('--mcp' if kwargs['enable_mcp'] else '--no-mcp')
        
        # Remove empty strings
        cmd = [c for c in cmd if c]
        
        try:
            # Start the process
            self.continuous_ai_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Wait a moment to ensure it started
            await asyncio.sleep(2)
            
            # Check if it's still running
            if self.continuous_ai_process.poll() is not None:
                # Process ended, get error
                stdout, stderr = self.continuous_ai_process.communicate()
                return {
                    'success': False,
                    'message': f"Failed to start continuous AI system",
                    'error': stderr or stdout
                }
            
            # Get updated status
            new_status = await self.check_continuous_ai_status()
            
            return {
                'success': True,
                'message': f"Continuous AI system started successfully (PID: {self.continuous_ai_process.pid})",
                'status': new_status
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error starting continuous AI system: {str(e)}",
                'error': str(e)
            }
    
    async def stop_continuous_ai_system(self, graceful: bool = True) -> Dict[str, Any]:
        """Stop the continuous AI system.
        
        Args:
            graceful: If True, attempt graceful shutdown with SIGTERM
            
        Returns:
            Dict with success status and message
        """
        # Check current status
        status = await self.check_continuous_ai_status()
        if not status['running']:
            return {
                'success': True,
                'message': "Continuous AI system is not running",
                'status': status
            }
        
        pid = status['pid']
        
        try:
            if graceful:
                # Send SIGTERM for graceful shutdown
                os.kill(pid, signal.SIGTERM)
                self.logger.info(f"Sent SIGTERM to continuous AI process (PID: {pid})")
                
                # Wait for process to end (up to 30 seconds)
                for i in range(30):
                    await asyncio.sleep(1)
                    if not psutil.pid_exists(pid):
                        return {
                            'success': True,
                            'message': f"Continuous AI system stopped gracefully",
                            'shutdown_time': i + 1
                        }
                
                # If still running, force kill
                self.logger.warning("Graceful shutdown timed out, forcing termination")
                os.kill(pid, signal.SIGKILL)
            else:
                # Force kill immediately
                os.kill(pid, signal.SIGKILL)
            
            # Wait for process to end
            await asyncio.sleep(1)
            
            return {
                'success': True,
                'message': "Continuous AI system stopped",
                'forced': not graceful or i >= 30
            }
            
        except ProcessLookupError:
            return {
                'success': True,
                'message': "Continuous AI system process not found (may have already stopped)"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error stopping continuous AI system: {str(e)}",
                'error': str(e)
            }
    
    async def monitor_system_health(self) -> Dict[str, Any]:
        """Get detailed health monitoring information about the continuous AI system.
        
        Returns:
            Dict containing detailed health metrics
        """
        # Get basic status
        status = await self.check_continuous_ai_status()
        
        health_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system_running': status['running'],
            'basic_status': status
        }
        
        if not status['running']:
            health_info['health_score'] = 0
            health_info['issues'] = ["System not running"]
            return health_info
        
        # Calculate health score
        health_score = 100
        issues = []
        recommendations = []
        
        # Check system metrics
        if status['system_health'] is not None:
            health_info['system_health'] = status['system_health']
            if status['system_health'] < 70:
                health_score -= 20
                issues.append(f"System health below threshold: {status['system_health']}%")
                recommendations.append("Consider investigating system logs for errors")
        
        # Check worker activity
        if status['active_workers'] == 0 and status['queued_tasks'] > 0:
            health_score -= 30
            issues.append("No active workers but tasks are queued")
            recommendations.append("Workers may be stuck - consider restarting the system")
        
        # Check task throughput
        if status['completed_tasks'] == 0 and status.get('uptime'):
            # Only flag if running for more than 5 minutes
            if 'h' in status['uptime'] or (int(status['uptime'].split('m')[0]) > 5):
                health_score -= 10
                issues.append("No tasks completed since startup")
                recommendations.append("Check if task generation is working properly")
        
        # Check state file freshness
        if status['last_activity']:
            try:
                last_update = datetime.fromisoformat(status['last_activity'].replace('Z', '+00:00'))
                age_minutes = (datetime.now(timezone.utc) - last_update).total_seconds() / 60
                if age_minutes > 5:
                    health_score -= 15
                    issues.append(f"State file not updated for {int(age_minutes)} minutes")
                    recommendations.append("System may be hung - check logs")
            except:
                pass
        
        # Get resource usage if process is running
        if status['pid']:
            try:
                process = psutil.Process(status['pid'])
                health_info['resource_usage'] = {
                    'cpu_percent': process.cpu_percent(interval=1),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'num_threads': process.num_threads()
                }
                
                # Check for high resource usage
                if health_info['resource_usage']['cpu_percent'] > 80:
                    health_score -= 10
                    issues.append("High CPU usage detected")
                    recommendations.append("Monitor for performance issues")
                    
                if health_info['resource_usage']['memory_mb'] > 1000:
                    health_score -= 10
                    issues.append("High memory usage (>1GB)")
                    recommendations.append("Consider restarting if memory continues to grow")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Ensure health score doesn't go negative
        health_score = max(0, health_score)
        
        health_info['health_score'] = health_score
        health_info['health_status'] = (
            'healthy' if health_score >= 80 else
            'degraded' if health_score >= 50 else
            'unhealthy'
        )
        health_info['issues'] = issues
        health_info['recommendations'] = recommendations
        
        # Add performance metrics
        health_info['performance'] = {
            'workers': {
                'active': status['active_workers'],
                'efficiency': (status['active_workers'] / 3 * 100) if status['active_workers'] else 0
            },
            'tasks': {
                'queued': status['queued_tasks'],
                'completed': status['completed_tasks'],
                'throughput': 'N/A'  # Could calculate if we track time
            }
        }
        
        return health_info
    
    # REMOVED - AI handles all continuous AI commands with full context
    
    # Reset Functionality Methods
    
    async def analyze_reset_need(self) -> Dict[str, Any]:
        """Analyze if the system needs a reset based on various health indicators.
        
        Returns:
            Dict containing analysis results and recommendations
        """
        analysis = {
            'needs_reset': False,
            'urgency': 'low',  # low, medium, high, critical
            'reasons': [],
            'recommended_type': None,
            'health_indicators': {}
        }
        
        # Check continuous AI health
        ai_status = await self.check_continuous_ai_status()
        if ai_status['running']:
            health = await self.monitor_system_health()
            analysis['health_indicators']['continuous_ai'] = health
            
            if health['health_score'] < 50:
                analysis['needs_reset'] = True
                analysis['urgency'] = 'high' if health['health_score'] < 30 else 'medium'
                analysis['reasons'].append(f"Continuous AI health score is low: {health['health_score']}/100")
        
        # Check log file sizes
        log_files = [
            Path("continuous_ai.log"),
            Path("continuous_ai_log.txt"),
            Path("scripts/god_mode.log"),
            Path("continuous_orchestrator.log")
        ]
        
        total_log_size = 0
        large_logs = []
        for log_file in log_files:
            if log_file.exists():
                size_mb = log_file.stat().st_size / 1024 / 1024
                total_log_size += size_mb
                if size_mb > 100:  # Log file over 100MB
                    large_logs.append((log_file.name, size_mb))
        
        analysis['health_indicators']['log_size_mb'] = total_log_size
        
        if total_log_size > 500:  # Total logs over 500MB
            analysis['needs_reset'] = True
            analysis['urgency'] = 'medium'
            analysis['reasons'].append(f"Log files are very large: {total_log_size:.1f}MB total")
            if not analysis['recommended_type']:
                analysis['recommended_type'] = ResetType.LOGS_ONLY
        
        # Check state file corruption
        state_files = [
            Path("system_state.json"),
            Path("task_state.json"),
            Path("continuous_orchestrator_state.json")
        ]
        
        corrupted_state_files = []
        for state_file in state_files:
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    corrupted_state_files.append(state_file.name)
        
        if corrupted_state_files:
            analysis['needs_reset'] = True
            analysis['urgency'] = 'high'
            analysis['reasons'].append(f"Corrupted state files: {', '.join(corrupted_state_files)}")
            analysis['recommended_type'] = ResetType.STATE_ONLY
        
        # Check for error patterns in recent logs
        error_count = 0
        if Path("continuous_ai.log").exists():
            try:
                with open("continuous_ai.log", 'r') as f:
                    # Read last 1000 lines
                    lines = f.readlines()[-1000:]
                    for line in lines:
                        if 'ERROR' in line or 'CRITICAL' in line or 'Traceback' in line:
                            error_count += 1
            except:
                pass
        
        analysis['health_indicators']['recent_errors'] = error_count
        
        if error_count > 50:
            analysis['needs_reset'] = True
            if error_count > 100:
                analysis['urgency'] = 'critical'
                analysis['recommended_type'] = ResetType.EMERGENCY
            else:
                analysis['urgency'] = 'high' if analysis['urgency'] == 'low' else analysis['urgency']
            analysis['reasons'].append(f"High error count in logs: {error_count} errors in recent logs")
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024 ** 3)
            analysis['health_indicators']['free_disk_gb'] = free_gb
            
            if free_gb < 1:
                analysis['needs_reset'] = True
                analysis['urgency'] = 'critical'
                analysis['reasons'].append(f"Very low disk space: {free_gb:.1f}GB free")
                if not analysis['recommended_type']:
                    analysis['recommended_type'] = ResetType.FULL
        except:
            pass
        
        # Determine recommended reset type if needed but not set
        if analysis['needs_reset'] and not analysis['recommended_type']:
            if analysis['urgency'] == 'critical':
                analysis['recommended_type'] = ResetType.EMERGENCY
            elif len(analysis['reasons']) > 2:
                analysis['recommended_type'] = ResetType.FULL
            else:
                analysis['recommended_type'] = ResetType.SELECTIVE
        
        return analysis
    
    async def recommend_reset_type(self, user_input: str) -> Dict[str, Any]:
        """Recommend the appropriate reset type based on user input and system state.
        
        Args:
            user_input: The user's request
            
        Returns:
            Dict with recommendation details
        """
        lower_input = user_input.lower()
        
        # First, analyze system health
        health_analysis = await self.analyze_reset_need()
        
        recommendation = {
            'type': None,
            'reason': '',
            'preserve': [],
            'delete': [],
            'warnings': [],
            'estimated_time': '1-2 minutes',
            'health_based': health_analysis['needs_reset']
        }
        
        # Parse user intent
        if 'emergency' in lower_input or 'urgent' in lower_input or 'crashed' in lower_input:
            recommendation['type'] = ResetType.EMERGENCY
            recommendation['reason'] = "Emergency reset for critical system issues"
            recommendation['delete'] = ['all state files', 'all logs', 'all caches', 'process locks']
            recommendation['warnings'].append("This will forcefully stop all running processes")
            recommendation['estimated_time'] = '30 seconds'
            
        elif 'everything' in lower_input or 'full' in lower_input or 'complete' in lower_input:
            recommendation['type'] = ResetType.FULL
            recommendation['reason'] = "Complete system reset to fresh state"
            recommendation['delete'] = ['all state files', 'all logs', 'all caches', 'task history', 'conversation memory']
            recommendation['warnings'].append("This will remove all system history and learned patterns")
            
        elif 'keep knowledge' in lower_input or 'preserve' in lower_input:
            recommendation['type'] = ResetType.SELECTIVE
            recommendation['reason'] = "Selective reset preserving important data"
            recommendation['preserve'] = ['AI response cache', 'conversation memory', 'learned patterns']
            recommendation['delete'] = ['state files', 'logs', 'temporary files']
            
        elif 'log' in lower_input:
            recommendation['type'] = ResetType.LOGS_ONLY
            recommendation['reason'] = "Clear only log files to free up space"
            recommendation['delete'] = ['all log files']
            recommendation['preserve'] = ['state', 'cache', 'configurations']
            recommendation['estimated_time'] = '10 seconds'
            
        elif 'state' in lower_input:
            recommendation['type'] = ResetType.STATE_ONLY
            recommendation['reason'] = "Reset system state while preserving logs"
            recommendation['delete'] = ['state files', 'task queues']
            recommendation['preserve'] = ['logs', 'cache', 'configurations']
            recommendation['estimated_time'] = '20 seconds'
            
        elif 'cache' in lower_input:
            recommendation['type'] = ResetType.CACHE_ONLY
            recommendation['reason'] = "Clear cached data only"
            recommendation['delete'] = ['AI response cache', 'temporary files']
            recommendation['preserve'] = ['state', 'logs', 'configurations']
            recommendation['estimated_time'] = '15 seconds'
            
        else:
            # Use health-based recommendation
            if health_analysis['recommended_type']:
                recommendation['type'] = health_analysis['recommended_type']
                recommendation['reason'] = f"Based on system health: {', '.join(health_analysis['reasons'])}"
                
                # Set details based on type
                if recommendation['type'] == ResetType.EMERGENCY:
                    recommendation['delete'] = ['all state files', 'all logs', 'all caches', 'process locks']
                    recommendation['warnings'].append("System is in critical state - emergency reset recommended")
                elif recommendation['type'] == ResetType.FULL:
                    recommendation['delete'] = ['all state files', 'all logs', 'all caches']
                    recommendation['warnings'].append("Multiple issues detected - full reset recommended")
                else:
                    recommendation['type'] = ResetType.SELECTIVE
                    recommendation['reason'] = "Selective reset to address specific issues"
                    recommendation['preserve'] = ['AI cache', 'important configurations']
                    recommendation['delete'] = ['problematic files', 'large logs']
        
        return recommendation
    
    async def execute_system_reset(self, reset_type: ResetType, dry_run: bool = False) -> Dict[str, Any]:
        """Execute the specified type of system reset.
        
        Args:
            reset_type: The type of reset to perform
            dry_run: If True, only show what would be deleted without actually deleting
            
        Returns:
            Dict with reset results
        """
        result = {
            'success': False,
            'type': reset_type.value,
            'deleted_files': [],
            'deleted_size_mb': 0,
            'preserved_files': [],
            'errors': [],
            'warnings': [],
            'dry_run': dry_run
        }
        
        # Define file patterns for each reset type
        patterns = {
            ResetType.FULL: {
                'delete': [
                    '*.log', '*.json', '*.pkl', '*.db', '*.cache',
                    'scripts/*.log', 'scripts/*.json', '__pycache__',
                    'scripts/__pycache__', '.pytest_cache', 'dump.rdb'
                ],
                'preserve': []
            },
            ResetType.SELECTIVE: {
                'delete': [
                    '*.log', 'system_state.json', 'task_state.json',
                    'continuous_orchestrator_state.json', '*.pkl',
                    'scripts/*.log', '__pycache__', 'scripts/__pycache__'
                ],
                'preserve': [
                    'scripts/ai_response_cache.json',
                    str(self.memory_file),
                    'scripts/learned_patterns.json'
                ]
            },
            ResetType.EMERGENCY: {
                'delete': [
                    '*.log', '*.json', '*.pkl', '*.lock', '*.pid',
                    'scripts/*.log', 'scripts/*.json', '__pycache__',
                    'scripts/__pycache__', 'dump.rdb'
                ],
                'preserve': [],
                'force_stop_processes': True
            },
            ResetType.LOGS_ONLY: {
                'delete': ['*.log', 'scripts/*.log', '*.txt'],
                'preserve': ['*.json', '*.pkl', '*.db']
            },
            ResetType.STATE_ONLY: {
                'delete': [
                    '*state*.json', 'task_history.json',
                    'scripts/*state*.json'
                ],
                'preserve': ['*.log', '*.cache']
            },
            ResetType.CACHE_ONLY: {
                'delete': [
                    '*.cache', '*cache*.json', '__pycache__',
                    'scripts/__pycache__', '.pytest_cache'
                ],
                'preserve': ['*.log', '*state*.json']
            }
        }
        
        pattern_config = patterns.get(reset_type, patterns[ResetType.SELECTIVE])
        
        # Stop continuous AI if needed
        if reset_type in [ResetType.FULL, ResetType.EMERGENCY]:
            if not dry_run:
                status = await self.check_continuous_ai_status()
                if status['running']:
                    self.logger.info("Stopping continuous AI system before reset...")
                    stop_result = await self.stop_continuous_ai_system(
                        graceful=(reset_type != ResetType.EMERGENCY)
                    )
                    if not stop_result['success']:
                        result['warnings'].append("Could not stop continuous AI system cleanly")
        
        # Force stop all Python processes if emergency reset
        if pattern_config.get('force_stop_processes') and not dry_run:
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('cwmai' in str(arg) for arg in cmdline):
                            if proc.info['pid'] != os.getpid():  # Don't kill ourselves
                                proc.terminate()
                                result['warnings'].append(f"Terminated process {proc.info['pid']}")
                    except:
                        pass
            except Exception as e:
                result['errors'].append(f"Error stopping processes: {e}")
        
        # Collect files to delete
        files_to_delete = []
        workspace_root = Path("/workspaces/cwmai")
        
        for pattern in pattern_config['delete']:
            # Handle directory patterns
            if '/' in pattern:
                base_pattern = pattern.split('/')[-1]
                base_dir = workspace_root / '/'.join(pattern.split('/')[:-1])
                if base_dir.exists():
                    files_to_delete.extend(base_dir.glob(base_pattern))
            else:
                files_to_delete.extend(workspace_root.glob(pattern))
        
        # Filter out preserved files
        preserve_paths = [Path(p).resolve() for p in pattern_config['preserve']]
        files_to_delete = [
            f for f in files_to_delete 
            if f.resolve() not in preserve_paths and f.exists()
        ]
        
        # Calculate total size
        total_size = 0
        for file in files_to_delete:
            try:
                if file.is_file():
                    total_size += file.stat().st_size
                elif file.is_dir():
                    total_size += sum(f.stat().st_size for f in file.rglob('*') if f.is_file())
            except:
                pass
        
        result['deleted_size_mb'] = total_size / 1024 / 1024
        
        # Delete files (or just list them if dry run)
        for file in files_to_delete:
            try:
                if dry_run:
                    result['deleted_files'].append(str(file.relative_to(workspace_root)))
                else:
                    if file.is_dir():
                        shutil.rmtree(file)
                    else:
                        file.unlink()
                    result['deleted_files'].append(str(file.relative_to(workspace_root)))
            except Exception as e:
                result['errors'].append(f"Could not delete {file}: {e}")
        
        # List preserved files
        for pattern in pattern_config['preserve']:
            preserved = list(workspace_root.glob(pattern))
            result['preserved_files'].extend([
                str(f.relative_to(workspace_root)) for f in preserved if f.exists()
            ])
        
        # Create fresh state files if needed (not in dry run)
        if not dry_run and reset_type != ResetType.LOGS_ONLY:
            try:
                # Create minimal fresh state
                fresh_state = {
                    'version': '1.0',
                    'last_reset': datetime.now(timezone.utc).isoformat(),
                    'reset_type': reset_type.value,
                    'initialized': True
                }
                
                if reset_type in [ResetType.FULL, ResetType.STATE_ONLY]:
                    with open(workspace_root / "system_state.json", 'w') as f:
                        json.dump(fresh_state, f, indent=2)
                    result['warnings'].append("Created fresh system state file")
            except Exception as e:
                result['errors'].append(f"Could not create fresh state: {e}")
        
        result['success'] = len(result['errors']) == 0
        
        return result
    
    # REMOVED - AI handles all reset requests with full context
        
        # We have a recommendation - show what will happen
        response = f"I understand you want to perform a {recommendation['type'].value} reset.\n\n"
        response += f"**Reason:** {recommendation['reason']}\n\n"
        
        if recommendation['delete']:
            response += "**What will be deleted:**\n"
            for item in recommendation['delete']:
                response += f"• {item}\n"
            response += "\n"
        
        if recommendation['preserve']:
            response += "**What will be preserved:**\n"
            for item in recommendation['preserve']:
                response += f"• {item}\n"
            response += "\n"
        
        if recommendation['warnings']:
            response += "⚠️ **Important:**\n"
            for warning in recommendation['warnings']:
                response += f"• {warning}\n"
            response += "\n"
        
        response += f"Estimated time: {recommendation['estimated_time']}\n\n"
        
        # First do a dry run to show exact files
        dry_run_result = await self.execute_system_reset(recommendation['type'], dry_run=True)
        
        if dry_run_result['deleted_files']:
            response += f"This will delete {len(dry_run_result['deleted_files'])} files "
            response += f"({dry_run_result['deleted_size_mb']:.1f}MB)\n\n"
        
        response += "Should I proceed with this reset? (yes/no)"
        
        # Store pending confirmation
        self.current_context['pending_confirmations']['reset'] = {
            'action': 'reset',
            'type': recommendation['type'],
            'original_command': user_input
        }
        
        return response
    
    # REMOVED - AI handles all problem reports with full context
    
    async def _execute_pending_action(self, pending: Dict[str, Any]) -> str:
        """Execute a pending action after confirmation."""
        action = pending.get('action')
        
        if action == 'reset':
            # Execute the reset
            reset_type = pending.get('type')
            result = await self.execute_system_reset(reset_type, dry_run=False)
            
            if result['success']:
                response = f"✓ Successfully completed {reset_type.value} reset!\n\n"
                response += f"• Deleted {len(result['deleted_files'])} files\n"
                response += f"• Freed up {result['deleted_size_mb']:.1f}MB of space\n"
                
                if result['preserved_files']:
                    response += f"• Preserved {len(result['preserved_files'])} important files\n"
                
                if result['warnings']:
                    response += "\n⚠️ Notes:\n"
                    for warning in result['warnings']:
                        response += f"• {warning}\n"
                
                response += "\nThe system has been reset. "
                
                # Offer to restart continuous AI if it was stopped
                if reset_type in [ResetType.FULL, ResetType.EMERGENCY]:
                    response += "Would you like me to start the continuous AI system?"
                
                return response
            else:
                response = f"I encountered some issues during the {reset_type.value} reset:\n\n"
                for error in result['errors']:
                    response += f"• {error}\n"
                response += "\nThe reset was partially completed. Would you like me to try again?"
                return response
        else:
            # Call parent implementation for other actions
            params = pending.get('params', {})
            
            # Execute the action
            result = await self.nli.process_natural_language(pending.get('original_command', ''))
            
            if result.get('success'):
                return f"Done! I've {action} as requested."
            else:
                return f"I ran into an issue: {result.get('reason', 'Unknown error')}. Would you like to try a different approach?"