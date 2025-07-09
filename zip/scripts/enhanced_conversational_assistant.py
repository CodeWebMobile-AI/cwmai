"""
Enhanced Conversational AI Assistant

A dramatically improved AI assistant with self-improvement capabilities,
background intelligence, and modular architecture.
"""

import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from scripts.ai_brain import IntelligentAIBrain
from scripts.http_ai_client import HTTPAIClient
from scripts.state_manager import StateManager
from scripts.tool_calling_system import ToolCallingSystem

# Enhanced imports
from scripts.self_improvement_orchestrator import SelfImprovementOrchestrator
from scripts.background_intelligence_daemon import BackgroundIntelligenceDaemon
from scripts.intelligent_codebase_context import IntelligentCodebaseContext
from scripts.dynamic_context_collector import DynamicContextCollector

# Natural Language Todo System imports
from scripts.natural_language_todo_queue import ConcurrentTodoQueue, TodoMessage, MessagePriority
from scripts.todo_processor_pool import DynamicProcessorPool
from scripts.todo_nlp_processor import TodoNLPProcessor, TodoIntent
from scripts.natural_language_todo_manager import NaturalLanguageTodoManager, TodoStatus

# Streaming support imports (with fallback)
try:
    from scripts.streaming_response_generator import StreamingResponseGenerator, VisualStateIndicator
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Logging
from scripts.conversational_ai_logger import setup_conversational_ai_logging, log_user_message, log_system_message


@dataclass
class ConversationContext:
    """Enhanced conversation context with learning capabilities."""
    user_input: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    response_time: float = 0.0
    user_satisfaction: Optional[float] = None
    assistant_response: Optional[str] = None
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    

class ToolExecutor:
    """Handles tool execution with enhanced error handling and learning."""
    
    def __init__(self, tool_system: ToolCallingSystem, improvement_orchestrator: SelfImprovementOrchestrator):
        self.tool_system = tool_system
        self.improvement_orchestrator = improvement_orchestrator
        self.logger = logging.getLogger(__name__)
        
    async def execute_tool(self, tool_call_str: str) -> Dict[str, Any]:
        """Execute a tool call with enhanced handling."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Parse tool call - updated to handle dots in tool names
            match = re.match(r'([\w\.]+)\((.*)\)$', tool_call_str, re.DOTALL)
            if not match:
                return {"success": False, "error": "Invalid tool call format"}
                
            tool_name = match.group(1)
            params_str = match.group(2).strip()
            
            # Parse parameters safely
            params = self._parse_parameters(params_str)
            
            # Execute tool
            result = await self.tool_system.call_tool(tool_name, **params)
            
            # Check if result is an error dict
            if isinstance(result, dict) and 'error' in result and not result.get('success', True):
                return {
                    "success": False,
                    "error": result['error'],
                    "tool_name": tool_name,
                    "execution_time": asyncio.get_event_loop().time() - start_time
                }
            
            # Record success
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "success": True,
                "result": result,
                "tool_name": tool_name,
                "execution_time": execution_time
            }
            
        except Exception as e:
            # Learn from failure
            error_context = {
                "tool_call": tool_call_str,
                "error": str(e),
                "type": "tool_execution_error"
            }
            
            # Let improvement orchestrator learn from this
            await self.improvement_orchestrator.analyze_interaction(
                tool_call_str,
                f"Error: {str(e)}",
                {"error": error_context}
            )
            
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name if 'tool_name' in locals() else "unknown"
            }
            
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse tool parameters safely."""
        import ast
        
        params = {}
        if not params_str:
            return params
            
        try:
            # Try AST parsing first
            tree = ast.parse(f"f({params_str})")
            call = tree.body[0].value
            
            for keyword in call.keywords:
                if keyword.arg:
                    try:
                        value = ast.literal_eval(compile(ast.Expression(keyword.value), '<string>', 'eval'))
                        params[keyword.arg] = value
                    except:
                        params[keyword.arg] = ast.unparse(keyword.value)
                        
        except:
            # Fallback to simple parsing
            param_matches = re.findall(r'(\w+)=(["\']?)([^,"\']+)\2', params_str)
            for key, _, value in param_matches:
                params[key] = value
                
        return params


class ResponseGenerator:
    """Generates intelligent responses based on context and learning."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        self.ai_brain = ai_brain
        self.response_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load response templates."""
        return {
            'success': [
                "✓ Done! {}",
                "✓ Successfully completed: {}",
                "✓ I've {} for you."
            ],
            'error': [
                "I encountered an issue: {}. Let me try a different approach.",
                "There was a problem: {}. Here's what I can do instead:",
                "I couldn't complete that because: {}. Would you like me to try something else?"
            ],
            'learning': [
                "I notice you often ask about {}. I'll remember this preference.",
                "Based on your previous requests, I've optimized {} for you.",
                "I've learned a better way to handle {} from our interactions."
            ]
        }
        
    async def generate_response(self, context: ConversationContext, results: List[Dict[str, Any]]) -> str:
        """Generate an intelligent response based on context and results."""
        if not results:
            return await self._generate_direct_response(context)
            
        # Check if all successful
        all_success = all(r.get('success', False) for r in results)
        
        if all_success:
            return await self._generate_success_response(context, results)
        else:
            return await self._generate_error_response(context, results)
            
    async def _generate_direct_response(self, context: ConversationContext) -> str:
        """Generate response without tool execution."""
        prompt = f"""Generate a helpful response to: {context.user_input}
        
Be conversational, helpful, and concise. If you need more information, ask for clarification."""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'claude'
        })
        
        return response.get('result', "I'm here to help. Could you please clarify what you'd like me to do?")
        
    async def _generate_success_response(self, context: ConversationContext, results: List[Dict[str, Any]]) -> str:
        """Generate response for successful operations."""
        # Summarize results
        summary_prompt = f"""The user asked: {context.user_input}

I successfully executed these operations:
{json.dumps(results, indent=2)}

Provide a natural, conversational summary of the results. Highlight key information."""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': summary_prompt,
            'model': 'claude'
        })
        
        return response.get('result', "Operation completed successfully!")
        
    async def _generate_error_response(self, context: ConversationContext, results: List[Dict[str, Any]]) -> str:
        """Generate helpful response for errors."""
        errors = [r for r in results if not r.get('success', False)]
        
        error_prompt = f"""The user asked: {context.user_input}

Some operations failed:
{json.dumps(errors, indent=2)}

Provide a helpful response that:
1. Explains what went wrong in simple terms
2. Suggests alternatives or solutions
3. Maintains a positive, helpful tone"""
        
        response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': error_prompt,
            'model': 'claude'
        })
        
        return response.get('result', "I encountered some issues, but I'm here to help find a solution.")


class EnhancedConversationalAssistant:
    """The dramatically improved conversational AI assistant."""
    
    def __init__(self):
        """Initialize the enhanced assistant."""
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.ai_brain = IntelligentAIBrain(enable_round_robin=False)
        self.http_client = HTTPAIClient(enable_round_robin=False)
        self.state_manager = StateManager()
        self.tool_system = ToolCallingSystem()
        
        # Enhanced components
        self.improvement_orchestrator = SelfImprovementOrchestrator()
        self.background_daemon = BackgroundIntelligenceDaemon()
        self.context_collector = DynamicContextCollector()
        self.codebase_context = IntelligentCodebaseContext()
        
        # Natural Language Todo System
        self.todo_queue = ConcurrentTodoQueue()
        self.todo_nlp = TodoNLPProcessor(use_ai=False)  # Start without AI for performance
        self.todo_manager = NaturalLanguageTodoManager()
        self.todo_processor_pool = None  # Initialized in initialize()
        
        # Streaming support
        if STREAMING_AVAILABLE:
            self.streamer = StreamingResponseGenerator(typing_speed=0.02)
            self.visual_indicator = VisualStateIndicator()
            self.enable_streaming = True
        else:
            self.streamer = None
            self.visual_indicator = None
            self.enable_streaming = False
        
        # Specialized handlers
        self.tool_executor = ToolExecutor(self.tool_system, self.improvement_orchestrator)
        self.response_generator = ResponseGenerator(self.ai_brain)
        
        # Conversation management
        self.conversation_history: List[ConversationContext] = []
        self.max_history = 50
        
        # Performance tracking
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "average_response_time": 0.0,
            "tools_created": 0,
            "improvements_applied": 0
        }
        
    async def initialize(self):
        """Initialize all components."""
        log_system_message("Initializing Enhanced Conversational Assistant...")
        
        # Tool system doesn't need initialization - it's ready after construction
        
        # Initialize todo processor pool
        self.todo_processor_pool = DynamicProcessorPool(
            self.todo_queue,
            self._process_todo_message,
            min_processors=2,
            max_processors=5
        )
        await self.todo_processor_pool.start()
        
        # Background processes disabled by default for performance
        # Uncomment to enable self-improvement features:
        # asyncio.create_task(self.improvement_orchestrator.start_background_learning())
        # asyncio.create_task(self.background_daemon.start())
        
        log_system_message("Enhanced Assistant initialized with self-improvement capabilities and natural language todo management")
        
    async def handle_conversation(self, user_input: str, stream_output: bool = None) -> str:
        """Handle user input with enhanced intelligence and optional streaming."""
        start_time = asyncio.get_event_loop().time()
        
        # Use streaming if available and enabled
        use_streaming = stream_output if stream_output is not None else self.enable_streaming
        
        # Show processing indicator if streaming
        if use_streaming and self.visual_indicator:
            await self.visual_indicator.show_state("processing", "Analyzing your request")
        
        # Create conversation context
        context = ConversationContext(user_input=user_input)
        
        try:
            # First, check if this is a todo-related request
            todo_command = await self.todo_nlp.process(user_input)
            
            if todo_command.intent != TodoIntent.UNKNOWN and todo_command.confidence > 0.6:
                # This is a todo request - queue it for concurrent processing
                priority = MessagePriority.HIGH if "urgent" in user_input.lower() else MessagePriority.NORMAL
                message = self.todo_queue.enqueue(user_input, priority)
                
                # Show queue status if there are other pending items
                queue_status = self.todo_queue.get_status()
                if queue_status['pending'] > 1 and self.visual_indicator:
                    await self.visual_indicator.show_queue_status(
                        queue_status['pending'], 
                        queue_status['processing']
                    )
                
                # Stream or return immediate acknowledgment
                if todo_command.intent == TodoIntent.ADD:
                    response = "Adding that to your todo list..."
                    if use_streaming and self.streamer:
                        await self.streamer.stream_response(response)
                        return ""  # Already streamed
                    return response
                elif todo_command.intent == TodoIntent.LIST:
                    # For list operations, wait for result
                    result = await self._wait_for_todo_result(message.id, timeout=2.0)
                    if result:
                        response = self._format_todo_result(result)
                        if use_streaming and self.streamer:
                            await self.streamer.stream_response(response)
                            return ""
                        return response
                elif todo_command.intent == TodoIntent.COMPLETE:
                    response = "Marking that task as complete..."
                    if use_streaming and self.streamer:
                        await self.streamer.stream_response(response)
                        return ""
                    return response
                else:
                    response = "Processing your todo request..."
                    if use_streaming and self.streamer:
                        await self.streamer.stream_response(response)
                        return ""
                    return response
            
            # Not a todo request - continue with normal processing
            # Get comprehensive context
            full_context = await self._gather_intelligent_context(user_input)
            
            # Process with AI
            response = await self._process_with_ai(user_input, full_context, context)
            self.logger.debug(f"Response type from _process_with_ai: {type(response)}")
            self.logger.debug(f"Response content: {response[:100] if isinstance(response, str) else response}")
            
            # Track performance
            context.response_time = asyncio.get_event_loop().time() - start_time
            context.success = True
            context.assistant_response = response  # Store the response
            
            # Learning disabled for performance - uncomment to enable:
            # await self.improvement_orchestrator.analyze_interaction(
            #     user_input,
            #     response,
            #     {"context": full_context, "tool_calls": context.tool_calls}
            # )
            
            # Update metrics
            self._update_metrics(context)
            
            # Stream response if enabled
            if use_streaming and self.streamer and response:
                await self.streamer.stream_response(response)
                if self.visual_indicator:
                    await self.visual_indicator.show_state("completed", "Response complete")
                return ""  # Already streamed
            
            return response
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error handling conversation: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            context.success = False
            
            # Learn from error
            await self.improvement_orchestrator.analyze_interaction(
                user_input,
                f"Error: {str(e)}",
                {"error": str(e), "type": "system_error"}
            )
            
            return "I apologize, but I encountered an error. Let me try a different approach..."
            
        finally:
            # Store context
            self.conversation_history.append(context)
            if len(self.conversation_history) > self.max_history:
                self.conversation_history.pop(0)
                
    async def _gather_intelligent_context(self, user_input: str) -> Dict[str, Any]:
        """Gather comprehensive context for the request."""
        # Get conversation history with full context
        # Dynamic history size based on content
        history_size = self._calculate_optimal_history_size(user_input)
        
        history = [{
            'user_input': ctx.user_input,
            'timestamp': ctx.timestamp.isoformat(),
            'success': ctx.success,
            'tool_calls': ctx.tool_calls,
            'assistant_response': ctx.assistant_response,
            'tool_results': ctx.tool_results,
            'intent': ctx.intent,
            'entities': ctx.entities
        } for ctx in self.conversation_history[-history_size:]]
        
        # Get dynamic context
        dynamic_context = await self.context_collector.gather_context_for_query(
            user_input,
            history
        )
        
        # Get codebase context
        codebase_info = await self.codebase_context.get_context_for_query(user_input)
        dynamic_context['codebase'] = codebase_info
        
        # Get learning insights
        learning_summary = await self.improvement_orchestrator.get_learning_summary()
        dynamic_context['learning'] = learning_summary
        
        # Get available tools
        tools = self.tool_system.get_tools_for_ai()
        dynamic_context['available_tools'] = len(tools)
        
        # Get real tool categories from the tool system
        dynamic_context['tool_categories'] = self.tool_system.get_tool_categories()
        
        # Get actual tool names
        dynamic_context['tool_names'] = [tool['name'] for tool in tools]
        
        # Get tools organized by category
        dynamic_context['tools_by_category'] = self.tool_system.get_tools_summary_by_category()
        
        # Add conversation history to context
        dynamic_context['conversation_history'] = history
        
        return dynamic_context
        
    async def _process_with_ai(self, user_input: str, context: Dict[str, Any], conversation_context: ConversationContext) -> str:
        """Process user input with AI and execute any needed tools."""
        # Build comprehensive prompt
        prompt = self._build_ai_prompt(user_input, context)
        
        # Get AI response
        ai_response = await self.ai_brain.execute_capability('intelligent_analysis', {
            'prompt': prompt,
            'model': 'claude'
        })
        
        # Handle both dict and list responses
        if isinstance(ai_response, dict):
            response_text = ai_response.get('result', ai_response.get('content', ''))
        elif isinstance(ai_response, list):
            # If it's a list, join the responses
            response_text = ' '.join(str(item) for item in ai_response)
        else:
            response_text = str(ai_response)
        
        # Extract and execute tool calls
        tool_results = await self._extract_and_execute_tools(response_text)
        
        # Store tool calls and results in conversation context
        for result in tool_results:
            conversation_context.tool_calls.append({
                'tool_name': result.get('tool_name'),
                'execution_time': result.get('execution_time'),
                'success': result.get('success')
            })
            conversation_context.tool_results.append(result)
        
        # Generate final response
        if tool_results:
            final_response = await self.response_generator.generate_response(
                conversation_context,
                tool_results
            )
            return final_response
        else:
            return response_text
            
    def _build_ai_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Build an intelligent prompt for the AI."""
        # Build conversation history section
        history_text = ""
        if 'conversation_history' in context and context['conversation_history']:
            history_count = len(context['conversation_history'])
            history_text = f"\nRecent Conversation History ({history_count} turns):\n"
            
            # For many turns, summarize older ones and show recent in detail
            if history_count > 10:
                # Summarize older turns
                history_text += "\n[Earlier conversation summary]\n"
                for turn in context['conversation_history'][:-5]:
                    history_text += f"• {turn['user_input'][:50]}..."
                    if turn.get('tool_calls'):
                        history_text += f" (used {len(turn['tool_calls'])} tools)"
                    history_text += "\n"
                
                history_text += "\n[Recent detailed history]\n"
                detailed_turns = context['conversation_history'][-5:]
            else:
                detailed_turns = context['conversation_history']
            
            # Show detailed history for recent/all turns
            for turn in detailed_turns:
                history_text += f"\nUser: {turn['user_input']}\n"
                if turn.get('assistant_response'):
                    # Truncate very long responses but keep meaningful content
                    response_preview = turn['assistant_response'][:500]
                    if len(turn['assistant_response']) > 500:
                        response_preview += "..."
                    history_text += f"Assistant: {response_preview}\n"
                if turn.get('tool_calls'):
                    for tool_call in turn['tool_calls']:
                        history_text += f"  → Used tool: {tool_call.get('tool_name')} (success: {tool_call.get('success')})\n"
        
        # Build tools section for the prompt
        tools_text = ""
        if 'tools_by_category' in context:
            tools_text = "\nAvailable Tools by Category:\n"
            for category, info in context['tools_by_category'].items():
                # Handle both dict and list formats
                if isinstance(info, dict):
                    tool_names = info.get('tools', [])
                elif isinstance(info, list):
                    tool_names = info
                else:
                    tool_names = []
                if tool_names:
                    # Show all tools - Gemini has 1M context window
                    tools_display = ', '.join(sorted(tool_names))
                    tools_text += f"- {category} ({len(tool_names)} tools): {tools_display}\n"
        
        return f"""You are an enhanced AI assistant with self-improvement capabilities.

Current Context:
- Total available tools: {context['available_tools']}
{tools_text}
- Recent learning insights: {context['learning']['total_insights']} insights gathered
- System has applied {context['learning']['applied_improvements']} improvements
{history_text}
User Request: {user_input}

Instructions:
1. Understand the user's intent clearly
2. Use available tools when needed (format: EXECUTE_TOOL::tool_name(params)::END_TOOL)
3. Use ONLY the exact tool names listed above - do NOT add category prefixes (e.g., use "list_files", NOT "retrieval.list_files")
4. If a needed tool doesn't exist, mention it (the system will learn and create it)
5. Provide helpful, conversational responses
6. Learn from any errors or issues
7. You can see your conversation history above - use it to answer questions about what you did previously

IMPORTANT: 
- Tool names are case-sensitive. Use them exactly as shown above.
- When showing examples or explaining how tools work, ALWAYS wrap them in backticks (`) or code blocks (```)
- Examples in code blocks will NOT be executed - this is how you show examples safely
- Only use EXECUTE_TOOL outside of code blocks when you actually need to run a tool
- Example format: `EXECUTE_TOOL::example_tool(param='value')::END_TOOL` ← This won't execute
- Actual execution: EXECUTE_TOOL::real_tool(param='value')::END_TOOL ← This WILL execute

Remember: You have the ability to improve yourself. If you notice patterns or opportunities for improvement, they will be automatically captured and applied.

Respond naturally and helpfully."""
        
    async def _extract_and_execute_tools(self, ai_response: str) -> List[Dict[str, Any]]:
        """Extract and execute tool calls from AI response."""
        results = []
        
        # First, remove any content within code blocks to prevent example execution
        # This preserves the original response but prevents execution of examples
        response_without_codeblocks = re.sub(r'```[\s\S]*?```', '', ai_response)
        response_without_inline_code = re.sub(r'`[^`]+`', '', response_without_codeblocks)
        
        # Find tool calls - updated pattern to handle dots in tool names (for backward compatibility)
        tool_pattern = r'EXECUTE_TOOL::([\w\.]+)\s*\((.*?)\)::END_TOOL'
        
        for match in re.finditer(tool_pattern, response_without_inline_code, re.DOTALL):
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Skip obvious example/placeholder tool names
            example_names = ['tool_name', 'example_tool', 'your_tool', 'some_tool', 'placeholder']
            if tool_name.lower() in example_names:
                self.logger.debug(f"Skipping example tool execution: {tool_name}")
                continue
            
            # Check if this tool call is preceded by example-indicating text
            start_pos = match.start()
            preceding_text = response_without_inline_code[max(0, start_pos-100):start_pos].lower()
            example_indicators = ['for example', 'example:', 'e.g.', 'such as', 'like this', 'here\'s how']
            if any(indicator in preceding_text for indicator in example_indicators):
                self.logger.debug(f"Skipping tool execution due to example context: {tool_name}")
                continue
            
            # Log tool call attempt
            self.logger.info(f"Extracting tool call: {tool_name}({params_str})")
            log_system_message(f"Attempting to execute tool: {tool_name}")
            
            # Handle tool names with prefixes (e.g., retrieval.list_files -> list_files)
            actual_tool_name = tool_name
            if '.' in tool_name and tool_name not in self.tool_system.tools:
                # Try without prefix
                simple_name = tool_name.split('.')[-1]
                if simple_name in self.tool_system.tools:
                    actual_tool_name = simple_name
                    self.logger.warning(f"Tool name '{tool_name}' not found, using '{actual_tool_name}' instead")
                    log_system_message(f"⚠️ Corrected tool name: {tool_name} → {actual_tool_name}")
            
            # Build tool call string with actual tool name
            tool_call = f"{actual_tool_name}({params_str})"
            
            try:
                # Execute the tool
                result = await self.tool_executor.execute_tool(tool_call)
                
                # Log the result
                if result.get('success'):
                    self.logger.info(f"Tool execution successful: {actual_tool_name}")
                    log_system_message(f"✓ Tool '{actual_tool_name}' executed successfully in {result.get('execution_time', 0):.2f}s")
                else:
                    self.logger.error(f"Tool execution failed: {actual_tool_name} - {result.get('error')}")
                    log_system_message(f"✗ Tool '{actual_tool_name}' failed: {result.get('error')}")
                
                # Add original tool name to result for tracking
                result['original_tool_name'] = tool_name
                result['resolved_tool_name'] = actual_tool_name
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Exception during tool execution: {tool_name} - {str(e)}")
                log_system_message(f"✗ Exception executing tool '{tool_name}': {str(e)}")
                
                # Add error result
                results.append({
                    'success': False,
                    'error': str(e),
                    'tool_name': actual_tool_name,
                    'original_tool_name': tool_name,
                    'resolved_tool_name': actual_tool_name
                })
        
        if results:
            self.logger.info(f"Executed {len(results)} tools with {sum(1 for r in results if r.get('success'))} successes")
        
        return results
        
    def _categorize_tools(self, tools: List[Dict[str, Any]]) -> List[str]:
        """Categorize available tools."""
        categories = set()
        
        for tool in tools:
            name = tool.get('name', '').lower()
            if 'create' in name or 'generate' in name:
                categories.add('creation')
            elif 'analyze' in name or 'check' in name:
                categories.add('analysis')
            elif 'count' in name or 'calculate' in name:
                categories.add('metrics')
            elif 'get' in name or 'list' in name:
                categories.add('retrieval')
            else:
                categories.add('utility')
                
        return list(categories)
    
    def _calculate_optimal_history_size(self, user_input: str) -> int:
        """Calculate optimal number of conversation turns to include based on query."""
        # Keywords that suggest need for more history
        history_keywords = [
            'previous', 'last', 'before', 'earlier', 'history', 'did you',
            'have you', 'what tools', 'conversation', 'remember', 'forgot'
        ]
        
        # Keywords that suggest debugging/complex tasks
        debug_keywords = [
            'error', 'debug', 'fix', 'issue', 'problem', 'why', 'explain',
            'trace', 'investigate'
        ]
        
        # Default sizes
        default_size = 5
        query_lower = user_input.lower()
        
        # Check for history-related queries
        if any(keyword in query_lower for keyword in history_keywords):
            return 20  # User asking about history, provide more context
        
        # Check for debugging queries
        if any(keyword in query_lower for keyword in debug_keywords):
            return 15  # Debugging often needs more context
        
        # Check for tool-related queries
        if 'tool' in query_lower:
            return 10  # Tool usage queries benefit from seeing recent tools
        
        # Check conversation history for multi-turn patterns
        if len(self.conversation_history) > 3:
            recent_topics = set()
            for ctx in self.conversation_history[-3:]:
                # Extract key topics
                words = ctx.user_input.lower().split()
                recent_topics.update(word for word in words if len(word) > 4)
            
            # If current query relates to recent topics, include more history
            current_words = set(word for word in query_lower.split() if len(word) > 4)
            if len(recent_topics & current_words) > 2:
                return 10  # Related to ongoing conversation
        
        # For simple queries, use less history to save tokens
        if len(user_input.split()) < 10:
            return 3
        
        return default_size
        
    def _update_metrics(self, context: ConversationContext):
        """Update performance metrics."""
        self.performance_metrics["total_interactions"] += 1
        
        if context.success:
            self.performance_metrics["successful_interactions"] += 1
            
        # Update average response time
        total = self.performance_metrics["total_interactions"]
        avg = self.performance_metrics["average_response_time"]
        new_avg = (avg * (total - 1) + context.response_time) / total
        self.performance_metrics["average_response_time"] = new_avg
        
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the assistant."""
        learning_summary = await self.improvement_orchestrator.get_learning_summary()
        
        return {
            "performance": self.performance_metrics,
            "learning": learning_summary,
            "conversation_history_size": len(self.conversation_history),
            "available_tools": len(self.tool_system.tools),
            "background_tasks": {
                "improvement_orchestrator": "running",
                "background_daemon": "running" if self.background_daemon.running else "stopped"
            }
        }
    
    async def _process_todo_message(self, message: TodoMessage) -> Dict[str, Any]:
        """Process a todo message using the todo manager."""
        try:
            # Parse with NLP
            command = await self.todo_nlp.process(message.content)
            
            # Execute based on intent
            if command.intent == TodoIntent.ADD:
                task = command.entities.get('task', message.content)
                priority = command.entities.get('priority', 'medium')
                todo = self.todo_manager.add_todo(task, priority=priority)
                return {
                    'success': True,
                    'action': 'added',
                    'message': f"✓ Added: \"{todo.description}\"",
                    'todo': todo
                }
                
            elif command.intent == TodoIntent.LIST:
                todos = self.todo_manager.list_todos(status=TodoStatus.PENDING)
                return {
                    'success': True,
                    'action': 'list',
                    'todos': todos,
                    'count': len(todos)
                }
                
            elif command.intent == TodoIntent.COMPLETE:
                task_ref = command.entities.get('task_reference', '')
                todo = self.todo_manager.complete_todo(task_ref)
                if todo:
                    return {
                        'success': True,
                        'action': 'completed',
                        'message': f"✓ Completed: \"{todo.description}\""
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Couldn't find task: \"{task_ref}\""
                    }
                    
            else:
                return {
                    'success': False,
                    'message': "I'm not sure what to do with that todo request."
                }
                
        except Exception as e:
            self.logger.error(f"Todo processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': "Error processing todo request."
            }
    
    async def _wait_for_todo_result(self, message_id: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Wait for a todo message to be processed."""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if message_id in self.todo_queue._completed:
                message = self.todo_queue._completed[message_id]
                if message.status == MessageStatus.COMPLETED:
                    return message.result
                else:
                    return {'success': False, 'error': message.error}
                    
            await asyncio.sleep(0.1)
            
        return None
    
    def _format_todo_result(self, result: Dict[str, Any]) -> str:
        """Format todo result for display."""
        if not result.get('success'):
            return result.get('message', 'Error processing todo request.')
            
        action = result.get('action')
        
        if action == 'list':
            todos = result.get('todos', [])
            if not todos:
                return "You have no pending tasks."
                
            lines = ["Here are your tasks:"]
            for i, todo in enumerate(todos, 1):
                priority_emoji = {"urgent": "🔴", "high": "🟡", "medium": "🟢", "low": "⚪"}.get(todo.priority, "")
                lines.append(f"{i}. {priority_emoji} {todo.description}")
                if todo.due_date:
                    lines.append(f"   Due: {todo.due_date.strftime('%b %d at %I:%M %p')}")
                    
            return "\n".join(lines)
            
        else:
            return result.get('message', 'Todo request processed.')


async def main():
    """Run the enhanced assistant with Claude Code-like streaming interface."""
    setup_conversational_ai_logging()
    
    assistant = EnhancedConversationalAssistant()
    await assistant.initialize()
    
    # Clear screen for clean interface
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("🤖 Enhanced AI Assistant")
    print("=" * 60)
    print("\n✨ Features:")
    print("  • Natural language understanding - no syntax required")
    print("  • Fast, instant responses")
    print("  • Concurrent input handling (type while I respond!)")
    print("  • Natural language todo management")
    print("  • Self-improvement & Background Intelligence")
    print("  • 20+ tools at your disposal")
    
    print("\n📝 Example commands:")
    print("  • 'Add a task to review the documentation tomorrow'")
    print("  • 'Show me my tasks'")
    print("  • 'What tools do you have?'")
    print("  • 'Create a tool to...'")
    
    print("\nType 'status' for system status, 'quit' to exit.\n")
    
    # Streaming disabled for performance - responses appear instantly
    
    # Import async input handler
    from scripts.async_input_handler import AsyncInputHandler
    input_handler = AsyncInputHandler()
    input_handler.start()
    
    # Track if we're currently processing
    is_processing = False
    pending_inputs = []
    
    # Show initial prompt
    sys.stdout.write("\n> ")
    sys.stdout.flush()
    
    while True:
        try:
            # Check for input (non-blocking)
            user_input = await input_handler.get_input(timeout=0.05)
            
            if user_input is None:
                # No input yet, continue loop
                await asyncio.sleep(0.05)
                continue
                
            user_input = user_input.strip()
            
            if user_input.lower() in ['quit', 'exit']:
                if assistant.enable_streaming and assistant.streamer:
                    await assistant.streamer.stream_response("Goodbye! The system will continue improving in the background.", indicator="👋")
                else:
                    print("\n👋 Goodbye! The system will continue improving in the background.")
                break
                
            if user_input.lower() == 'status':
                status = await assistant.get_status()
                print(f"\nSystem Status:\n{json.dumps(status, indent=2)}")
                continue
            
            # Check if already processing
            if is_processing:
                # Queue the input
                pending_inputs.append(user_input)
                print(f"📋 Queued your input (currently processing another request)")
                continue
                
            # Set processing flag
            is_processing = True
            
            # Show processing indicator if available
            if assistant.enable_streaming and assistant.visual_indicator:
                spinner_task = asyncio.create_task(
                    assistant.visual_indicator.show_spinner("Processing your request")
                )
            else:
                spinner_task = None
                print("⏳ Processing your request...")
            
            # Process in background so input loop continues
            async def process_request(input_text):
                try:
                    # Get response
                    response = await assistant.handle_conversation(input_text, stream_output=False)
                    
                    # Stop spinner if running
                    if spinner_task and assistant.visual_indicator:
                        assistant.visual_indicator.stop_spinner()
                        await asyncio.sleep(0.1)
                    
                    # Show response
                    if response:
                        print(f"\n🤖 {response}")
                        
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                finally:
                    nonlocal is_processing
                    is_processing = False
                    
                    # Process next queued input if any
                    if pending_inputs:
                        next_input = pending_inputs.pop(0)
                        print(f"\n📋 Processing queued input: {next_input}")
                        await process_request(next_input)
            
            # Start processing in background
            asyncio.create_task(process_request(user_input))
            
            # Show queue status if there are pending todo items
            if assistant.todo_queue:
                queue_status = assistant.todo_queue.get_status()
                if queue_status['pending'] > 0 and assistant.visual_indicator:
                    await assistant.visual_indicator.show_queue_status(
                        queue_status['pending'],
                        queue_status['processing']
                    )
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            input_handler.stop()
            break
        except Exception as e:
            import traceback
            print(f"\n❌ Error in main loop: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            # Continue running despite errors
            
    # Cleanup
    input_handler.stop()
            

if __name__ == "__main__":
    asyncio.run(main())