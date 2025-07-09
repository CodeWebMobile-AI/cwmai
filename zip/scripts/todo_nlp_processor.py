"""
Todo Natural Language Processor

Processes natural language input to extract todo-related intents and entities,
enabling conversational todo management without rigid syntax.
"""

import re
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from module_installer import module_installer

# Try to auto-install dateparser if not available
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    if module_installer.try_install('dateparser'):
        import dateparser
        DATEPARSER_AVAILABLE = True
    else:
        DATEPARSER_AVAILABLE = False

from http_ai_client import HTTPAIClient


class TodoIntent(Enum):
    """Types of todo-related intents."""
    ADD = "add"
    COMPLETE = "complete"
    DELETE = "delete"
    LIST = "list"
    UPDATE = "update"
    SEARCH = "search"
    PRIORITIZE = "prioritize"
    SCHEDULE = "schedule"
    CLARIFY = "clarify"
    HELP = "help"
    UNKNOWN = "unknown"


class TodoPriority(Enum):
    """Todo priority levels."""
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class TodoEntity:
    """Extracted entity from natural language."""
    entity_type: str
    value: Any
    confidence: float = 1.0
    span: Optional[Tuple[int, int]] = None  # Start, end position in text


@dataclass
class ProcessedTodoCommand:
    """Result of processing a natural language todo command."""
    original_text: str
    intent: TodoIntent
    entities: Dict[str, Any]
    confidence: float
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    suggested_actions: List[str] = None


class TodoNLPProcessor:
    """Processes natural language for todo management."""
    
    def __init__(self, use_ai: bool = True):
        """Initialize the NLP processor."""
        self.logger = logging.getLogger(__name__)
        self.use_ai = use_ai
        self.ai_client = HTTPAIClient() if use_ai else None
        
        # Pattern-based intent detection
        self.intent_patterns = {
            TodoIntent.ADD: [
                r"\b(add|create|new|make|todo|remind|remember|schedule)\b",
                r"\b(need to|have to|must|should|will)\b",
                r"^(?!.*\b(show|list|display|complete|done|finish)\b).*\btask\b"
            ],
            TodoIntent.COMPLETE: [
                r"\b(complete|done|finish|completed|finished|mark|check off)\b",
                r"\b(just|already|i)\s+(did|finished|completed)\b",
                r"✓|✔|☑"
            ],
            TodoIntent.DELETE: [
                r"\b(delete|remove|cancel|drop|clear)\b.*\b(task|todo|item)\b",
                r"\b(don't need|no longer need)\b"
            ],
            TodoIntent.LIST: [
                r"\b(show|list|display|what|get|see|view)\b.*\b(task|todo|list|agenda|schedule)\b",
                r"\b(what's|what is|whats)\s+(on|my|the)\b",
                r"^(todo|tasks|agenda)s?\s*$"
            ],
            TodoIntent.UPDATE: [
                r"\b(update|change|modify|edit|rename|move)\b",
                r"\b(actually|instead|make that)\b"
            ],
            TodoIntent.PRIORITIZE: [
                r"\b(prioritize|priority|urgent|important|asap|high|low)\b",
                r"\b(make|set)\b.*\b(priority|urgent|important)\b"
            ],
            TodoIntent.SCHEDULE: [
                r"\b(schedule|reschedule|postpone|delay|move)\b.*\b(to|for|until)\b",
                r"\b(tomorrow|today|next|this|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"
            ],
            TodoIntent.SEARCH: [
                r"\b(search|find|where|which|filter)\b.*\b(task|todo)\b",
                r"\b(tasks?|todos?)\b.*\b(about|related|containing)\b"
            ],
            TodoIntent.HELP: [
                r"\b(help|how|what can|commands|usage)\b",
                r"^\?+$"
            ]
        }
        
        # Priority keywords
        self.priority_keywords = {
            TodoPriority.URGENT: ["urgent", "asap", "immediately", "critical", "emergency"],
            TodoPriority.HIGH: ["high", "important", "priority", "soon"],
            TodoPriority.MEDIUM: ["medium", "normal", "regular"],
            TodoPriority.LOW: ["low", "whenever", "eventually", "someday"]
        }
        
        # Time expressions
        self.relative_time_patterns = {
            "today": 0,
            "tomorrow": 1,
            "day after tomorrow": 2,
            "next week": 7,
            "next month": 30,
            "in (\\d+) days?": lambda m: int(m.group(1)),
            "in (\\d+) hours?": lambda m: 0  # Same day
        }
        
    async def process(self, text: str) -> ProcessedTodoCommand:
        """Process natural language text to extract todo command."""
        self.logger.debug(f"Processing text: {text}")
        
        # Try AI processing first if available
        if self.use_ai and self.ai_client:
            try:
                ai_result = await self._process_with_ai(text)
                if ai_result and ai_result.confidence > 0.8:
                    return ai_result
            except Exception as e:
                self.logger.warning(f"AI processing failed, falling back to patterns: {e}")
        
        # Fall back to pattern-based processing
        return self._process_with_patterns(text)
        
    def _process_with_patterns(self, text: str) -> ProcessedTodoCommand:
        """Process using pattern matching."""
        text_lower = text.lower().strip()
        
        # Detect intent
        intent, intent_confidence = self._detect_intent(text_lower)
        
        # Extract entities based on intent
        entities = self._extract_entities(text, intent)
        
        # Check if clarification needed
        needs_clarification, clarification_prompt = self._check_clarification_needed(
            intent, entities, text
        )
        
        # Generate suggested actions
        suggested_actions = self._generate_suggestions(intent, entities, text)
        
        return ProcessedTodoCommand(
            original_text=text,
            intent=intent,
            entities=entities,
            confidence=intent_confidence,
            needs_clarification=needs_clarification,
            clarification_prompt=clarification_prompt,
            suggested_actions=suggested_actions
        )
        
    def _detect_intent(self, text: str) -> Tuple[TodoIntent, float]:
        """Detect the intent from text."""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            scores[intent] = score
            
        # Get highest scoring intent
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] > 0:
                # Calculate confidence based on pattern matches
                confidence = min(scores[best_intent] / len(self.intent_patterns[best_intent]), 1.0)
                return best_intent, confidence
                
        return TodoIntent.UNKNOWN, 0.0
        
    def _extract_entities(self, text: str, intent: TodoIntent) -> Dict[str, Any]:
        """Extract entities from text based on intent."""
        entities = {}
        
        # Extract task description
        if intent in [TodoIntent.ADD, TodoIntent.UPDATE]:
            task_desc = self._extract_task_description(text, intent)
            if task_desc:
                entities["task"] = task_desc
                
        # Extract priority
        priority = self._extract_priority(text)
        if priority != TodoPriority.NONE:
            entities["priority"] = priority.value
            
        # Extract time/date
        time_info = self._extract_temporal_info(text)
        if time_info:
            entities.update(time_info)
            
        # Extract task reference for complete/update/delete
        if intent in [TodoIntent.COMPLETE, TodoIntent.UPDATE, TodoIntent.DELETE]:
            task_ref = self._extract_task_reference(text)
            if task_ref:
                entities["task_reference"] = task_ref
                
        # Extract search query
        if intent == TodoIntent.SEARCH:
            query = self._extract_search_query(text)
            if query:
                entities["query"] = query
                
        # Extract tags/categories
        tags = self._extract_tags(text)
        if tags:
            entities["tags"] = tags
            
        return entities
        
    def _extract_task_description(self, text: str, intent: TodoIntent) -> Optional[str]:
        """Extract the task description from text."""
        # Remove intent keywords
        cleaned = text
        
        # Remove common prefixes
        prefixes = [
            r"^(please\s+)?(can you\s+)?(could you\s+)?",
            r"^(add|create|new|make|todo|remind me to|remember to)\s+((a|an|the)\s+)?(task|todo|reminder)?\s*(to|for|about|that)?\s*",
            r"^(i\s+)?(need to|have to|must|should|will)\s+"
        ]
        
        for prefix in prefixes:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()
            
        # Remove temporal expressions at the end
        time_suffixes = [
            r"\s+(by|on|at|before|after|until|for|in)\s+.+$",
            r"\s+(today|tomorrow|tonight|later|soon|asap)$"
        ]
        
        for suffix in time_suffixes:
            match = re.search(suffix, cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[:match.start()].strip()
                
        return cleaned if cleaned else None
        
    def _extract_priority(self, text: str) -> TodoPriority:
        """Extract priority from text."""
        text_lower = text.lower()
        
        for priority, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
                    
        # Check for emphasis (all caps, exclamation marks)
        if text.isupper() or text.count('!') >= 2:
            return TodoPriority.HIGH
            
        return TodoPriority.NONE
        
    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract time and date information."""
        temporal_info = {}
        
        # Try dateparser first for complex expressions if available
        if DATEPARSER_AVAILABLE:
            try:
                # Look for date expressions
                date_match = dateparser.search.search_dates(
                    text,
                    settings={'PREFER_DATES_FROM': 'future'}
                )
                
                if date_match:
                    # Use the first found date
                    date_str, parsed_date = date_match[0]
                    temporal_info["due_date"] = parsed_date.isoformat()
                    temporal_info["date_string"] = date_str
                    return temporal_info
            except:
                pass
            
        # Fall back to simple patterns
        text_lower = text.lower()
        
        # Check relative time patterns
        for pattern, days in self.relative_time_patterns.items():
            if isinstance(days, int):
                if pattern in text_lower:
                    due_date = datetime.now(timezone.utc) + timedelta(days=days)
                    temporal_info["due_date"] = due_date.isoformat()
                    temporal_info["date_string"] = pattern
                    break
            else:
                # It's a regex pattern with a function
                match = re.search(pattern, text_lower)
                if match:
                    days_offset = days(match)
                    due_date = datetime.now(timezone.utc) + timedelta(days=days_offset)
                    temporal_info["due_date"] = due_date.isoformat()
                    temporal_info["date_string"] = match.group(0)
                    break
                    
        # Extract time of day
        time_match = re.search(r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)
            
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
                
            temporal_info["time"] = f"{hour:02d}:{minute:02d}"
            
        return temporal_info
        
    def _extract_task_reference(self, text: str) -> Optional[str]:
        """Extract reference to an existing task."""
        # Look for quoted text
        quoted = re.search(r'"([^"]+)"', text)
        if quoted:
            return quoted.group(1)
            
        # Look for "the X task"
        the_task = re.search(r"the\s+(\w+(?:\s+\w+)*)\s+task", text, re.IGNORECASE)
        if the_task:
            return the_task.group(1)
            
        # Remove intent words and use the rest
        cleaned = text
        remove_patterns = [
            r"^(mark|complete|finish|delete|remove|update|change)\s+",
            r"\s+(as\s+)?(done|complete|finished)$",
            r"^(the|a|an)\s+"
        ]
        
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            
        return cleaned if cleaned else None
        
    def _extract_search_query(self, text: str) -> Optional[str]:
        """Extract search query from text."""
        # Look for "about/containing/related to X"
        patterns = [
            r"(?:about|containing|related to|for|with)\s+(.+)$",
            r"(?:find|search for|show me)\s+(?:tasks?\s+)?(.+)$"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        return None
        
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags or categories from text."""
        tags = []
        
        # Look for #hashtags
        hashtags = re.findall(r"#(\w+)", text)
        tags.extend(hashtags)
        
        # Look for @mentions (could be categories)
        mentions = re.findall(r"@(\w+)", text)
        tags.extend(mentions)
        
        # Look for category keywords
        category_match = re.search(
            r"(?:category|tag|label|type)(?:\s+is)?\s*:?\s*(\w+(?:\s+\w+)*)",
            text,
            re.IGNORECASE
        )
        if category_match:
            tags.append(category_match.group(1).strip())
            
        return list(set(tags))  # Remove duplicates
        
    def _check_clarification_needed(
        self,
        intent: TodoIntent,
        entities: Dict[str, Any],
        text: str
    ) -> Tuple[bool, Optional[str]]:
        """Check if clarification is needed."""
        if intent == TodoIntent.UNKNOWN:
            return True, "I'm not sure what you want to do. You can add tasks, list them, or mark them complete."
            
        if intent == TodoIntent.ADD and "task" not in entities:
            return True, "What task would you like to add?"
            
        if intent in [TodoIntent.COMPLETE, TodoIntent.DELETE, TodoIntent.UPDATE]:
            if "task_reference" not in entities:
                return True, "Which task are you referring to? Please be more specific."
                
        return False, None
        
    def _generate_suggestions(
        self,
        intent: TodoIntent,
        entities: Dict[str, Any],
        text: str
    ) -> List[str]:
        """Generate suggested actions based on context."""
        suggestions = []
        
        if intent == TodoIntent.ADD:
            if "priority" not in entities:
                suggestions.append("Set priority: high, medium, or low")
            if "due_date" not in entities:
                suggestions.append("Add due date: 'by tomorrow', 'next week', etc.")
                
        elif intent == TodoIntent.LIST:
            suggestions.extend([
                "Filter by: 'show high priority tasks'",
                "Time filter: 'show tasks for today'",
                "Search: 'find tasks about project X'"
            ])
            
        elif intent == TodoIntent.UNKNOWN:
            suggestions.extend([
                "Add a task: 'add review documentation'",
                "List tasks: 'show my tasks'",
                "Complete task: 'mark API task as done'"
            ])
            
        return suggestions
        
    async def _process_with_ai(self, text: str) -> Optional[ProcessedTodoCommand]:
        """Process using AI for better understanding."""
        prompt = f"""Analyze this todo-related message and extract the intent and entities.
        
Message: "{text}"

Return a JSON object with:
- intent: one of [add, complete, delete, list, update, search, prioritize, schedule, help, unknown]
- entities: object with relevant extracted information
- confidence: float between 0 and 1

Example response:
{{
    "intent": "add",
    "entities": {{
        "task": "Review API documentation",
        "priority": "high",
        "due_date": "2024-01-15T14:00:00Z"
    }},
    "confidence": 0.95
}}

Analyze the message and return only the JSON object."""

        try:
            response = await self.ai_client.generate_enhanced_response(prompt)
            
            # Parse response
            if isinstance(response, dict):
                data = response
            else:
                data = json.loads(str(response))
                
            # Convert to our format
            intent = TodoIntent(data.get("intent", "unknown"))
            entities = data.get("entities", {})
            confidence = float(data.get("confidence", 0.5))
            
            return ProcessedTodoCommand(
                original_text=text,
                intent=intent,
                entities=entities,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"AI processing error: {e}")
            return None


# Example usage
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_processor():
        processor = TodoNLPProcessor(use_ai=False)  # Test without AI
        
        test_inputs = [
            "Add a task to review the API documentation tomorrow",
            "Show me what I need to do today",
            "Mark the documentation task as done",
            "What's my highest priority task?",
            "Create a reminder to call John at 3pm",
            "I finished the API review",
            "Delete the meeting task",
            "Update the project deadline to next Friday",
            "Find all tasks about documentation",
            "Make the report task high priority"
        ]
        
        for text in test_inputs:
            result = await processor.process(text)
            print(f"\nInput: {text}")
            print(f"Intent: {result.intent.value} (confidence: {result.confidence:.2f})")
            print(f"Entities: {result.entities}")
            if result.needs_clarification:
                print(f"Clarification: {result.clarification_prompt}")
            if result.suggested_actions:
                print(f"Suggestions: {result.suggested_actions}")
                
    asyncio.run(test_processor())