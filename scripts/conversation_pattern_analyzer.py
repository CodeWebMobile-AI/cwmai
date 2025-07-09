#!/usr/bin/env python3
"""
Conversation Pattern Analyzer

Advanced module for learning and analyzing conversation patterns to improve
natural language understanding and user interaction quality.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ConversationPattern:
    """Represents a learned conversation pattern."""
    pattern_id: str
    pattern_type: str  # sequence, query_response, multi_turn, clarification
    frequency: int
    confidence: float
    examples: List[Dict[str, Any]]
    context_requirements: List[str]
    learned_at: datetime
    last_seen: datetime
    success_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserIntent:
    """Represents a user's intent with confidence scores."""
    primary_intent: str
    confidence: float
    secondary_intents: List[Tuple[str, float]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    context_clues: List[str] = field(default_factory=list)


@dataclass
class DialogueFlow:
    """Tracks the flow of a multi-turn dialogue."""
    turns: List[Dict[str, Any]]
    current_topic: str
    topic_history: List[str]
    clarifications_needed: List[str]
    user_satisfaction: float  # 0-1 score
    completion_status: str  # ongoing, completed, abandoned


class ConversationPatternAnalyzer:
    """Analyzes conversation patterns to improve NLU capabilities."""
    
    def __init__(self, storage_path: str = "conversation_patterns.json"):
        """Initialize the pattern analyzer.
        
        Args:
            storage_path: Path to store learned patterns
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = Path(storage_path)
        
        # Pattern storage
        self.patterns: Dict[str, ConversationPattern] = {}
        self.intent_patterns: Dict[str, List[str]] = defaultdict(list)
        self.dialogue_flows: List[DialogueFlow] = []
        
        # Learning components
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 3))
        self.intent_clusters: Optional[KMeans] = None
        self.pattern_templates: Dict[str, List[str]] = self._initialize_templates()
        
        # Metrics
        self.pattern_metrics = {
            'total_conversations': 0,
            'successful_interactions': 0,
            'failed_interactions': 0,
            'clarifications_required': 0,
            'average_turns_per_conversation': 0.0,
            'user_satisfaction_score': 0.0
        }
        
        # Load existing patterns
        self._load_patterns()
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize common conversation pattern templates."""
        return {
            'task_request': [
                r'(please|can you|could you|would you) (.+)',
                r'i (need|want) (to|you to) (.+)',
                r'(help|assist) (me|with) (.+)',
                r'(create|make|build|generate) (.+)'
            ],
            'information_query': [
                r'(what|how|when|where|why|who) (.+)\?',
                r'(tell me|explain|describe) (.+)',
                r'(show me|list|display) (.+)',
                r'(is|are|can|does) (.+)\?'
            ],
            'clarification': [
                r'(what do you mean|can you clarify) (.+)',
                r'i (dont understand|didnt get) (.+)',
                r'(be more specific|elaborate) (.+)',
                r'(which|what kind of) (.+) do you mean'
            ],
            'feedback': [
                r'(that|this) (worked|didnt work|failed)',
                r'(perfect|great|excellent|good job)',
                r'(not what i wanted|wrong|incorrect)',
                r'(try again|do it differently)'
            ],
            'multi_step': [
                r'first (.+) then (.+)',
                r'(.+) and (then|after that) (.+)',
                r'step \d+: (.+)',
                r'(.+) followed by (.+)'
            ]
        }
    
    def analyze_conversation_turn(self, user_input: str, assistant_response: str, 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single conversation turn for patterns.
        
        Args:
            user_input: The user's input
            assistant_response: The assistant's response
            context: Current conversation context
            
        Returns:
            Analysis results including detected patterns and intents
        """
        analysis = {
            'detected_patterns': [],
            'intent': self._detect_intent(user_input, context),
            'entities': self._extract_entities(user_input),
            'sentiment': self._analyze_sentiment(user_input),
            'complexity': self._assess_complexity(user_input),
            'requires_clarification': False,
            'suggested_followups': []
        }
        
        # Check for pattern matches
        for pattern_type, templates in self.pattern_templates.items():
            for template in templates:
                if re.search(template, user_input.lower()):
                    analysis['detected_patterns'].append({
                        'type': pattern_type,
                        'template': template,
                        'confidence': 0.8
                    })
        
        # Analyze dialogue flow
        if context.get('conversation_history'):
            flow_analysis = self._analyze_dialogue_flow(
                user_input, 
                assistant_response, 
                context['conversation_history']
            )
            analysis.update(flow_analysis)
        
        # Learn from this interaction
        self._update_patterns(user_input, assistant_response, analysis, context)
        
        return analysis
    
    def _detect_intent(self, user_input: str, context: Dict[str, Any]) -> UserIntent:
        """Detect user intent using advanced NLU techniques.
        
        Args:
            user_input: The user's input
            context: Current conversation context
            
        Returns:
            Detected user intent with confidence scores
        """
        # Extract features
        features = self._extract_features(user_input)
        
        # Check against known intent patterns
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = self._calculate_intent_score(user_input, patterns, context)
            if score > 0.3:
                intent_scores[intent] = score
        
        # Sort by confidence
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_intents:
            primary = sorted_intents[0]
            secondary = sorted_intents[1:3] if len(sorted_intents) > 1 else []
            
            return UserIntent(
                primary_intent=primary[0],
                confidence=primary[1],
                secondary_intents=secondary,
                entities=self._extract_entities(user_input),
                context_clues=self._extract_context_clues(user_input, context)
            )
        
        # Default intent if none detected
        return UserIntent(
            primary_intent='general_query',
            confidence=0.5,
            entities=self._extract_entities(user_input)
        )
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from user input.
        
        Args:
            text: User input text
            
        Returns:
            Extracted entities
        """
        entities = {}
        
        # Extract file paths
        file_pattern = r'(?:[\w\-./]+\.[\w]+|/[\w\-./]+)'
        files = re.findall(file_pattern, text)
        if files:
            entities['files'] = files
        
        # Extract URLs
        url_pattern = r'https?://(?:[-\w.])+(?:\:\d+)?(?:/[^\s]*)?'
        urls = re.findall(url_pattern, text)
        if urls:
            entities['urls'] = urls
        
        # Extract numbers
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)
        if numbers:
            entities['numbers'] = [int(n) for n in numbers]
        
        # Extract quoted strings
        quote_pattern = r'"([^"]*)"'
        quotes = re.findall(quote_pattern, text)
        if quotes:
            entities['quoted_strings'] = quotes
        
        # Extract action verbs
        action_verbs = ['create', 'delete', 'update', 'modify', 'analyze', 
                        'generate', 'test', 'run', 'execute', 'search']
        found_actions = [verb for verb in action_verbs if verb in text.lower()]
        if found_actions:
            entities['actions'] = found_actions
        
        return entities
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of user input.
        
        Args:
            text: User input text
            
        Returns:
            Sentiment scores
        """
        # Simple keyword-based sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'perfect', 'thanks', 
                          'helpful', 'awesome', 'wonderful']
        negative_words = ['bad', 'wrong', 'incorrect', 'failed', 'error', 
                          'problem', 'issue', 'broken']
        neutral_words = ['okay', 'fine', 'alright']
        
        text_lower = text.lower()
        
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        neutral_score = sum(1 for word in neutral_words if word in text_lower)
        
        total = positive_score + negative_score + neutral_score
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        return {
            'positive': positive_score / total,
            'negative': negative_score / total,
            'neutral': neutral_score / total
        }
    
    def _assess_complexity(self, text: str) -> str:
        """Assess the complexity of user input.
        
        Args:
            text: User input text
            
        Returns:
            Complexity level (simple, moderate, complex)
        """
        # Factors for complexity
        word_count = len(text.split())
        has_multiple_sentences = len(re.split(r'[.!?]', text)) > 2
        has_technical_terms = any(term in text.lower() for term in 
                                  ['api', 'database', 'algorithm', 'architecture', 
                                   'framework', 'integration'])
        has_multiple_tasks = len(re.findall(r'\band\b|\bthen\b', text.lower())) > 1
        
        complexity_score = 0
        if word_count > 20:
            complexity_score += 1
        if has_multiple_sentences:
            complexity_score += 1
        if has_technical_terms:
            complexity_score += 1
        if has_multiple_tasks:
            complexity_score += 2
        
        if complexity_score >= 3:
            return 'complex'
        elif complexity_score >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _analyze_dialogue_flow(self, user_input: str, assistant_response: str,
                               history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the flow of multi-turn dialogue.
        
        Args:
            user_input: Current user input
            assistant_response: Current assistant response
            history: Conversation history
            
        Returns:
            Dialogue flow analysis
        """
        analysis = {
            'dialogue_state': 'ongoing',
            'topic_shift': False,
            'requires_context': False,
            'references_previous': False
        }
        
        if not history:
            return analysis
        
        # Check for pronouns or references to previous turns
        reference_patterns = [
            r'\b(it|that|this|they|them)\b',
            r'\b(the same|similar|like before)\b',
            r'\b(previous|last|earlier)\b'
        ]
        
        for pattern in reference_patterns:
            if re.search(pattern, user_input.lower()):
                analysis['references_previous'] = True
                analysis['requires_context'] = True
                break
        
        # Check for topic shifts
        if len(history) >= 2:
            prev_entities = history[-1].get('entities', {})
            curr_entities = self._extract_entities(user_input)
            
            # Simple topic shift detection based on entity overlap
            prev_keys = set(prev_entities.keys())
            curr_keys = set(curr_entities.keys())
            
            if prev_keys and curr_keys and not prev_keys.intersection(curr_keys):
                analysis['topic_shift'] = True
        
        return analysis
    
    def _extract_features(self, text: str) -> np.ndarray:
        """Extract features from text for ML processing.
        
        Args:
            text: Input text
            
        Returns:
            Feature vector
        """
        # For now, return empty array - would use TF-IDF in production
        return np.array([])
    
    def _calculate_intent_score(self, user_input: str, patterns: List[str], 
                                context: Dict[str, Any]) -> float:
        """Calculate intent matching score.
        
        Args:
            user_input: User input text
            patterns: Known patterns for this intent
            context: Current context
            
        Returns:
            Confidence score (0-1)
        """
        if not patterns:
            return 0.0
        
        # Simple pattern matching for now
        max_score = 0.0
        input_lower = user_input.lower()
        
        for pattern in patterns:
            pattern_lower = pattern.lower()
            
            # Exact match
            if pattern_lower == input_lower:
                max_score = max(max_score, 1.0)
            # Substring match
            elif pattern_lower in input_lower or input_lower in pattern_lower:
                max_score = max(max_score, 0.7)
            # Word overlap
            else:
                pattern_words = set(pattern_lower.split())
                input_words = set(input_lower.split())
                overlap = len(pattern_words.intersection(input_words))
                if overlap > 0:
                    score = overlap / max(len(pattern_words), len(input_words))
                    max_score = max(max_score, score * 0.5)
        
        # Boost score if context matches
        if context.get('previous_intent') == patterns[0]:
            max_score = min(1.0, max_score * 1.2)
        
        return max_score
    
    def _extract_context_clues(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Extract context clues from user input.
        
        Args:
            text: User input text
            context: Current context
            
        Returns:
            List of context clues
        """
        clues = []
        
        # Time references
        time_patterns = [
            (r'\b(now|immediately|urgent)\b', 'urgent_request'),
            (r'\b(later|tomorrow|next week)\b', 'future_request'),
            (r'\b(yesterday|last|previous)\b', 'past_reference')
        ]
        
        for pattern, clue in time_patterns:
            if re.search(pattern, text.lower()):
                clues.append(clue)
        
        # Scope references
        if re.search(r'\b(all|every|entire)\b', text.lower()):
            clues.append('global_scope')
        elif re.search(r'\b(specific|particular|certain)\b', text.lower()):
            clues.append('specific_scope')
        
        # Continuation indicators
        if re.search(r'\b(also|additionally|furthermore)\b', text.lower()):
            clues.append('continuation')
        
        return clues
    
    def _update_patterns(self, user_input: str, assistant_response: str,
                         analysis: Dict[str, Any], context: Dict[str, Any]):
        """Update learned patterns based on interaction.
        
        Args:
            user_input: User input
            assistant_response: Assistant response
            analysis: Analysis results
            context: Current context
        """
        # Update intent patterns
        if analysis['intent'].confidence > 0.7:
            intent = analysis['intent'].primary_intent
            self.intent_patterns[intent].append(user_input)
            
            # Keep only unique patterns
            self.intent_patterns[intent] = list(set(self.intent_patterns[intent]))
        
        # Create new pattern if significant
        if analysis['detected_patterns'] and context.get('interaction_success', True):
            pattern_id = f"pattern_{len(self.patterns)}"
            pattern = ConversationPattern(
                pattern_id=pattern_id,
                pattern_type=analysis['detected_patterns'][0]['type'],
                frequency=1,
                confidence=analysis['detected_patterns'][0]['confidence'],
                examples=[{
                    'user_input': user_input,
                    'assistant_response': assistant_response,
                    'context': context
                }],
                context_requirements=analysis.get('context_clues', []),
                learned_at=datetime.now(),
                last_seen=datetime.now()
            )
            
            self.patterns[pattern_id] = pattern
        
        # Update metrics
        self.pattern_metrics['total_conversations'] += 1
        if context.get('interaction_success', True):
            self.pattern_metrics['successful_interactions'] += 1
        else:
            self.pattern_metrics['failed_interactions'] += 1
    
    def get_pattern_suggestions(self, user_input: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get pattern-based suggestions for handling user input.
        
        Args:
            user_input: Current user input
            context: Current context
            
        Returns:
            List of suggestions based on learned patterns
        """
        suggestions = []
        
        # Analyze current input
        analysis = self.analyze_conversation_turn(user_input, "", context)
        
        # Find similar patterns
        for pattern_id, pattern in self.patterns.items():
            if pattern.pattern_type in [p['type'] for p in analysis['detected_patterns']]:
                if pattern.success_rate > 0.7:
                    suggestions.append({
                        'pattern_id': pattern_id,
                        'pattern_type': pattern.pattern_type,
                        'confidence': pattern.confidence * pattern.success_rate,
                        'suggested_approach': self._generate_approach(pattern),
                        'examples': pattern.examples[:2]  # Show max 2 examples
                    })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _generate_approach(self, pattern: ConversationPattern) -> str:
        """Generate suggested approach based on pattern.
        
        Args:
            pattern: The conversation pattern
            
        Returns:
            Suggested approach description
        """
        approaches = {
            'task_request': "Execute the requested task with clear status updates",
            'information_query': "Provide detailed information with examples",
            'clarification': "Ask specific questions to understand the requirement",
            'feedback': "Acknowledge feedback and adjust approach accordingly",
            'multi_step': "Break down into steps and execute sequentially"
        }
        
        return approaches.get(pattern.pattern_type, "Handle based on context")
    
    def learn_from_conversation_history(self, history: List[Dict[str, Any]]):
        """Learn patterns from entire conversation history.
        
        Args:
            history: List of conversation turns
        """
        if not history:
            return
        
        # Analyze conversation sequences
        for i in range(len(history) - 1):
            curr_turn = history[i]
            next_turn = history[i + 1]
            
            # Learn transition patterns
            self._learn_transition_pattern(curr_turn, next_turn)
        
        # Analyze overall conversation flow
        self._analyze_conversation_success(history)
        
        # Update dialogue flows
        flow = DialogueFlow(
            turns=history,
            current_topic=self._extract_topic(history[-1]),
            topic_history=self._extract_topic_history(history),
            clarifications_needed=[],
            user_satisfaction=self._estimate_satisfaction(history),
            completion_status='completed'
        )
        
        self.dialogue_flows.append(flow)
        
        # Save learned patterns
        self._save_patterns()
    
    def _learn_transition_pattern(self, curr_turn: Dict[str, Any], next_turn: Dict[str, Any]):
        """Learn patterns from conversation transitions.
        
        Args:
            curr_turn: Current conversation turn
            next_turn: Next conversation turn
        """
        # Extract transition type
        curr_intent = curr_turn.get('intent', 'unknown')
        next_intent = next_turn.get('intent', 'unknown')
        
        transition_key = f"{curr_intent}_to_{next_intent}"
        
        # Track transition frequency
        if not hasattr(self, 'transition_patterns'):
            self.transition_patterns = defaultdict(int)
        
        self.transition_patterns[transition_key] += 1
    
    def _extract_topic(self, turn: Dict[str, Any]) -> str:
        """Extract the main topic from a conversation turn.
        
        Args:
            turn: Conversation turn
            
        Returns:
            Extracted topic
        """
        # Simple topic extraction based on entities and actions
        entities = turn.get('entities', {})
        
        if 'actions' in entities and entities['actions']:
            action = entities['actions'][0]
            if 'files' in entities:
                return f"{action}_files"
            else:
                return action
        
        return 'general_conversation'
    
    def _extract_topic_history(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extract topic progression from conversation history.
        
        Args:
            history: Conversation history
            
        Returns:
            List of topics in order
        """
        topics = []
        for turn in history:
            topic = self._extract_topic(turn)
            if not topics or topic != topics[-1]:
                topics.append(topic)
        return topics
    
    def _estimate_satisfaction(self, history: List[Dict[str, Any]]) -> float:
        """Estimate user satisfaction from conversation history.
        
        Args:
            history: Conversation history
            
        Returns:
            Satisfaction score (0-1)
        """
        if not history:
            return 0.5
        
        # Factors for satisfaction
        positive_signals = 0
        negative_signals = 0
        
        for turn in history:
            sentiment = turn.get('sentiment', {})
            
            # Check sentiment
            if sentiment.get('positive', 0) > 0.5:
                positive_signals += 1
            elif sentiment.get('negative', 0) > 0.5:
                negative_signals += 1
            
            # Check for success indicators
            response = turn.get('assistant_response', '')
            if any(word in response.lower() for word in ['completed', 'done', 'created', 'successful']):
                positive_signals += 1
            elif any(word in response.lower() for word in ['failed', 'error', 'problem', 'issue']):
                negative_signals += 1
        
        # Calculate satisfaction
        total_signals = positive_signals + negative_signals
        if total_signals == 0:
            return 0.7  # Default neutral-positive
        
        return positive_signals / total_signals
    
    def _analyze_conversation_success(self, history: List[Dict[str, Any]]):
        """Analyze overall conversation success.
        
        Args:
            history: Conversation history
        """
        # Update average turns
        if history:
            current_avg = self.pattern_metrics['average_turns_per_conversation']
            total_convs = self.pattern_metrics['total_conversations']
            
            new_avg = (current_avg * (total_convs - 1) + len(history)) / total_convs
            self.pattern_metrics['average_turns_per_conversation'] = new_avg
        
        # Update satisfaction score
        satisfaction = self._estimate_satisfaction(history)
        current_satisfaction = self.pattern_metrics['user_satisfaction_score']
        total_convs = self.pattern_metrics['total_conversations']
        
        new_satisfaction = (current_satisfaction * (total_convs - 1) + satisfaction) / total_convs
        self.pattern_metrics['user_satisfaction_score'] = new_satisfaction
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get insights from learned conversation patterns.
        
        Returns:
            Dictionary of conversation insights
        """
        insights = {
            'metrics': self.pattern_metrics,
            'top_intents': self._get_top_intents(),
            'common_patterns': self._get_common_patterns(),
            'improvement_suggestions': self._generate_improvements(),
            'user_preferences': self._analyze_user_preferences()
        }
        
        return insights
    
    def _get_top_intents(self) -> List[Tuple[str, int]]:
        """Get most common user intents.
        
        Returns:
            List of (intent, count) tuples
        """
        intent_counts = Counter()
        for intent, patterns in self.intent_patterns.items():
            intent_counts[intent] = len(patterns)
        
        return intent_counts.most_common(10)
    
    def _get_common_patterns(self) -> List[Dict[str, Any]]:
        """Get most common conversation patterns.
        
        Returns:
            List of common patterns
        """
        # Sort patterns by frequency
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )
        
        return [{
            'pattern_type': p.pattern_type,
            'frequency': p.frequency,
            'success_rate': p.success_rate,
            'example': p.examples[0] if p.examples else None
        } for p in sorted_patterns[:10]]
    
    def _generate_improvements(self) -> List[str]:
        """Generate improvement suggestions based on patterns.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check for high clarification rate
        if self.pattern_metrics['clarifications_required'] > self.pattern_metrics['total_conversations'] * 0.3:
            suggestions.append("Improve initial query understanding to reduce clarifications")
        
        # Check for low satisfaction
        if self.pattern_metrics['user_satisfaction_score'] < 0.7:
            suggestions.append("Focus on improving response quality and task completion")
        
        # Check for complex conversations
        if self.pattern_metrics['average_turns_per_conversation'] > 5:
            suggestions.append("Optimize task execution to reduce conversation length")
        
        # Check for failed patterns
        failed_patterns = [p for p in self.patterns.values() if p.success_rate < 0.5]
        if failed_patterns:
            suggestions.append(f"Review and improve {len(failed_patterns)} underperforming patterns")
        
        return suggestions
    
    def _analyze_user_preferences(self) -> Dict[str, Any]:
        """Analyze user preferences from patterns.
        
        Returns:
            User preference analysis
        """
        preferences = {
            'preferred_interaction_style': 'concise',  # Default
            'common_tasks': [],
            'preferred_tools': [],
            'communication_patterns': []
        }
        
        # Analyze communication style
        if self.patterns:
            # Check for verbose vs concise preferences
            avg_input_length = np.mean([
                len(p.examples[0]['user_input'].split()) 
                for p in self.patterns.values() 
                if p.examples
            ])
            
            if avg_input_length > 15:
                preferences['preferred_interaction_style'] = 'detailed'
            elif avg_input_length < 7:
                preferences['preferred_interaction_style'] = 'minimal'
        
        # Extract common tasks
        task_patterns = [p for p in self.patterns.values() if p.pattern_type == 'task_request']
        if task_patterns:
            preferences['common_tasks'] = [
                p.examples[0]['user_input'] 
                for p in sorted(task_patterns, key=lambda x: x.frequency, reverse=True)[:5]
                if p.examples
            ]
        
        return preferences
    
    def _save_patterns(self):
        """Save learned patterns to storage."""
        data = {
            'patterns': {
                pid: {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type,
                    'frequency': p.frequency,
                    'confidence': p.confidence,
                    'examples': p.examples,
                    'context_requirements': p.context_requirements,
                    'learned_at': p.learned_at.isoformat(),
                    'last_seen': p.last_seen.isoformat(),
                    'success_rate': p.success_rate,
                    'metadata': p.metadata
                }
                for pid, p in self.patterns.items()
            },
            'intent_patterns': dict(self.intent_patterns),
            'metrics': self.pattern_metrics,
            'transition_patterns': dict(getattr(self, 'transition_patterns', {}))
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_patterns(self):
        """Load patterns from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load patterns
            for pid, p_data in data.get('patterns', {}).items():
                self.patterns[pid] = ConversationPattern(
                    pattern_id=p_data['pattern_id'],
                    pattern_type=p_data['pattern_type'],
                    frequency=p_data['frequency'],
                    confidence=p_data['confidence'],
                    examples=p_data['examples'],
                    context_requirements=p_data['context_requirements'],
                    learned_at=datetime.fromisoformat(p_data['learned_at']),
                    last_seen=datetime.fromisoformat(p_data['last_seen']),
                    success_rate=p_data.get('success_rate', 1.0),
                    metadata=p_data.get('metadata', {})
                )
            
            # Load other data
            # Ensure intent_patterns is a dict (handle case where it might be a list)
            intent_data = data.get('intent_patterns', {})
            if isinstance(intent_data, list):
                # Convert list to dict if needed
                self.intent_patterns = defaultdict(list)
                self.logger.warning("Converting intent_patterns from list to dict format")
            else:
                self.intent_patterns = defaultdict(list, intent_data)
            
            self.pattern_metrics = data.get('metrics', self.pattern_metrics)
            
            # Handle transition_patterns similarly
            transition_data = data.get('transition_patterns', {})
            if isinstance(transition_data, list):
                self.transition_patterns = defaultdict(int)
                self.logger.warning("Converting transition_patterns from list to dict format")
            else:
                self.transition_patterns = defaultdict(int, transition_data)
            
        except Exception as e:
            self.logger.error(f"Failed to load patterns: {e}")