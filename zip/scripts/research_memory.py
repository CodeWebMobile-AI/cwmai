"""
Research Memory System

Tracks researched problems using embeddings to detect similarity
and prevent circular patterns without hardcoded categories.
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
import logging
import hashlib


@dataclass
class ResearchAttempt:
    """Record of a research attempt."""
    problem_statement: str
    problem_fingerprint: str  # Hash for quick comparison
    embedding: List[float]  # Semantic embedding for similarity
    timestamp: str
    success: bool
    validation_score: float
    severity_score: float
    rejection_reason: Optional[str] = None
    project_created: Optional[str] = None
    
    
class ResearchMemory:
    """Manages memory of past research using embeddings for similarity."""
    
    def __init__(self, memory_file: str = "research_memory.json"):
        """Initialize research memory.
        
        Args:
            memory_file: Path to store research memory
        """
        self.memory_file = memory_file
        self.logger = logging.getLogger(__name__)
        self.attempts: List[ResearchAttempt] = []
        self.load_memory()
        
        # Similarity threshold (0-1, where 1 is identical)
        self.similarity_threshold = 0.85
        
    def load_memory(self):
        """Load research memory from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.attempts = [
                        ResearchAttempt(**attempt) for attempt in data.get('attempts', [])
                    ]
                    self.logger.info(f"Loaded {len(self.attempts)} research attempts")
            except Exception as e:
                self.logger.error(f"Error loading research memory: {e}")
                self.attempts = []
                
    def save_memory(self):
        """Save research memory to file."""
        try:
            data = {
                'attempts': [asdict(attempt) for attempt in self.attempts],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving research memory: {e}")
            
    def create_fingerprint(self, problem: str) -> str:
        """Create a unique fingerprint for a problem.
        
        Args:
            problem: Problem statement
            
        Returns:
            Fingerprint hash
        """
        return hashlib.sha256(problem.lower().strip().encode()).hexdigest()[:16]
    
    def create_embedding(self, problem: str, ai_brain) -> List[float]:
        """Create semantic embedding for a problem.
        
        Args:
            problem: Problem statement
            ai_brain: AI brain for generating embeddings
            
        Returns:
            Embedding vector
        """
        try:
            # Use AI to generate a semantic representation
            prompt = f"""
            Convert this problem into a semantic vector representation.
            Return exactly 10 numbers between 0 and 1 that capture:
            1. Industry/domain (2 values)
            2. Problem type (2 values)  
            3. Technical complexity (2 values)
            4. User demographics (2 values)
            5. Solution characteristics (2 values)
            
            Problem: {problem}
            
            Return ONLY the 10 numbers separated by commas, like: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
            """
            
            response = ai_brain.generate_enhanced_response_sync(prompt)
            content = response.get('content', '') if isinstance(response, dict) else str(response)
            
            # Parse the response to get numbers
            numbers = [float(x.strip()) for x in content.strip().split(',') if x.strip()][:10]
            
            # Ensure we have exactly 10 values
            while len(numbers) < 10:
                numbers.append(0.5)
                
            return numbers[:10]
            
        except Exception as e:
            self.logger.error(f"Error creating embedding: {e}")
            # Return a default embedding
            return [0.5] * 10
            
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
            
    def is_similar_to_previous(self, problem: str, embedding: List[float]) -> Tuple[bool, Optional[ResearchAttempt]]:
        """Check if problem is similar to previously researched ones.
        
        Args:
            problem: Problem statement
            embedding: Problem embedding
            
        Returns:
            Tuple of (is_similar, most_similar_attempt)
        """
        fingerprint = self.create_fingerprint(problem)
        
        # First check exact matches by fingerprint
        for attempt in self.attempts:
            if attempt.problem_fingerprint == fingerprint:
                return True, attempt
                
        # Then check semantic similarity
        max_similarity = 0.0
        most_similar = None
        
        for attempt in self.attempts:
            similarity = self.calculate_similarity(embedding, attempt.embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar = attempt
                
        if max_similarity >= self.similarity_threshold:
            return True, most_similar
            
        return False, None
        
    def record_attempt(self, problem: str, embedding: List[float], 
                      success: bool, validation_score: float, severity_score: float,
                      rejection_reason: Optional[str] = None, 
                      project_created: Optional[str] = None):
        """Record a research attempt.
        
        Args:
            problem: Problem statement
            embedding: Problem embedding
            success: Whether the research led to project creation
            validation_score: Validation score achieved
            severity_score: Severity score achieved
            rejection_reason: Why it was rejected (if applicable)
            project_created: Name of created project (if applicable)
        """
        attempt = ResearchAttempt(
            problem_statement=problem,
            problem_fingerprint=self.create_fingerprint(problem),
            embedding=embedding,
            timestamp=datetime.now().isoformat(),
            success=success,
            validation_score=validation_score,
            severity_score=severity_score,
            rejection_reason=rejection_reason,
            project_created=project_created
        )
        
        self.attempts.append(attempt)
        self.save_memory()
        
    def get_success_patterns(self) -> Dict[str, Any]:
        """Analyze successful patterns to guide future research.
        
        Returns:
            Dictionary with success pattern analysis
        """
        successful = [a for a in self.attempts if a.success]
        failed = [a for a in self.attempts if not a.success]
        
        if not successful:
            return {
                'average_validation_score': 0.9,  # Default high threshold
                'average_severity_score': 0.9,
                'success_rate': 0.0,
                'common_rejection_reasons': []
            }
            
        # Calculate average scores for successful attempts
        avg_validation = sum(a.validation_score for a in successful) / len(successful)
        avg_severity = sum(a.severity_score for a in successful) / len(successful)
        
        # Analyze rejection reasons
        rejection_counts = {}
        for attempt in failed:
            if attempt.rejection_reason:
                reason = attempt.rejection_reason.split(':')[0]  # Get main reason
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                
        common_rejections = sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'average_validation_score': avg_validation,
            'average_severity_score': avg_severity,
            'success_rate': len(successful) / len(self.attempts) if self.attempts else 0,
            'common_rejection_reasons': common_rejections,
            'total_attempts': len(self.attempts),
            'successful_projects': [a.project_created for a in successful if a.project_created]
        }
        
    def get_diversity_score(self) -> float:
        """Calculate how diverse the research attempts have been.
        
        Returns:
            Diversity score (0-1, where 1 is most diverse)
        """
        if len(self.attempts) < 2:
            return 1.0  # Maximum diversity when starting
            
        # Calculate average pairwise similarity
        total_similarity = 0
        comparisons = 0
        
        for i in range(len(self.attempts)):
            for j in range(i + 1, len(self.attempts)):
                similarity = self.calculate_similarity(
                    self.attempts[i].embedding,
                    self.attempts[j].embedding
                )
                total_similarity += similarity
                comparisons += 1
                
        if comparisons == 0:
            return 1.0
            
        avg_similarity = total_similarity / comparisons
        # Convert to diversity (inverse of similarity)
        diversity = 1.0 - avg_similarity
        
        return diversity
        
    def suggest_exploration_direction(self, ai_brain) -> str:
        """Suggest a new direction for research based on past attempts.
        
        Args:
            ai_brain: AI brain for generating suggestions
            
        Returns:
            Suggested research direction
        """
        if not self.attempts:
            return "Explore any problem space that interests you"
            
        # Get success patterns
        patterns = self.get_success_patterns()
        diversity = self.get_diversity_score()
        
        # Create a prompt for AI to suggest new direction
        recent_problems = [a.problem_statement for a in self.attempts[-5:]]
        
        prompt = f"""
        Based on our research history, suggest a NEW direction to explore.
        
        Recent attempts: {recent_problems}
        Success rate: {patterns['success_rate']:.1%}
        Diversity score: {diversity:.2f}
        
        Generate a research direction that:
        1. Is different from what we've explored
        2. Has potential for success
        3. Explores new problem spaces
        
        Return a single sentence describing what to research next.
        """
        
        response = ai_brain.generate_enhanced_response_sync(prompt)
        return response.get('content', 'Explore emerging business problems') if isinstance(response, dict) else str(response)
    
    def get_exhausted_domains(self, max_attempts: int = 3) -> Set[str]:
        """Get domains that have been researched too many times.
        
        Args:
            max_attempts: Maximum attempts before considering domain exhausted
            
        Returns:
            Set of exhausted domain names
        """
        domain_counts = {}
        
        # Count attempts per problem domain
        for attempt in self.attempts:
            # Extract domain keywords from problem statement
            words = attempt.problem_statement.lower().split()
            key_words = [w for w in words if len(w) > 5]  # Focus on meaningful words
            
            for word in key_words:
                domain_counts[word] = domain_counts.get(word, 0) + 1
        
        # Find exhausted domains
        exhausted = set()
        for domain, count in domain_counts.items():
            if count >= max_attempts:
                exhausted.add(domain)
                
        return exhausted
    
    def get_recent_attempts(self, hours: int = 24) -> List[ResearchAttempt]:
        """Get all attempts in the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent research attempts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for attempt in self.attempts:
            attempt_time = datetime.fromisoformat(attempt.timestamp)
            if attempt_time > cutoff:
                recent.append(attempt)
                
        return recent
    
    def get_unexplored_sectors(self) -> List[str]:
        """Get sectors that haven't been explored yet.
        
        Returns:
            List of unexplored business sectors (empty list since we don't predefine sectors)
        """
        # We don't hardcode sectors - everything emerges organically
        # This method returns empty list to indicate no predefined sectors
        # The AI will discover new sectors through research
        return []
        
    def get_unexplored_embedding_space(self) -> List[float]:
        """Find an unexplored region in the embedding space.
        
        Returns:
            Embedding vector pointing to unexplored space
        """
        if not self.attempts:
            # Return a random embedding if no attempts yet
            return [0.5] * 10
            
        # Find the centroid of all attempts
        embeddings = np.array([a.embedding for a in self.attempts])
        centroid = np.mean(embeddings, axis=0)
        
        # Find the direction with maximum variance (least explored)
        variances = np.var(embeddings, axis=0)
        max_var_idx = np.argmax(variances)
        
        # Create new embedding pointing away from centroid in high variance direction
        new_embedding = centroid.copy()
        new_embedding[max_var_idx] = 1.0 - new_embedding[max_var_idx]  # Flip the value
        
        # Normalize to ensure valid range
        new_embedding = np.clip(new_embedding, 0, 1)
        
        return new_embedding.tolist()
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about research patterns.
        
        Returns:
            Dictionary with analytics data
        """
        total_attempts = len(self.attempts)
        
        if total_attempts == 0:
            return {
                "total_attempts": 0,
                "domains_explored": 0,
                "average_validation_score": 0,
                "average_severity_score": 0,
                "success_rate": 0,
                "exhausted_domains": [],
                "unexplored_sectors": [],
                "most_rejected_domains": []
            }
        
        # Get unique domains
        domains = set()
        for attempt in self.attempts:
            # Extract domain from problem statement (simplified)
            words = attempt.problem_statement.lower().split()
            domains.update(words)
        
        # Calculate average scores
        avg_validation = sum(a.validation_score for a in self.attempts) / total_attempts
        avg_severity = sum(a.severity_score for a in self.attempts) / total_attempts
        
        # Count successful attempts
        successful = len([a for a in self.attempts if a.success])
        
        # Most rejected problems
        rejection_counts = {}
        for attempt in self.attempts:
            if attempt.rejection_reason:
                key = attempt.problem_statement[:50]
                rejection_counts[key] = rejection_counts.get(key, 0) + 1
                
        most_rejected = sorted(rejection_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_attempts": total_attempts,
            "domains_explored": len(domains),
            "average_validation_score": avg_validation,
            "average_severity_score": avg_severity,
            "success_rate": successful / total_attempts if total_attempts > 0 else 0,
            "exhausted_domains": list(self.get_exhausted_domains()),
            "unexplored_sectors": [],  # No predefined sectors - everything emerges organically
            "most_rejected_domains": most_rejected
        }