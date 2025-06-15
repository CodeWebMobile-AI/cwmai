"""
Semantic Memory System for Conversational AI

Provides vector-based memory storage and retrieval for intelligent context awareness.
Uses embeddings to find semantically similar past interactions and knowledge.
"""

import json
import numpy as np
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import pickle

# Try to import vector DB libraries
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    
# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

from scripts.http_ai_client import HTTPAIClient


@dataclass
class Memory:
    """Represents a memory in the system."""
    id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    timestamp: datetime
    relevance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }


@dataclass 
class MemoryCluster:
    """Represents a cluster of related memories."""
    topic: str
    memories: List[Memory] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    importance: float = 0.0


class SemanticMemorySystem:
    """Advanced semantic memory system with vector search capabilities."""
    
    def __init__(self, memory_dir: str = "semantic_memory"):
        """Initialize the semantic memory system."""
        self.logger = logging.getLogger(__name__)
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        # Initialize AI client for embeddings if needed
        self.ai_client = HTTPAIClient(enable_round_robin=True)
        
        # Initialize embedding model
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use a small, fast model
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Local embedding model loaded")
            except Exception as e:
                self.logger.warning(f"Could not load embedding model: {e}")
        
        # Initialize vector store
        self.vector_store = None
        if CHROMADB_AVAILABLE:
            try:
                self.vector_store = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(self.memory_dir / "chroma")
                ))
                self.collection = self.vector_store.get_or_create_collection("memories")
                self.logger.info("ChromaDB vector store initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize ChromaDB: {e}")
                
        # Fallback: Simple in-memory storage
        self.memories: Dict[str, Memory] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.memory_clusters: List[MemoryCluster] = []
        
        # Load existing memories
        self._load_memories()
        
    async def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> Memory:
        """Add a new memory to the system.
        
        Args:
            content: The content to remember
            metadata: Additional metadata (type, source, tags, etc.)
            
        Returns:
            The created Memory object
        """
        # Generate ID
        memory_id = self._generate_id(content)
        
        # Generate embedding
        embedding = await self._generate_embedding(content)
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc)
        )
        
        # Store in vector database if available
        if self.vector_store and self.collection and embedding is not None:
            try:
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[content],
                    metadatas=[metadata or {}],
                    ids=[memory_id]
                )
            except Exception as e:
                self.logger.error(f"Error adding to vector store: {e}")
        
        # Store in local cache
        self.memories[memory_id] = memory
        if embedding is not None:
            self.embeddings_cache[memory_id] = embedding
        
        # Update clusters
        await self._update_clusters(memory)
        
        # Persist
        self._save_memories()
        
        return memory
    
    async def search_memories(self, query: str, k: int = 5, 
                            filter_metadata: Dict[str, Any] = None) -> List[Memory]:
        """Search for relevant memories.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant memories sorted by relevance
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        if self.vector_store and self.collection and query_embedding is not None:
            # Use vector database search
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=k,
                    where=filter_metadata
                )
                
                # Convert results to Memory objects
                memories = []
                for i, doc_id in enumerate(results['ids'][0]):
                    if doc_id in self.memories:
                        memory = self.memories[doc_id]
                        memory.relevance_score = 1.0 - results['distances'][0][i]
                        memory.access_count += 1
                        memory.last_accessed = datetime.now(timezone.utc)
                        memories.append(memory)
                
                return memories
                
            except Exception as e:
                self.logger.error(f"Vector search error: {e}")
        
        # Fallback: Simple similarity search
        return await self._simple_similarity_search(query, query_embedding, k, filter_metadata)
    
    async def _simple_similarity_search(self, query: str, query_embedding: Optional[np.ndarray],
                                      k: int, filter_metadata: Dict[str, Any] = None) -> List[Memory]:
        """Simple similarity search using cosine similarity."""
        if query_embedding is None:
            # Fallback to keyword search
            return self._keyword_search(query, k, filter_metadata)
        
        scores = []
        for memory_id, memory in self.memories.items():
            # Apply metadata filter
            if filter_metadata:
                if not all(memory.metadata.get(key) == value for key, value in filter_metadata.items()):
                    continue
            
            # Calculate similarity
            if memory_id in self.embeddings_cache:
                memory_embedding = self.embeddings_cache[memory_id]
                similarity = self._cosine_similarity(query_embedding, memory_embedding)
                scores.append((memory, similarity))
        
        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update access stats and return top k
        results = []
        for memory, score in scores[:k]:
            memory.relevance_score = score
            memory.access_count += 1
            memory.last_accessed = datetime.now(timezone.utc)
            results.append(memory)
        
        return results
    
    def _keyword_search(self, query: str, k: int, filter_metadata: Dict[str, Any] = None) -> List[Memory]:
        """Fallback keyword-based search."""
        query_words = set(query.lower().split())
        scores = []
        
        for memory in self.memories.values():
            # Apply metadata filter
            if filter_metadata:
                if not all(memory.metadata.get(key) == value for key, value in filter_metadata.items()):
                    continue
            
            # Calculate keyword overlap
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words)
            if overlap > 0:
                score = overlap / max(len(query_words), len(memory_words))
                scores.append((memory, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for memory, score in scores[:k]:
            memory.relevance_score = score
            memory.access_count += 1
            memory.last_accessed = datetime.now(timezone.utc)
            results.append(memory)
        
        return results
    
    async def get_context_memories(self, current_context: str, k: int = 3) -> List[Memory]:
        """Get memories relevant to the current context.
        
        Args:
            current_context: The current conversation or task context
            k: Number of memories to retrieve
            
        Returns:
            List of contextually relevant memories
        """
        # Search for relevant memories
        memories = await self.search_memories(current_context, k * 2)
        
        # Filter by recency and relevance
        now = datetime.now(timezone.utc)
        scored_memories = []
        
        for memory in memories:
            # Calculate combined score (relevance + recency)
            age_hours = (now - memory.timestamp).total_seconds() / 3600
            recency_score = 1.0 / (1.0 + age_hours / 24)  # Decay over days
            combined_score = 0.7 * memory.relevance_score + 0.3 * recency_score
            
            scored_memories.append((memory, combined_score))
        
        # Sort by combined score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in scored_memories[:k]]
    
    async def find_related_memories(self, memory_id: str, k: int = 5) -> List[Memory]:
        """Find memories related to a given memory.
        
        Args:
            memory_id: ID of the reference memory
            k: Number of related memories to find
            
        Returns:
            List of related memories
        """
        if memory_id not in self.memories:
            return []
        
        reference_memory = self.memories[memory_id]
        
        # Search using the reference memory's content
        related = await self.search_memories(reference_memory.content, k + 1)
        
        # Remove the reference memory itself
        return [m for m in related if m.id != memory_id][:k]
    
    async def summarize_memory_cluster(self, topic: str) -> str:
        """Generate a summary of memories related to a topic.
        
        Args:
            topic: The topic to summarize
            
        Returns:
            A summary of relevant memories
        """
        # Find relevant memories
        memories = await self.search_memories(topic, k=10)
        
        if not memories:
            return f"No memories found related to '{topic}'"
        
        # Use AI to generate summary
        memory_contents = "\n\n".join([
            f"[{m.timestamp.strftime('%Y-%m-%d')}] {m.content}"
            for m in memories[:5]
        ])
        
        prompt = f"""Summarize these memories related to '{topic}':

{memory_contents}

Provide a concise summary of the key points and patterns."""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        return response.get('content', f"Summary of {len(memories)} memories about '{topic}'")
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_file = self.memory_dir / f"embeddings/{text_hash}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Use local model if available
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                
                # Cache the embedding
                cache_file.parent.mkdir(exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
                
                return embedding
            except Exception as e:
                self.logger.error(f"Local embedding error: {e}")
        
        # Fallback: Use AI to generate embeddings (mock)
        # In production, you'd use an embedding API
        return self._generate_mock_embedding(text)
    
    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """Generate a mock embedding for testing."""
        # Simple hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numbers
        embedding = []
        for i in range(0, len(text_hash), 2):
            value = int(text_hash[i:i+2], 16) / 255.0
            embedding.append(value)
        
        # Pad to standard size (384 for MiniLM)
        while len(embedding) < 384:
            embedding.append(0.0)
        
        return np.array(embedding[:384])
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def _update_clusters(self, new_memory: Memory):
        """Update memory clusters with new memory."""
        # Find best matching cluster
        best_cluster = None
        best_score = 0.0
        
        for cluster in self.memory_clusters:
            if cluster.memories:
                # Compare with cluster centroid or representative memory
                representative = cluster.memories[0]
                if representative.id in self.embeddings_cache and new_memory.embedding is not None:
                    similarity = self._cosine_similarity(
                        self.embeddings_cache[representative.id],
                        new_memory.embedding
                    )
                    if similarity > best_score and similarity > 0.7:  # Threshold
                        best_score = similarity
                        best_cluster = cluster
        
        if best_cluster:
            best_cluster.memories.append(new_memory)
        else:
            # Create new cluster
            topic = new_memory.metadata.get('topic', 'general')
            new_cluster = MemoryCluster(topic=topic, memories=[new_memory])
            self.memory_clusters.append(new_cluster)
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return hashlib.md5(f"{content}{timestamp}".encode()).hexdigest()[:16]
    
    def _save_memories(self):
        """Save memories to disk."""
        # Save memory metadata
        memories_data = {
            memory_id: memory.to_dict()
            for memory_id, memory in self.memories.items()
        }
        
        with open(self.memory_dir / "memories.json", 'w') as f:
            json.dump(memories_data, f, indent=2)
        
        # Save embeddings
        with open(self.memory_dir / "embeddings.pkl", 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
    
    def _load_memories(self):
        """Load memories from disk."""
        memories_file = self.memory_dir / "memories.json"
        embeddings_file = self.memory_dir / "embeddings.pkl"
        
        if memories_file.exists():
            try:
                with open(memories_file, 'r') as f:
                    memories_data = json.load(f)
                
                for memory_id, data in memories_data.items():
                    memory = Memory(
                        id=memory_id,
                        content=data['content'],
                        embedding=None,
                        metadata=data['metadata'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        relevance_score=data.get('relevance_score', 0.0),
                        access_count=data.get('access_count', 0),
                        last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None
                    )
                    self.memories[memory_id] = memory
                
                self.logger.info(f"Loaded {len(self.memories)} memories")
            except Exception as e:
                self.logger.error(f"Error loading memories: {e}")
        
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embeddings_cache)} embeddings")
            except Exception as e:
                self.logger.error(f"Error loading embeddings: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        total_memories = len(self.memories)
        
        # Calculate access patterns
        accessed_memories = [m for m in self.memories.values() if m.access_count > 0]
        avg_access_count = sum(m.access_count for m in accessed_memories) / max(len(accessed_memories), 1)
        
        # Get topic distribution
        topic_counts = {}
        for memory in self.memories.values():
            topic = memory.metadata.get('topic', 'general')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_memories": total_memories,
            "total_clusters": len(self.memory_clusters),
            "accessed_memories": len(accessed_memories),
            "average_access_count": avg_access_count,
            "topic_distribution": topic_counts,
            "vector_store_enabled": self.vector_store is not None,
            "embedding_model": self.embedding_model.__class__.__name__ if self.embedding_model else "mock"
        }