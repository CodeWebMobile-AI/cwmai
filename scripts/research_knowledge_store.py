"""
Research Knowledge Store - Persistent storage system for CWMAI research insights.

This module provides intelligent storage, retrieval, and management of research data
with deduplication, expiration, and quality tracking capabilities.
"""

import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import shutil
from research_scheduler import ResearchJSONEncoder


class ResearchKnowledgeStore:
    """Persistent storage system with intelligent retrieval and learning."""
    
    def __init__(self, storage_path: str = "research_knowledge"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize storage structure
        self._initialize_storage_structure()
        
        # Load or create indices
        self.index = self._load_or_create_index()
        self.hash_index = self._load_or_create_hash_index()
        self.metadata = self._load_or_create_metadata()
        
        # Cache for frequently accessed research
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _initialize_storage_structure(self):
        """Create necessary subdirectories if they don't exist."""
        subdirs = [
            "raw_research/task_performance",
            "raw_research/claude_interactions",
            "raw_research/multi_agent_coordination",
            "raw_research/outcome_patterns",
            "raw_research/portfolio_management",
            "processed_insights/patterns",
            "processed_insights/recommendations",
            "processed_insights/success_factors",
            "processed_insights/failure_analysis",
            "knowledge_graph",
            "metadata",
            "backups"
        ]
        
        for subdir in subdirs:
            (self.storage_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def _load_or_create_index(self) -> Dict:
        """Load or create the main research index."""
        index_path = self.storage_path / "metadata" / "research_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return {
            "research_entries": {},
            "category_index": {},
            "timestamp_index": {},
            "quality_index": {}
        }
    
    def _load_or_create_hash_index(self) -> Dict:
        """Load or create content hash index for deduplication."""
        hash_path = self.storage_path / "metadata" / "hash_index.json"
        if hash_path.exists():
            with open(hash_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_or_create_metadata(self) -> Dict:
        """Load or create metadata tracking."""
        metadata_path = self.storage_path / "metadata" / "quality_scores.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {
            "quality_scores": {},
            "usage_frequency": {},
            "value_delivered": {},
            "expiration_dates": {}
        }
    
    def _save_indices(self):
        """Save all indices to disk."""
        # Save main index
        index_path = self.storage_path / "metadata" / "research_index.json"
        with open(index_path, 'w') as f:
            json.dump(self.index, f, indent=2, cls=ResearchJSONEncoder)
        
        # Save hash index
        hash_path = self.storage_path / "metadata" / "hash_index.json"
        with open(hash_path, 'w') as f:
            json.dump(self.hash_index, f, indent=2, cls=ResearchJSONEncoder)
        
        # Save metadata
        metadata_path = self.storage_path / "metadata" / "quality_scores.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=ResearchJSONEncoder)
    
    def _calculate_content_hash(self, content: Dict) -> str:
        """Calculate hash of research content for deduplication."""
        # Create a canonical representation, handling non-serializable objects
        def make_serializable(obj):
            if hasattr(obj, '__dict__'):
                return str(obj)
            elif hasattr(obj, 'value'):  # Enum objects
                return obj.value
            else:
                return str(obj)
        
        try:
            canonical = json.dumps(content, sort_keys=True, default=make_serializable)
        except (TypeError, ValueError):
            # Fallback to string representation
            canonical = str(content)
        
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _make_json_serializable(self, obj):
        """Make objects JSON serializable."""
        if hasattr(obj, '__dict__'):
            return str(obj)
        elif hasattr(obj, 'value'):  # Enum objects
            return obj.value
        else:
            return str(obj)
    
    def store_research(self, research_type: str, content: Dict, 
                      quality_score: float = None) -> str:
        """
        Store research with deduplication and metadata.
        
        Args:
            research_type: Category of research (e.g., 'task_performance')
            content: The research content to store
            quality_score: Optional quality score (0-1)
            
        Returns:
            research_id: Unique identifier for the stored research
        """
        # Check for duplicates
        content_hash = self._calculate_content_hash(content)
        if content_hash in self.hash_index:
            existing_id = self.hash_index[content_hash]
            # Update usage frequency
            self.metadata["usage_frequency"][existing_id] = \
                self.metadata["usage_frequency"].get(existing_id, 0) + 1
            self._save_indices()
            return existing_id
        
        # Generate unique ID
        timestamp = datetime.now().isoformat()
        research_id = f"{research_type}_{int(time.time() * 1000)}"
        
        # Prepare research entry
        research_entry = {
            "id": research_id,
            "type": research_type,
            "content": content,
            "timestamp": timestamp,
            "hash": content_hash,
            "quality_score": quality_score or 0.5,
            "usage_count": 1,
            "last_accessed": timestamp
        }
        
        # Determine storage path based on research type
        # Map research areas to appropriate subdirectories
        type_mapping = {
            "task_performance": "task_performance",
            "claude_interactions": "claude_interactions",
            "multi_agent_coordination": "multi_agent_coordination",
            "outcome_patterns": "outcome_patterns",
            "portfolio_management": "portfolio_management",
            # Map new research types to appropriate folders
            "innovation": "multi_agent_coordination",
            "efficiency": "task_performance",
            "growth": "portfolio_management",
            "strategic_planning": "portfolio_management",
            "continuous_improvement": "outcome_patterns",
            "adaptive_learning": "outcome_patterns",
            "pattern_learning": "outcome_patterns",
            "critical_performance": "task_performance",
            "general": "claude_interactions"
        }
        
        # Get mapped directory or default to research type
        mapped_dir = type_mapping.get(research_type, research_type)
        
        # If mapped directory is one of our subdirectories, use it
        if mapped_dir in ["task_performance", "claude_interactions", 
                         "multi_agent_coordination", "outcome_patterns", 
                         "portfolio_management"]:
            file_path = self.storage_path / "raw_research" / mapped_dir / f"{research_id}.json"
        else:
            # Otherwise store in raw_research root
            file_path = self.storage_path / "raw_research" / f"{research_id}.json"
        
        # Save research content
        with open(file_path, 'w') as f:
            json.dump(research_entry, f, indent=2, cls=ResearchJSONEncoder)
        
        # Update indices
        self.index["research_entries"][research_id] = {
            "path": str(file_path),
            "type": research_type,
            "timestamp": timestamp,
            "quality_score": quality_score or 0.5
        }
        
        # Update category index
        if research_type not in self.index["category_index"]:
            self.index["category_index"][research_type] = []
        self.index["category_index"][research_type].append(research_id)
        
        # Update hash index
        self.hash_index[content_hash] = research_id
        
        # Update metadata
        self.metadata["quality_scores"][research_id] = quality_score or 0.5
        self.metadata["usage_frequency"][research_id] = 1
        self.metadata["value_delivered"][research_id] = 0
        
        # Set expiration (30 days default)
        expiration = (datetime.now() + timedelta(days=30)).isoformat()
        self.metadata["expiration_dates"][research_id] = expiration
        
        self._save_indices()
        return research_id
    
    def retrieve_research(self, research_id: str = None, 
                         research_type: str = None,
                         min_quality: float = None,
                         limit: int = None) -> List[Dict]:
        """
        Retrieve research based on various criteria.
        
        Args:
            research_id: Specific research ID to retrieve
            research_type: Filter by research type
            min_quality: Minimum quality score filter
            limit: Maximum number of results
            
        Returns:
            List of research entries matching criteria
        """
        results = []
        
        # Check cache first
        cache_key = f"{research_id}_{research_type}_{min_quality}_{limit}"
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        
        self.cache_misses += 1
        
        # Single ID retrieval
        if research_id:
            if research_id in self.index["research_entries"]:
                entry_info = self.index["research_entries"][research_id]
                with open(entry_info["path"], 'r') as f:
                    entry = json.load(f)
                    # Update access metadata
                    entry["last_accessed"] = datetime.now().isoformat()
                    self.metadata["usage_frequency"][research_id] = \
                        self.metadata["usage_frequency"].get(research_id, 0) + 1
                    results = [entry]
        
        # Type-based retrieval
        elif research_type:
            if research_type in self.index["category_index"]:
                for rid in self.index["category_index"][research_type]:
                    if rid in self.index["research_entries"]:
                        entry_info = self.index["research_entries"][rid]
                        
                        # Apply quality filter
                        if min_quality and entry_info["quality_score"] < min_quality:
                            continue
                        
                        with open(entry_info["path"], 'r') as f:
                            entry = json.load(f)
                            results.append(entry)
                        
                        if limit and len(results) >= limit:
                            break
        
        # Sort by quality and recency
        results.sort(key=lambda x: (x.get("quality_score", 0), x["timestamp"]), 
                    reverse=True)
        
        # Cache results
        self.cache[cache_key] = results
        
        # Limit cache size
        if len(self.cache) > 100:
            self.cache = {}
        
        self._save_indices()
        return results
    
    def search_research(self, query: str, search_in: List[str] = None) -> List[Dict]:
        """
        Search research content for specific terms.
        
        Args:
            query: Search query
            search_in: List of fields to search in
            
        Returns:
            List of matching research entries
        """
        results = []
        query_lower = query.lower()
        
        if not search_in:
            search_in = ["content", "type", "id"]
        
        for research_id, entry_info in self.index["research_entries"].items():
            try:
                with open(entry_info["path"], 'r') as f:
                    entry = json.load(f)
                    
                    # Search in specified fields
                    for field in search_in:
                        if field in entry:
                            field_content = json.dumps(entry[field]).lower()
                            if query_lower in field_content:
                                results.append(entry)
                                break
                                
            except Exception as e:
                print(f"Error searching {research_id}: {e}")
                continue
        
        return results
    
    def update_quality_score(self, research_id: str, new_score: float):
        """Update quality score based on outcome."""
        if research_id in self.metadata["quality_scores"]:
            # Weighted average with existing score
            old_score = self.metadata["quality_scores"][research_id]
            self.metadata["quality_scores"][research_id] = \
                (old_score * 0.7 + new_score * 0.3)
            
            # Update in index
            if research_id in self.index["research_entries"]:
                self.index["research_entries"][research_id]["quality_score"] = \
                    self.metadata["quality_scores"][research_id]
            
            self._save_indices()
    
    def record_value_delivered(self, research_id: str, value: float):
        """Record value delivered by research."""
        if research_id in self.metadata["value_delivered"]:
            self.metadata["value_delivered"][research_id] += value
        else:
            self.metadata["value_delivered"][research_id] = value
        
        # Boost quality score for high-value research
        if value > 0:
            current_quality = self.metadata["quality_scores"].get(research_id, 0.5)
            new_quality = min(1.0, current_quality + value * 0.1)
            self.update_quality_score(research_id, new_quality)
    
    def cleanup_expired_research(self):
        """Remove expired research entries."""
        current_time = datetime.now()
        removed_count = 0
        
        for research_id, expiration in list(self.metadata["expiration_dates"].items()):
            exp_time = datetime.fromisoformat(expiration)
            if current_time > exp_time:
                # Check if it's high-value research
                if self.metadata.get("value_delivered", {}).get(research_id, 0) > 10:
                    # Extend expiration for high-value research
                    new_expiration = (current_time + timedelta(days=90)).isoformat()
                    self.metadata["expiration_dates"][research_id] = new_expiration
                    continue
                
                # Remove research
                if research_id in self.index["research_entries"]:
                    entry_info = self.index["research_entries"][research_id]
                    
                    # Delete file
                    try:
                        os.remove(entry_info["path"])
                    except:
                        pass
                    
                    # Remove from indices
                    del self.index["research_entries"][research_id]
                    
                    # Remove from category index
                    for category, ids in self.index["category_index"].items():
                        if research_id in ids:
                            ids.remove(research_id)
                    
                    # Remove from metadata
                    for metadata_dict in [self.metadata["quality_scores"],
                                         self.metadata["usage_frequency"],
                                         self.metadata["value_delivered"],
                                         self.metadata["expiration_dates"]]:
                        if research_id in metadata_dict:
                            del metadata_dict[research_id]
                    
                    removed_count += 1
        
        if removed_count > 0:
            self._save_indices()
            print(f"Cleaned up {removed_count} expired research entries")
    
    def backup_research(self):
        """Create backup of all research data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.storage_path / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all research files
        for category in ["raw_research", "processed_insights", "knowledge_graph"]:
            src = self.storage_path / category
            dst = backup_dir / category
            if src.exists():
                shutil.copytree(src, dst)
        
        # Copy metadata
        metadata_src = self.storage_path / "metadata"
        metadata_dst = backup_dir / "metadata"
        if metadata_src.exists():
            shutil.copytree(metadata_src, metadata_dst)
        
        print(f"Backup created at: {backup_dir}")
    
    def get_statistics(self) -> Dict:
        """Get storage statistics."""
        total_entries = len(self.index["research_entries"])
        
        stats = {
            "total_entries": total_entries,
            "entries_by_type": {},
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "total_value_delivered": sum(self.metadata["value_delivered"].values()),
            "average_quality_score": sum(self.metadata["quality_scores"].values()) / max(1, total_entries),
            "storage_size_mb": sum(
                os.path.getsize(f) for f in self.storage_path.rglob("*.json")
            ) / 1024 / 1024
        }
        
        # Count by type
        for category, ids in self.index["category_index"].items():
            stats["entries_by_type"][category] = len(ids)
        
        return stats
    
    def get_insights(self, research_type: str, min_quality: float = 0.6) -> List[Dict]:
        """
        Get insights for a specific research type.
        
        Args:
            research_type: Type of research to get insights for
            min_quality: Minimum quality score for insights
            
        Returns:
            List of insight dictionaries
        """
        # First, try to get processed insights
        insights = []
        
        # Check for processed insights in the appropriate directory
        insights_dir = self.storage_path / "processed_insights"
        type_mapping = {
            "continuous_improvement": "recommendations",
            "pattern_learning": "patterns",
            "success_analysis": "success_factors",
            "failure_analysis": "failure_analysis"
        }
        
        mapped_dir = type_mapping.get(research_type, "recommendations")
        specific_dir = insights_dir / mapped_dir
        
        # Get insights from files
        if specific_dir.exists():
            for file_path in specific_dir.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and data.get("quality_score", 0) >= min_quality:
                            insights.append({
                                "type": research_type,
                                "data": data.get("content", data),
                                "quality_score": data.get("quality_score", 0.7),
                                "timestamp": data.get("timestamp", datetime.now().isoformat())
                            })
                except Exception as e:
                    print(f"Error loading insight from {file_path}: {e}")
        
        # If no processed insights, get raw research
        if not insights:
            raw_research = self.retrieve_research(
                research_type=research_type,
                min_quality=min_quality,
                limit=10
            )
            
            # Transform raw research into insights format
            for research in raw_research:
                insights.append({
                    "type": research_type,
                    "data": research.get("content", {}),
                    "quality_score": research.get("quality_score", 0.5),
                    "timestamp": research.get("timestamp", datetime.now().isoformat())
                })
        
        # Sort by quality score
        insights.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        return insights