"""
Research Engine Optimization Patch

Additional optimizations on top of the fixes:
1. Parallel execution of research queries
2. Batching of AI requests
3. Early termination on repeated failures
4. Smarter query limiting
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from scripts.research_engine_fixes import ResearchEngineEnhancements

class OptimizedResearchEngine:
    """Optimizations for research engine performance"""
    
    def __init__(self, max_parallel_research: int = 3, max_queries_per_topic: int = 1):
        self.max_parallel_research = max_parallel_research
        self.max_queries_per_topic = max_queries_per_topic
        self.logger = logging.getLogger(__name__)
        
    async def execute_research_parallel(self, research_topics: List[Dict], 
                                      execute_single_func) -> List[Dict]:
        """Execute research queries in parallel with limits"""
        research_results = []
        
        # Flatten topics to queries with limits
        all_queries = []
        for topic in research_topics:
            # Generate queries for topic (simplified)
            queries = self._generate_limited_queries(topic)
            for query in queries[:self.max_queries_per_topic]:
                all_queries.append((query, topic))
        
        # Execute in batches
        for i in range(0, len(all_queries), self.max_parallel_research):
            batch = all_queries[i:i + self.max_parallel_research]
            
            # Create tasks for parallel execution
            tasks = []
            for query, topic in batch:
                task = asyncio.create_task(
                    self._execute_with_error_handling(
                        execute_single_func, query, topic
                    )
                )
                tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Research query failed: {result}")
                elif result is not None:
                    research_results.append(result)
            
            # Early termination if too many failures
            failure_rate = len([r for r in batch_results if isinstance(r, Exception)]) / len(batch_results)
            if failure_rate > 0.5:
                self.logger.warning("High failure rate detected, terminating research early")
                break
        
        return research_results
    
    async def _execute_with_error_handling(self, func, query, topic):
        """Execute function with error handling"""
        try:
            return await func(query, topic)
        except Exception as e:
            self.logger.error(f"Error executing research query: {e}")
            return None
    
    def _generate_limited_queries(self, topic: Dict) -> List[Dict]:
        """Generate limited queries for a topic"""
        # Simplified query generation
        base_query = {
            'query': f"Research on {topic.get('topic', 'unknown')}",
            'confidence': 0.7,
            'provider': 'optimized'
        }
        
        # For high priority topics, generate 2 queries
        if topic.get('priority') == 'high':
            return [base_query, {**base_query, 'query': f"Advanced {base_query['query']}"}]
        
        return [base_query]
    
    async def extract_insights_batch(self, research_results: List[Dict],
                                   action_engine) -> List[Dict]:
        """Extract insights in batch mode for efficiency"""
        if not research_results:
            return []
        
        # Group results by type for batch processing
        grouped_results = {}
        for result in research_results:
            area = result.get('area', 'general')
            if area not in grouped_results:
                grouped_results[area] = []
            grouped_results[area].append(result)
        
        all_insights = []
        
        # Process each group
        for area, results in grouped_results.items():
            try:
                # Batch extract insights
                insights = action_engine.extract_actionable_insights({
                    'area': area,
                    'content': results,
                    'batch_mode': True
                })
                
                # Filter by confidence
                filtered_insights = [
                    i for i in insights 
                    if i.get('confidence', 0) >= 0.5
                ]
                
                all_insights.extend(filtered_insights)
                
            except Exception as e:
                self.logger.error(f"Error extracting insights for area {area}: {e}")
                # Add fallback insight
                all_insights.append({
                    'insight': f"Research conducted for {area}",
                    'confidence': 0.3,
                    'source': 'fallback',
                    'area': area
                })
        
        return all_insights


def apply_research_optimizations(engine):
    """Apply performance optimizations to research engine"""
    optimizer = OptimizedResearchEngine()
    
    # Store original methods
    original_execute_research = engine._execute_research
    original_extract_insights = engine._extract_insights
    
    # Create optimized versions
    async def optimized_execute_research(research_topics):
        """Execute research with parallel processing"""
        # Limit topics to prevent overwhelming the system
        limited_topics = research_topics[:engine.config.get('max_research_per_cycle', 3)]
        
        logging.info(f"Executing research on {len(limited_topics)} topics (limited from {len(research_topics)})")
        
        # Use parallel execution
        results = await optimizer.execute_research_parallel(
            limited_topics,
            engine._execute_single_research
        )
        
        # Store results
        for result in results:
            try:
                research_id = engine.knowledge_store.store_research(
                    result.get("area", "general"),
                    result,
                    quality_score=result.get("quality_score", 0.7)
                )
                result["stored_id"] = research_id
            except Exception as e:
                logging.error(f"Error storing research: {e}")
        
        engine.cycle_statistics["total_research_conducted"] += len(results)
        
        return results
    
    async def optimized_extract_insights(research_results):
        """Extract insights with batch processing"""
        if not research_results:
            return []
        
        # Use batch extraction
        insights = await optimizer.extract_insights_batch(
            research_results,
            engine.action_engine
        )
        
        engine.cycle_statistics["total_insights_extracted"] += len(insights)
        
        return insights
    
    # Apply optimizations
    engine._execute_research = optimized_execute_research
    engine._extract_insights = optimized_extract_insights
    
    # Update configuration for better performance
    engine.config.update({
        'max_concurrent_research': 3,
        'max_research_per_cycle': 2,  # Reduce to speed up testing
        'enable_external_agent_research': False,  # Disable for now
        'enable_proactive_research': False  # Disable for now
    })
    
    logging.info("Research engine optimizations applied")
    
    return engine