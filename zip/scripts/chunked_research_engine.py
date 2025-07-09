"""
Chunked Research Engine - Breaks down research into smaller pieces to avoid context limits

This module implements a chunked approach to market research, processing smaller
batches and aggregating results to avoid token/context window limitations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os
from datetime import datetime

from scripts.ai_brain import IntelligentAIBrain
from scripts.json_response_parser import JSONResponseParser


@dataclass
class ResearchChunk:
    """Represents a chunk of research work"""
    chunk_id: int
    sectors: List[str]
    status: str = "pending"  # pending, processing, completed, failed
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0


class ChunkedResearchEngine:
    """Performs market research in smaller chunks to avoid context limits"""
    
    def __init__(self, 
                 chunk_size: int = 3,  # Process 3 sectors at a time
                 max_problems_per_chunk: int = 2,  # Limit problems per chunk
                 max_retries: int = 2):
        """
        Initialize chunked research engine.
        
        Args:
            chunk_size: Number of sectors to process per chunk
            max_problems_per_chunk: Maximum problems to discover per chunk
            max_retries: Maximum retry attempts for failed chunks
        """
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.max_problems_per_chunk = max_problems_per_chunk
        self.max_retries = max_retries
        self.ai_brain = IntelligentAIBrain()
        self.parser = JSONResponseParser()
        
        # State tracking
        self.chunks: List[ResearchChunk] = []
        self.aggregated_results = {
            "explored_sectors": [],
            "discovered_problems": [],
            "validated_opportunities": [],
            "market_insights": {}
        }
    
    async def conduct_chunked_research(self, 
                                     focus_areas: List[str],
                                     save_progress: bool = True) -> Dict[str, Any]:
        """
        Conduct research in chunks to avoid context limits.
        
        Args:
            focus_areas: List of market sectors to research
            save_progress: Whether to save progress between chunks
            
        Returns:
            Aggregated research results
        """
        print(f"🔬 Starting Chunked Research Engine")
        print(f"📊 Processing {len(focus_areas)} sectors in chunks of {self.chunk_size}")
        
        # Step 1: Create chunks
        self._create_chunks(focus_areas)
        print(f"📦 Created {len(self.chunks)} chunks")
        
        # Step 2: Process each chunk
        for chunk in self.chunks:
            await self._process_chunk(chunk)
            
            # Save progress if enabled
            if save_progress:
                self._save_progress()
            
            # Small delay between chunks to avoid rate limits
            await asyncio.sleep(2)
        
        # Step 3: Aggregate and rank results
        final_results = await self._aggregate_and_rank_results()
        
        return final_results
    
    def _create_chunks(self, focus_areas: List[str]):
        """Divide focus areas into manageable chunks."""
        for i in range(0, len(focus_areas), self.chunk_size):
            chunk_sectors = focus_areas[i:i + self.chunk_size]
            chunk = ResearchChunk(
                chunk_id=len(self.chunks),
                sectors=chunk_sectors
            )
            self.chunks.append(chunk)
    
    async def _process_chunk(self, chunk: ResearchChunk):
        """Process a single chunk of research."""
        print(f"\n🔄 Processing chunk {chunk.chunk_id + 1}/{len(self.chunks)}")
        print(f"   Sectors: {', '.join(chunk.sectors)}")
        
        chunk.status = "processing"
        
        try:
            # Phase 1: Explore sectors (lightweight)
            sector_insights = await self._explore_sectors_lightweight(chunk.sectors)
            
            # Phase 2: Discover problems (limited)
            problems = await self._discover_problems_limited(
                sector_insights, 
                limit=self.max_problems_per_chunk
            )
            
            # Phase 3: Quick validation
            validated = await self._quick_validate_problems(problems)
            
            # Store results
            chunk.results = {
                "sector_insights": sector_insights,
                "problems": problems,
                "validated": validated
            }
            chunk.status = "completed"
            
            # Update aggregated results
            self._update_aggregated_results(chunk)
            
            print(f"✅ Chunk {chunk.chunk_id + 1} completed: {len(problems)} problems, {len(validated)} validated")
            
        except Exception as e:
            chunk.error = str(e)
            chunk.status = "failed"
            chunk.retry_count += 1
            
            print(f"❌ Chunk {chunk.chunk_id + 1} failed: {e}")
            
            # Retry if under limit
            if chunk.retry_count < self.max_retries:
                print(f"🔄 Retrying chunk {chunk.chunk_id + 1} (attempt {chunk.retry_count + 1})")
                await asyncio.sleep(5)  # Wait before retry
                await self._process_chunk(chunk)
    
    async def _explore_sectors_lightweight(self, sectors: List[str]) -> Dict[str, Any]:
        """Lightweight sector exploration with minimal output."""
        prompt = f"""
        Analyze these {len(sectors)} market sectors and identify the TOP opportunity in each.
        
        Sectors: {json.dumps(sectors)}
        
        For each sector, provide ONLY:
        1. biggest_problem: One sentence describing the main pain point
        2. market_size: Approximate TAM in one line
        3. opportunity_score: 1-10
        
        Return a compact JSON object with sector names as keys.
        Keep responses extremely brief to avoid token limits.
        
        Example format:
        {{
            "Health and wellness": {{
                "biggest_problem": "Lack of personalized health tracking",
                "market_size": "$50B TAM",
                "opportunity_score": 8
            }}
        }}
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
        return self.parser.parse(response.get('content', ''), expected_type='object') or {}
    
    async def _discover_problems_limited(self, 
                                       sector_insights: Dict[str, Any], 
                                       limit: int) -> List[Dict[str, Any]]:
        """Discover limited number of problems to stay within context."""
        prompt = f"""
        Based on these sector insights, identify the {limit} BEST problems to solve.
        
        Sector insights: {json.dumps(sector_insights)}
        
        For each problem provide ONLY these fields in a JSON array:
        - problem_id: P1, P2, etc.
        - title: One line problem statement
        - impact: $ value or time saved
        - users: Number affected
        - solution_hint: One line solution idea
        - score: 1-10
        
        Keep it extremely concise. No explanations.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
        return self.parser.parse(response.get('content', ''), expected_type='array') or []
    
    async def _quick_validate_problems(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Quick validation without extensive research."""
        validated = []
        
        for problem in problems:
            prompt = f"""
            Quick validation for this problem:
            {json.dumps(problem)}
            
            Provide a brief JSON validation:
            {{
                "viable": true/false,
                "reason": "One line reason",
                "revenue_potential": "$X-Y MRR",
                "confidence": 0-100
            }}
            
            Be extremely concise.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
            validation = self.parser.parse(response.get('content', ''), expected_type='object')
            
            if validation and validation.get('viable') and validation.get('confidence', 0) > 70:
                problem['validation'] = validation
                validated.append(problem)
        
        return validated
    
    def _update_aggregated_results(self, chunk: ResearchChunk):
        """Update aggregated results with chunk data."""
        if chunk.results:
            self.aggregated_results["explored_sectors"].extend(chunk.sectors)
            self.aggregated_results["discovered_problems"].extend(
                chunk.results.get("problems", [])
            )
            self.aggregated_results["validated_opportunities"].extend(
                chunk.results.get("validated", [])
            )
            self.aggregated_results["market_insights"].update(
                chunk.results.get("sector_insights", {})
            )
    
    async def _aggregate_and_rank_results(self) -> Dict[str, Any]:
        """Aggregate all chunk results and select top opportunities."""
        print("\n🏆 Aggregating and ranking all results...")
        
        # Get all validated opportunities
        all_opportunities = self.aggregated_results["validated_opportunities"]
        print(f"📊 Total validated opportunities: {len(all_opportunities)}")
        
        if not all_opportunities:
            return {
                "success": False,
                "message": "No validated opportunities found",
                "results": self.aggregated_results
            }
        
        # Rank by score
        sorted_opps = sorted(
            all_opportunities, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )[:5]  # Top 5
        
        # Final enhancement of top opportunities
        enhanced_opportunities = []
        for opp in sorted_opps:
            enhanced = await self._enhance_opportunity(opp)
            if enhanced:
                enhanced_opportunities.append(enhanced)
        
        return {
            "success": True,
            "top_opportunities": enhanced_opportunities,
            "stats": {
                "sectors_analyzed": len(self.aggregated_results["explored_sectors"]),
                "problems_discovered": len(self.aggregated_results["discovered_problems"]),
                "opportunities_validated": len(all_opportunities),
                "chunks_processed": len([c for c in self.chunks if c.status == "completed"])
            },
            "full_results": self.aggregated_results
        }
    
    async def _enhance_opportunity(self, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhance a single opportunity with more details."""
        prompt = f"""
        Enhance this validated opportunity with implementation details:
        {json.dumps(opportunity)}
        
        Add these specific details in JSON format:
        {{
            "solution_name": "Catchy product name",
            "tagline": "One line value prop",
            "mvp_features": ["feature1", "feature2", "feature3"],
            "tech_stack": ["tech1", "tech2"],
            "go_to_market": "Brief GTM strategy",
            "pricing_tiers": {{"tier": "price"}},
            "3_month_milestones": ["m1", "m2", "m3"]
        }}
        
        Keep it concise but actionable.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
        enhancement = self.parser.parse(response.get('content', ''), expected_type='object')
        
        if enhancement:
            opportunity.update(enhancement)
            return opportunity
        return None
    
    def _save_progress(self):
        """Save current progress to file."""
        progress_file = "chunked_research_progress.json"
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "sectors": c.sectors,
                    "status": c.status,
                    "error": c.error,
                    "retry_count": c.retry_count
                }
                for c in self.chunks
            ],
            "aggregated_results": self.aggregated_results
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"💾 Progress saved to {progress_file}")
    
    def load_progress(self, progress_file: str = "chunked_research_progress.json") -> bool:
        """Load previous progress if available."""
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            self.aggregated_results = progress_data.get("aggregated_results", {})
            print(f"📂 Loaded progress from {progress_file}")
            return True
        return False


async def run_chunked_research():
    """Run chunked research with all default sectors."""
    from scripts.gemini_deep_research import GeminiDeepResearch
    
    # Get default sectors
    default_sectors = GeminiDeepResearch.DEFAULT_FOCUS_AREAS
    
    # Create chunked engine
    engine = ChunkedResearchEngine(
        chunk_size=3,  # Process 3 sectors at a time
        max_problems_per_chunk=2  # Find 2 problems per chunk
    )
    
    # Run research
    results = await engine.conduct_chunked_research(default_sectors)
    
    # Save final results
    output_file = f"chunked_research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Research complete! Results saved to {output_file}")
    
    # Display summary
    if results.get("success"):
        print(f"\n🏆 Top Opportunities Found:")
        for i, opp in enumerate(results.get("top_opportunities", []), 1):
            print(f"\n{i}. {opp.get('title', 'Untitled')}")
            print(f"   Solution: {opp.get('solution_name', 'TBD')}")
            print(f"   Revenue: {opp.get('revenue_potential', 'TBD')}")
            print(f"   Score: {opp.get('score', 0)}/10")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    asyncio.run(run_chunked_research())