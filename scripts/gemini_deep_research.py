"""
Optimized Gemini Deep Research Integration with Smart Decision Making

Key optimizations:
1. Progressive filtering - reduce candidates at each phase
2. Search result caching - reuse data across phases
3. Batch processing - multiple opportunities per API call
4. Early termination - stop processing low-scoring opportunities
5. Simplified prompts - base context + phase additions
6. Deduplication - cluster similar problems
7. Efficient data storage - store only essential data
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import hashlib
from collections import defaultdict
import re
import aiohttp
import ssl

from scripts.ai_brain import IntelligentAIBrain
from scripts.research_memory import ResearchMemory
from scripts.dynamic_research_prompt_generator import DynamicResearchPromptGenerator
from scripts.emergent_learning_system import EmergentLearningSystem
from scripts.custom_json_encoder import safe_json_dump
from scripts.research_summary_manager import ResearchSummaryManager


@dataclass
class MarketOpportunity:
    """Represents a detailed market opportunity from deep research."""
    problem_description: str
    affected_audience: str
    current_solutions: str
    opportunity: str
    monetization_model: str
    market_size: str
    competition_level: str
    validation_sources: List[str]  # URLs, studies, reports from Google Search
    key_insights: List[str]
    implementation_difficulty: str
    potential_revenue: str
    unique_advantages: List[str]
    search_evidence: Dict[str, Any]  # Raw search data used for validation


@dataclass
class OpportunityScore:
    """Lightweight scoring data for opportunities."""
    opportunity_id: str
    phase: str
    score: float
    passed: bool
    key_metric: str


class SearchCache:
    """Cache for search results to avoid redundant API calls."""
    
    def __init__(self):
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_key(self, query: str) -> str:
        """Generate cache key from search query."""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        # Remove year variations for better matching
        normalized = re.sub(r'\b20\d{2}\b', 'YEAR', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached result if available."""
        key = self.get_key(query)
        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        self.miss_count += 1
        return None
    
    def set(self, query: str, result: Dict):
        """Cache a search result."""
        key = self.get_key(query)
        self.cache[key] = result
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {
            'hits': self.hit_count,
            'misses': self.miss_count,
            'hit_rate': f"{hit_rate:.1f}%",
            'cached_queries': len(self.cache)
        }


class GeminiDeepResearchOptimized:
    """Optimized deep market research using smart decision making."""
    
    # Configuration for progressive filtering
    PHASE_LIMITS = {
        'phase_1': 50,      # Initial discovery
        'phase_2': 30,      # After initial validation
        'phase_2_5': 20,    # After user research
        'phase_2_75': 15,   # After revenue validation
        'phase_3': 10,      # After competition analysis
        'phase_3_5': 8,     # After technical validation
        'phase_4': 5,       # After market sizing
        'phase_4_5': 5,     # After unit economics
        'phase_4_75': 3,    # After MVP simulation
        'phase_5': 1        # Final selection
    }
    
    # Minimum score thresholds for early termination
    MIN_SCORES = {
        'phase_2': 5.0,
        'phase_2_5': 6.0,
        'phase_2_75': 6.5,
        'phase_3': 7.0,
        'phase_3_5': 7.0,
        'phase_4': 7.5,
        'phase_4_5': 7.5,
        'phase_4_75': 8.0
    }
    
    # Batch sizes for API calls
    BATCH_SIZES = {
        'phase_1': 20,      # Can process many at once for discovery
        'phase_2': 10,      # Initial validation
        'phase_2_5': 5,     # More detailed analysis
        'phase_2_75': 5,
        'phase_3': 5,
        'phase_3_5': 3,
        'phase_4': 3,
        'phase_4_5': 3,
        'phase_4_75': 2,
        'phase_5': 1
    }
    
    def __init__(self, ai_brain: Optional[IntelligentAIBrain] = None):
        """Initialize optimized research system."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # Less verbose
        
        # Initialize components
        self.ai_brain = ai_brain or IntelligentAIBrain()
        self.research_memory = ResearchMemory()
        self.prompt_generator = DynamicResearchPromptGenerator(self.ai_brain, self.research_memory)
        self.learning_system = EmergentLearningSystem()
        
        # Search cache
        self.search_cache = SearchCache()
        
        # Opportunity tracking - lightweight
        self.opportunities = {}  # id -> essential data only
        self.scores = defaultdict(dict)  # opportunity_id -> phase -> score
        self.active_opportunities = set()  # Currently being processed
        
        # Base context for all prompts
        self.base_context = self._create_base_context()
        
        # API tracking
        self.api_calls = 0
        self.start_time = None
        
        # Initialize research summary manager
        self.summary_manager = ResearchSummaryManager()
        
        # Track discovered sectors for summary
        self.all_sectors_researched = set()
        
        # Store research phases data (for compatibility with market_discovery.py)
        self.research_phases_data = {
            'phase_0_market_timing': None,
            'phase_1_problems_discovered': [],
            'phase_2_problems_validated': [],
            'phase_2_5_user_research': [],
            'phase_2_75_revenue_validation': [],
            'phase_3_competition_analysis': [],
            'phase_3_5_technical_validation': [],
            'phase_4_market_sizing': [],
            'phase_4_5_unit_economics': [],
            'phase_4_75_mvp_simulation': [],
            'phase_5_final_selection': []
        }
        
        # Get Gemini API key
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.gemini_api_key:
            self.logger.warning("GOOGLE_API_KEY not found in environment")
    
    def _create_base_context(self) -> str:
        """Create base context used across all phases."""
        return """
        Core Requirements for ALL opportunities:
        - Passive income potential (minimal ongoing effort)
        - Transaction or usage-based revenue model
        - 90%+ automation capability
        - Global 24/7 market applicability
        - Clear path to $10K+ MRR
        - Can be built by small team (1-3 developers)
        - Defensible technical moat
        
        Scoring focus on:
        1. Revenue certainty (evidence of willingness to pay)
        2. Implementation speed (MVP in < 4 weeks)
        3. Market timing (why now?)
        4. Competitive advantage (10x better than alternatives)
        """
    
    def _deduplicate_problems(self, problems: List[Dict]) -> List[Dict]:
        """Remove duplicate or highly similar problems using semantic similarity."""
        unique_problems = []
        seen_concepts = []
        
        # Common words to ignore
        stop_words = {'for', 'the', 'and', 'with', 'from', 'that', 'this', 'have', 
                     'been', 'are', 'was', 'were', 'been', 'being', 'have', 'has',
                     'had', 'will', 'would', 'could', 'should', 'may', 'might'}
        
        for problem in problems:
            # Create a concept signature
            desc = problem.get('problem_description', '').lower()
            audience = problem.get('affected_audience', '').lower()
            
            # Extract meaningful terms (not just length-based)
            desc_terms = set()
            audience_terms = set()
            
            # Clean and extract terms
            for term in desc.split():
                term = term.strip('.,!?;:"')
                if len(term) > 3 and term not in stop_words:
                    desc_terms.add(term)
                    # Also add word stems for better matching
                    if term.endswith('ing'):
                        desc_terms.add(term[:-3])
                    elif term.endswith('ed'):
                        desc_terms.add(term[:-2])
                    elif term.endswith('s') and len(term) > 4:
                        desc_terms.add(term[:-1])
            
            for term in audience.split():
                term = term.strip('.,!?;:"')
                if len(term) > 3 and term not in stop_words:
                    audience_terms.add(term)
            
            # Check similarity with seen problems
            is_duplicate = False
            for seen_desc_terms, seen_audience_terms, seen_problem in seen_concepts:
                # Calculate overlap
                desc_overlap = len(desc_terms & seen_desc_terms)
                audience_overlap = len(audience_terms & seen_audience_terms)
                
                # Calculate similarity ratios
                desc_similarity = desc_overlap / max(len(desc_terms), len(seen_desc_terms), 1)
                audience_similarity = audience_overlap / max(len(audience_terms), len(seen_audience_terms), 1)
                
                # Consider duplicate if high similarity in both description and audience
                if desc_similarity > 0.5 and audience_similarity > 0.5:
                    is_duplicate = True
                    self.logger.debug(f"Found duplicate: '{desc[:50]}...' similar to '{seen_problem[:50]}...'")
                    break
                # Also check for very high description similarity alone
                elif desc_similarity > 0.7:
                    is_duplicate = True
                    self.logger.debug(f"Found duplicate: '{desc[:50]}...' similar to '{seen_problem[:50]}...'")
                    break
            
            if not is_duplicate:
                unique_problems.append(problem)
                seen_concepts.append((desc_terms, audience_terms, desc))
                
        self.logger.info(f"Deduplication: {len(problems)} -> {len(unique_problems)} problems")
        return unique_problems
    
    async def _batch_process_phase(self, 
                                  opportunities: List[Dict], 
                                  phase_name: str,
                                  phase_prompt_generator,
                                  batch_size: int) -> List[Dict]:
        """Process opportunities in batches for efficiency."""
        results = []
        
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} of {phase_name} ({len(batch)} items)")
            
            # Generate batch prompt
            batch_prompt = phase_prompt_generator(batch)
            
            # Make API call
            self.api_calls += 1
            response = await self._call_gemini_structured(batch_prompt['prompt'], batch_prompt['schema'])
            
            if response:
                results.extend(response)
        
        return results
    
    async def _progressive_filter(self, 
                                 opportunities: List[Dict], 
                                 phase_name: str,
                                 limit: int,
                                 min_score: float = 0.0) -> List[Dict]:
        """Filter opportunities based on scores and limits."""
        # Score and sort
        scored_opps = []
        for opp in opportunities:
            opp_id = opp.get('id', opp.get('problem_description', '')[:50])
            
            # Get latest score
            latest_score = 0.0
            if opp_id in self.scores:
                phase_scores = self.scores[opp_id]
                if phase_scores:
                    latest_score = max(phase_scores.values())
            
            # Early termination check
            if latest_score < min_score:
                self.logger.info(f"Early termination for '{opp_id[:30]}...' (score: {latest_score:.1f} < {min_score})")
                continue
                
            scored_opps.append((latest_score, opp))
        
        # Sort by score descending
        scored_opps.sort(key=lambda x: x[0], reverse=True)
        
        # Take top N
        filtered = [opp for _, opp in scored_opps[:limit]]
        
        self.logger.info(f"Progressive filter for {phase_name}: {len(opportunities)} -> {len(filtered)} opportunities")
        
        return filtered
    
    def _get_cached_search_result(self, query: str) -> Optional[Dict]:
        """Get search result from cache or None."""
        result = self.search_cache.get(query)
        if result:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
        return result
    
    def _cache_search_result(self, query: str, result: Dict):
        """Cache a search result."""
        self.search_cache.set(query, result)
    
    async def _call_gemini_structured(self, prompt: str, response_schema: Dict, max_retries: int = 3) -> Optional[Any]:
        """Make direct Gemini API call with structured output and caching."""
        # Check if this is a search query we've seen before
        if "search" in prompt.lower() and len(prompt) < 1000:
            cached = self._get_cached_search_result(prompt)
            if cached:
                return cached
        
        if not self.gemini_api_key:
            self.logger.error("Gemini API key not available")
            return None
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add base context to prompt
        full_prompt = f"{self.base_context}\n\n{prompt}"
        
        # Construct request with structured output configuration
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "maxOutputTokens": 8192
            }
        }
        
        for attempt in range(max_retries):
            try:
                # Create SSL context that doesn't verify certificates (for development)
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    self.logger.debug(f"[Attempt {attempt + 1}/{max_retries}] Sending request to Gemini")
                    
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            raw_response_text = await response.text()
                            result = json.loads(raw_response_text)
                            
                            # Extract the structured response
                            if 'candidates' in result and result['candidates']:
                                candidate = result['candidates'][0]
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    text = candidate['content']['parts'][0].get('text', '')
                                    if text:
                                        try:
                                            parsed_result = json.loads(text)
                                            
                                            # Cache if it's a search result
                                            if "search" in prompt.lower() and parsed_result:
                                                self._cache_search_result(prompt, parsed_result)
                                            
                                            return parsed_result
                                        except json.JSONDecodeError as e:
                                            self.logger.error(f"Failed to parse JSON: {e}")
                                            self.logger.error(f"Raw text: {text[:500]}...")
                                            if attempt < max_retries - 1:
                                                wait_time = 2 ** attempt
                                                self.logger.info(f"JSON decode error. Retrying in {wait_time} seconds...")
                                                await asyncio.sleep(wait_time)
                                                continue
                            
                            self.logger.error(f"Unexpected response structure: {json.dumps(result)[:500]}...")
                            
                        else:
                            error_text = await response.text()
                            self.logger.error(f"API request failed with status {response.status}: {error_text}")
                            
                            if "too many states" in error_text:
                                self.logger.error("Schema complexity error - need simpler schema")
                                return None
                            
                            # For rate limiting, wait before retry
                            if response.status == 429 and attempt < max_retries - 1:
                                wait_time = 2 ** attempt
                                self.logger.info(f"Rate limited. Waiting {wait_time} seconds...")
                                await asyncio.sleep(wait_time)
                                continue
                                
            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    
        return None
    
    async def research_market_opportunities(self, 
                                          count: int = 1,
                                          existing_portfolio: Dict[str, Any] = None) -> List[Any]:
        """Main entry point - optimized research flow."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting optimized market research to find {count} opportunities")
        
        # Phase 0: Market timing (unchanged)
        market_timing = await self._phase_0_market_timing()
        if not market_timing:
            self.logger.warning("Market timing analysis failed")
            market_timing = {}
        
        # Phase 1: Problem discovery with deduplication
        problems = await self._phase_1_problem_discovery(market_timing, count)
        if not problems:
            self.logger.error("No problems discovered")
            return []
        
        # Deduplicate similar problems
        problems = self._deduplicate_problems(problems)
        
        # Convert to opportunities with IDs
        opportunities = []
        for i, problem in enumerate(problems[:self.PHASE_LIMITS['phase_1']]):
            opp_id = f"opp_{i}_{hashlib.md5(str(problem).encode()).hexdigest()[:8]}"
            problem['id'] = opp_id
            opportunities.append(problem)
            self.opportunities[opp_id] = {
                'id': opp_id,
                'problem': problem.get('problem_description', '')[:200],
                'audience': problem.get('affected_audience', '')[:100]
            }
        
        # Phase 2: Initial validation with batching
        opportunities = await self._phase_2_validate_with_data(
            opportunities,
            batch_size=self.BATCH_SIZES['phase_2']
        )
        opportunities = await self._progressive_filter(
            opportunities, 
            'phase_2',
            self.PHASE_LIMITS['phase_2'],
            self.MIN_SCORES['phase_2']
        )
        
        if not opportunities:
            self.logger.warning("No opportunities passed initial validation")
            return []
        
        # Continue through phases with progressive filtering
        phase_methods = [
            ('phase_2_5', self._phase_2_5_user_research),
            ('phase_2_75', self._phase_2_75_revenue_validation),
            ('phase_3', self._phase_3_competition_analysis),
            ('phase_3_5', self._phase_3_5_technical_validation),
            ('phase_4', self._phase_4_market_sizing),
            ('phase_4_5', self._phase_4_5_unit_economics),
            ('phase_4_75', self._phase_4_75_mvp_simulation)
        ]
        
        for phase_name, phase_method in phase_methods:
            # Process phase
            opportunities = await phase_method(
                opportunities,
                batch_size=self.BATCH_SIZES.get(phase_name, 5)
            )
            
            # Progressive filter
            opportunities = await self._progressive_filter(
                opportunities,
                phase_name,
                self.PHASE_LIMITS.get(phase_name, 10),
                self.MIN_SCORES.get(phase_name, 0.0)
            )
            
            if not opportunities:
                self.logger.warning(f"No opportunities passed {phase_name}")
                return []
        
        # Phase 5: Final selection (optimized)
        final_opportunities = await self._phase_5_final_selection_optimized(
            opportunities,
            existing_portfolio,
            count
        )
        
        # Print optimization stats
        self._print_optimization_stats()
        
        # Save research knowledge and update summary
        if final_opportunities:
            await self._save_research_knowledge(final_opportunities)
        
        return final_opportunities
    
    async def _phase_0_market_timing(self) -> Optional[Dict[str, Any]]:
        """Analyze market conditions - simplified prompt."""
        self.logger.info("Phase 0: Market Timing Analysis")
        
        prompt = """
        Analyze current market conditions for software opportunities.
        
        Search for:
        1. "software market trends 2024 2025"
        2. "emerging technology opportunities"
        3. "startup funding trends recent"
        
        Return top 3 trending areas with evidence.
        """
        
        schema = {
            "type": "object",
            "properties": {
                "trending_areas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "string"},
                            "evidence": {"type": "string"},
                            "timing_score": {"type": "number"}
                        }
                    },
                    "maxItems": 3
                }
            }
        }
        
        self.api_calls += 1
        result = await self._call_gemini_structured(prompt, schema)
        return result
    
    async def _phase_1_problem_discovery(self, market_timing: Dict, target_count: int) -> List[Dict]:
        """Discover problems dynamically based on market timing - batched for efficiency."""
        self.logger.info("Phase 1: Problem Discovery (Dynamic)")
        
        # Use market timing to guide discovery
        trending_areas = market_timing.get('trending_areas', [])
        if not trending_areas:
            self.logger.warning("No trending areas found, using broad discovery")
            trending_areas = [{"area": "emerging technology", "evidence": "general search"}]
        
        # Build dynamic search queries from trends
        search_queries = []
        for area in trending_areas[:3]:  # Use top 3 trending areas
            area_name = area.get('area', '')
            if area_name:
                search_queries.append(f"{area_name} problems")
                search_queries.append(f"{area_name} pain points")
        
        # Calculate how many problems to discover (more than target for filtering)
        # Limited to 25 to avoid schema complexity errors (tested max is 29)
        discover_count = min(25, target_count * 2)
        
        prompt = f"""
        Discover real problems people face based on current market trends.
        
        Current trending areas (from market analysis):
        {json.dumps(trending_areas, indent=2)}
        
        Search broadly for problems in these trending areas and related fields:
        {chr(10).join(f'- {q}' for q in search_queries)}
        
        Also explore:
        - Problems mentioned in recent tech news
        - Underserved markets in trending sectors  
        - Cross-industry opportunities
        - Problems that existing solutions handle poorly
        
        Return {discover_count} specific problems with clear audiences.
        Focus on problems with transaction potential and passive income possibilities.
        Be diverse - don't just focus on one sector.
        """
        
        # Use object wrapper to avoid array schema complexity
        schema = {
            "type": "object",
            "properties": {
                "problems": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "problem_description": {"type": "string"},
                            "affected_audience": {"type": "string"}
                        },
                        "required": ["problem_description", "affected_audience"]
                    },
                    "maxItems": discover_count
                }
            },
            "required": ["problems"]
        }
        
        self.api_calls += 1
        result = await self._call_gemini_structured(prompt, schema)
        
        # Extract problems from wrapped response
        if result and 'problems' in result:
            problems = result['problems']
            # Track all discovered sectors
            for problem in problems:
                if 'affected_audience' in problem:
                    self.all_sectors_researched.add(problem['affected_audience'])
            return problems
        
        return []
    
    async def _phase_2_validate_with_data(self, opportunities: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Validate opportunities in batches."""
        self.logger.info(f"Phase 2: Validating {len(opportunities)} opportunities")
        
        validated = []
        
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Create simplified batch validation prompt
            batch_summary = []
            for j, opp in enumerate(batch):
                batch_summary.append(f"{i+j}. {opp.get('problem_description', '')[:100]}")
            
            prompt = f"""
            Validate these {len(batch)} opportunities for market potential.
            
            Opportunities:
            {chr(10).join(batch_summary)}
            
            For each, search for evidence of:
            - Market demand
            - Willingness to pay
            - Current solutions
            
            Score each 0-10 based on validation strength.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "validation_score": {"type": "number"},
                        "evidence": {"type": "string"},
                        "revenue_model": {"type": "string"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                # Match results back to opportunities
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['validation_score'] = result.get('validation_score', 0)
                        opp['validation_evidence'] = result.get('evidence', '')
                        opp['revenue_model'] = result.get('revenue_model', '')
                        
                        # Store score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_2'] = opp['validation_score']
                        
                        validated.append(opp)
        
        return validated
    
    async def _phase_2_5_user_research(self, opportunities: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Lightweight user research validation."""
        self.logger.info(f"Phase 2.5: User research for {len(opportunities)} opportunities")
        
        # Similar batch processing pattern
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            Quick user validation for {len(batch)} opportunities.
            
            Search for evidence of user demand:
            - Forums/Reddit discussions
            - Product Hunt requests
            - Twitter complaints
            
            Rate user pain level 0-10.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "user_pain_score": {"type": "number"},
                        "evidence_summary": {"type": "string"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['user_pain_score'] = result.get('user_pain_score', 0)
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_2_5'] = opp['user_pain_score']
        
        return opportunities
    
    async def _phase_2_75_revenue_validation(self, opportunities: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Revenue potential validation."""
        self.logger.info(f"Phase 2.75: Revenue validation for {len(opportunities)} opportunities")
        
        # Process in batches with focused revenue search
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            Validate revenue potential for {len(batch)} opportunities.
            
            Search for:
            - Similar product pricing
            - Market size estimates
            - Transaction volumes
            
            Estimate monthly revenue potential.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "revenue_potential": {"type": "string"},
                        "revenue_score": {"type": "number"},
                        "pricing_model": {"type": "string"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['revenue_potential'] = result.get('revenue_potential', '')
                        opp['revenue_score'] = result.get('revenue_score', 0)
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_2_75'] = opp['revenue_score']
        
        return opportunities
    
    async def _phase_3_competition_analysis(self, opportunities: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Analyze competition efficiently."""
        self.logger.info(f"Phase 3: Competition analysis for {len(opportunities)} opportunities")
        
        # Batch process with simplified competition check
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            Quick competition check for {len(batch)} opportunities.
            
            Find:
            - Main competitors (if any)
            - Market gaps
            - Differentiation potential
            
            Score opportunity 0-10 (10 = low competition, high potential).
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "competition_score": {"type": "number"},
                        "main_competitors": {"type": "string"},
                        "differentiation": {"type": "string"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['competition_score'] = result.get('competition_score', 0)
                        opp['differentiation'] = result.get('differentiation', '')
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_3'] = opp['competition_score']
        
        return opportunities
    
    async def _phase_3_5_technical_validation(self, opportunities: List[Dict], batch_size: int = 3) -> List[Dict]:
        """Technical feasibility check."""
        self.logger.info(f"Phase 3.5: Technical validation for {len(opportunities)} opportunities")
        
        # Quick technical assessment
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            Technical feasibility for {len(batch)} opportunities.
            
            Assess:
            - Development complexity (1-10, 1=easy)
            - Required tech stack
            - Time to MVP (weeks)
            
            Score feasibility 0-10 (10 = very feasible).
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "technical_score": {"type": "number"},
                        "complexity": {"type": "number"},
                        "mvp_weeks": {"type": "number"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['technical_score'] = result.get('technical_score', 0)
                        opp['mvp_weeks'] = result.get('mvp_weeks', 4)
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_3_5'] = opp['technical_score']
        
        return opportunities
    
    async def _phase_4_market_sizing(self, opportunities: List[Dict], batch_size: int = 3) -> List[Dict]:
        """Market size validation."""
        self.logger.info(f"Phase 4: Market sizing for {len(opportunities)} opportunities")
        
        # Simplified market sizing
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            Market size for {len(batch)} opportunities.
            
            Estimate:
            - Total addressable market
            - Reachable market in year 1
            - Growth potential
            
            Score market size 0-10.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "market_score": {"type": "number"},
                        "tam": {"type": "string"},
                        "year_1_potential": {"type": "string"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['market_score'] = result.get('market_score', 0)
                        opp['market_size'] = result.get('tam', '')
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_4'] = opp['market_score']
        
        return opportunities
    
    async def _phase_4_5_unit_economics(self, opportunities: List[Dict], batch_size: int = 3) -> List[Dict]:
        """Unit economics validation."""
        self.logger.info(f"Phase 4.5: Unit economics for {len(opportunities)} opportunities")
        
        # Quick unit economics
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            Unit economics for {len(batch)} opportunities.
            
            Calculate:
            - Customer acquisition cost estimate
            - Lifetime value potential
            - Break-even timeline
            
            Score economics 0-10.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "economics_score": {"type": "number"},
                        "cac_estimate": {"type": "string"},
                        "ltv_estimate": {"type": "string"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['economics_score'] = result.get('economics_score', 0)
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_4_5'] = opp['economics_score']
        
        return opportunities
    
    async def _phase_4_75_mvp_simulation(self, opportunities: List[Dict], batch_size: int = 2) -> List[Dict]:
        """MVP planning simulation."""
        self.logger.info(f"Phase 4.75: MVP simulation for {len(opportunities)} opportunities")
        
        # Detailed MVP planning for top candidates
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            prompt = f"""
            MVP planning for {len(batch)} opportunities.
            
            Define:
            - Core features (max 5)
            - Launch strategy
            - First revenue timeline
            
            Score MVP viability 0-10.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "mvp_score": {"type": "number"},
                        "core_features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5
                        },
                        "days_to_revenue": {"type": "number"}
                    }
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 0)
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        opp['mvp_score'] = result.get('mvp_score', 0)
                        opp['core_features'] = result.get('core_features', [])
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_4_75'] = opp['mvp_score']
        
        return opportunities
    
    async def _phase_5_final_selection_optimized(self, 
                                                 opportunities: List[Dict],
                                                 existing_portfolio: Dict,
                                                 target_count: int = 1) -> List[Any]:
        """Optimized final selection - returns top opportunities up to target_count."""
        self.logger.info(f"Phase 5: Final selection of {target_count} from {len(opportunities)} candidates")
        
        # Calculate composite scores
        for opp in opportunities:
            opp_id = opp.get('id')
            if opp_id in self.scores:
                phase_scores = self.scores[opp_id]
                # Weighted average of all phase scores
                weights = {
                    'phase_2': 0.10,
                    'phase_2_5': 0.10,
                    'phase_2_75': 0.20,  # Revenue weight
                    'phase_3': 0.15,
                    'phase_3_5': 0.10,
                    'phase_4': 0.15,
                    'phase_4_5': 0.10,
                    'phase_4_75': 0.10
                }
                
                total_score = 0
                total_weight = 0
                for phase, weight in weights.items():
                    if phase in phase_scores:
                        total_score += phase_scores[phase] * weight
                        total_weight += weight
                
                opp['final_score'] = total_score / total_weight if total_weight > 0 else 0
            else:
                opp['final_score'] = 0
        
        # Sort by final score
        opportunities.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        # Take the top opportunities based on target_count
        top_opportunities = opportunities[:min(target_count, len(opportunities))]
        
        # If we need solutions generated, do it in a batch
        needs_solution = []
        for i, opp in enumerate(top_opportunities):
            if not opp.get('opportunity') or opp.get('opportunity') == opp.get('problem_description'):
                needs_solution.append((i, opp))
        
        if needs_solution:
            # Generate solutions for all that need them
            problems_list = [f"{i+1}. {opp.get('problem_description', '')}" for i, opp in needs_solution]
            
            prompt = f"""
            Generate innovative AI-powered solutions for these problems:
            
            {chr(10).join(problems_list)}
            
            For each problem, provide:
            - A specific solution approach
            - Key differentiator
            - Core technology/method
            
            Focus on passive income potential and automation.
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "solution": {"type": "string"},
                        "differentiator": {"type": "string"},
                        "core_tech": {"type": "string"}
                    }
                },
                "maxItems": len(needs_solution)
            }
            
            solutions = await self._call_gemini_structured(prompt, schema)
            
            if solutions:
                # Apply solutions to opportunities
                for sol in solutions:
                    idx = sol.get('index', 1) - 1
                    if 0 <= idx < len(needs_solution):
                        opp_idx, opp = needs_solution[idx]
                        top_opportunities[opp_idx]['opportunity'] = sol.get('solution', '')
                        top_opportunities[opp_idx]['differentiator'] = sol.get('differentiator', '')
                        top_opportunities[opp_idx]['core_tech'] = sol.get('core_tech', '')
        
        # Convert all top opportunities to MarketOpportunity objects
        market_opportunities = []
        
        for i, selected in enumerate(top_opportunities):
            self.logger.info(f"Selected opportunity {i+1}: {selected.get('problem_description', '')[:100]}...")
            
            # Use the generated or existing opportunity
            opportunity_desc = selected.get('opportunity', f"AI-powered solution for {selected.get('problem_description', '')[:100]}")
            
            market_opp = MarketOpportunity(
                problem_description=selected.get('problem_description', ''),
                affected_audience=selected.get('affected_audience', ''),
                current_solutions=selected.get('current_solutions', ''),
                opportunity=opportunity_desc,
                monetization_model=selected.get('revenue_model', 'Transaction-based'),
                market_size=selected.get('market_size', '$1B+'),
                competition_level='medium',
                validation_sources=[selected.get('validation_evidence', '')],
                key_insights=selected.get('core_features', []),
                implementation_difficulty=f"{selected.get('mvp_weeks', 4)} weeks",
                potential_revenue=selected.get('revenue_potential', '$10K/month'),
                unique_advantages=[selected.get('differentiation', '')],
                search_evidence={'summary': 'Validated through optimized research'}
            )
            
            market_opportunities.append(market_opp)
        
        self.logger.info(f"Returning {len(market_opportunities)} opportunities")
        return market_opportunities
    
    def _print_optimization_stats(self):
        """Print statistics about the optimization."""
        if not self.start_time:
            return
            
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("OPTIMIZATION STATISTICS")
        print("="*60)
        print(f"Total API calls: {self.api_calls}")
        print(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"API calls/minute: {self.api_calls / (duration/60):.1f}")
        
        # Cache stats
        cache_stats = self.search_cache.get_stats()
        print(f"\nSearch Cache Performance:")
        print(f"  Cache hits: {cache_stats['hits']}")
        print(f"  Cache misses: {cache_stats['misses']}")
        print(f"  Hit rate: {cache_stats['hit_rate']}")
        print(f"  Cached queries: {cache_stats['cached_queries']}")
        
        # Phase progression
        print(f"\nProgressive Filtering:")
        for phase, limit in self.PHASE_LIMITS.items():
            phase_count = len([s for s in self.scores.values() if phase in s])
            if phase_count > 0:
                print(f"  {phase}: {phase_count} opportunities (limit: {limit})")
        
        # Early terminations
        terminated = len([s for s in self.scores.values() if len(s) < 8])
        print(f"\nEarly terminations: {terminated}")
        
        print("="*60 + "\n")
    
    async def _save_research_knowledge(self, opportunities: List[MarketOpportunity]):
        """Save research knowledge and update summary.
        
        Args:
            opportunities: List of validated market opportunities
        """
        try:
            # Create research directory if needed
            research_dir = Path("research_knowledge")
            research_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"research_{timestamp}.json"
            filepath = research_dir / filename
            
            # Prepare research data
            research_data = {
                'research_timestamp': datetime.now(timezone.utc).isoformat(),
                'opportunities_discovered': len(opportunities),
                'api_calls_made': self.api_calls,
                'research_phases': self.research_phases_data,
                'opportunities': [asdict(opp) for opp in opportunities],
                'optimization_stats': {
                    'api_calls': self.api_calls,
                    'cache_hits': self.search_cache.hit_count,
                    'cache_misses': self.search_cache.miss_count,
                    'early_terminations': len([s for s in self.scores.values() if len(s) < 8])
                }
            }
            
            # Save research data
            with open(filepath, 'w') as f:
                safe_json_dump(research_data, f, indent=2)
            
            self.logger.info(f"✅ Saved research knowledge to: {filepath}")
            
            # Update research summary using the manager
            stats = self.summary_manager.add_research_session(
                opportunities=opportunities,
                research_file=filename,
                sectors=self.all_sectors_researched
            )
            
            self.logger.info(
                f"📊 Summary updated: {stats['problems_added']} added, "
                f"{stats['problems_duplicate']} duplicates filtered"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save research knowledge: {e}")
            # Don't fail the whole process if saving fails


# Make it compatible with existing system
GeminiDeepResearch = GeminiDeepResearchOptimized