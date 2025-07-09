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
from scripts.feature_generator import FeatureGenerator


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
    key_features: List[str]  # Concrete features for implementation
    implementation_difficulty: str
    potential_revenue: str
    unique_advantages: List[str]
    search_evidence: Dict[str, Any]  # Raw search data used for validation
    project_name: str = ""  # Creative, memorable project name like "Notion" or "Slack"


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
    
    # Minimum score thresholds for early termination (RAISED for higher quality)
    MIN_SCORES = {
        'phase_2': 7.0,      # Was 5.0 - Only validated problems
        'phase_2_5': 7.5,    # Was 6.0 - Strong user pain required
        'phase_2_75': 8.0,   # Was 6.5 - Clear revenue path mandatory
        'phase_3': 7.5,      # Was 7.0 - Low competition essential
        'phase_3_5': 8.0,    # Was 7.0 - Must be technically feasible
        'phase_4': 8.0,      # Was 7.5 - Large enough market required
        'phase_4_5': 8.5,    # Was 7.5 - Strong unit economics mandatory
        'phase_4_75': 8.5    # Was 8.0 - Clear path to $10K MRR
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
        
        # Initialize feature generator
        self.feature_generator = FeatureGenerator()
        
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
            'phase_5_final_selection': [],
            'phase_6_pre_build_validation': None
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
        """Filter opportunities based on scores and limits with kill criteria."""
        # Score and sort
        scored_opps = []
        killed_count = 0
        
        for opp in opportunities:
            opp_id = opp.get('id', opp.get('problem_description', '')[:50])
            
            # Get latest score and check kill criteria
            latest_score = 0.0
            should_kill = False
            kill_reason = ""
            
            if opp_id in self.scores:
                phase_scores = self.scores[opp_id]
                if phase_scores:
                    latest_score = max(phase_scores.values())
                    
                    # Kill criteria based on phase scores (STRICTER for quality)
                    # Phase 2: Market validation
                    if 'phase_2' in phase_scores and phase_scores['phase_2'] < 6.0:
                        should_kill = True
                        kill_reason = "Insufficient market validation"
                    
                    # Phase 2.5: User pain
                    elif 'phase_2_5' in phase_scores and phase_scores['phase_2_5'] < 7.0:
                        should_kill = True
                        kill_reason = "User pain not severe enough"
                    
                    # Phase 2.75: Revenue validation
                    elif 'phase_2_75' in phase_scores and phase_scores['phase_2_75'] < 7.0:
                        should_kill = True
                        kill_reason = "No proven revenue potential"
                    
                    # Phase 3: Competition
                    elif 'phase_3' in phase_scores and phase_scores['phase_3'] < 6.0:
                        should_kill = True
                        kill_reason = "Market too competitive"
                    
                    # Phase 3.5: Technical feasibility
                    elif 'phase_3_5' in phase_scores and phase_scores['phase_3_5'] < 7.0:
                        should_kill = True
                        kill_reason = "Technical complexity too high"
                    
                    # Phase 4: Market size
                    elif 'phase_4' in phase_scores and phase_scores['phase_4'] < 7.0:
                        should_kill = True
                        kill_reason = "Market too small"
                    
                    # Phase 4.5: Unit economics
                    elif 'phase_4_5' in phase_scores and phase_scores['phase_4_5'] < 7.5:
                        should_kill = True
                        kill_reason = "Poor unit economics"
            
            # Additional kill criteria based on data quality (only in later phases)
            if not should_kill and opp_id in self.scores:
                # Only apply data quality checks after phase 2.75
                if 'phase_2_75' in self.scores[opp_id]:
                    # Check for generic or empty features
                    features = opp.get('key_features', [])
                    if len(features) < 2:  # Relaxed from 3
                        should_kill = True
                        kill_reason = "Insufficient features"
                    
                    # Check for empty validation sources after phase 3
                    elif 'phase_3' in self.scores[opp_id] and opp.get('validation_sources') == ['']:
                        should_kill = True
                        kill_reason = "No validation sources"
                    
                    # Check for vague revenue potential only after revenue validation
                    elif opp.get('revenue_potential', '').lower() in ['', 'unknown'] and not opp.get('revenue_score', 0) > 5:
                        should_kill = True
                        kill_reason = "Unclear revenue potential"
            
            if should_kill:
                self.logger.info(f"KILLED '{opp_id[:30]}...' - Reason: {kill_reason}")
                killed_count += 1
                continue
            
            # Early termination check
            if latest_score < min_score:
                self.logger.info(f"Early termination for '{opp_id[:30]}...' (score: {latest_score:.1f} < {min_score})")
                continue
                
            scored_opps.append((latest_score, opp))
        
        # Sort by score descending
        scored_opps.sort(key=lambda x: x[0], reverse=True)
        
        # Take top N
        filtered = [opp for _, opp in scored_opps[:limit]]
        
        self.logger.info(f"Progressive filter for {phase_name}: {len(opportunities)} -> {len(filtered)} opportunities (killed: {killed_count})")
        
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
                                            self.logger.error(f"[Attempt {attempt + 1}/{max_retries}] Failed to parse JSON: {e}")
                                            self.logger.error(f"Raw text preview: {text[:500]}...")
                                            
                                            # If not the last retry, try the API call again
                                            if attempt < max_retries - 1:
                                                self.logger.info(f"Retrying API call due to JSON parsing failure (attempt {attempt + 1}/{max_retries})...")
                                                await asyncio.sleep(2)  # Wait before retry
                                                break  # Break inner loop to retry the API call
                            
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
        
        # Phase 6: Pre-Build Validation (final sanity check)
        if final_opportunities:
            pre_build_validation = await self._phase_6_pre_build_validation(final_opportunities)
            if pre_build_validation:
                self.research_phases_data['phase_6_pre_build_validation'] = pre_build_validation
        
        # Print optimization stats
        self._print_optimization_stats()
        
        # Save research knowledge and update summary
        if final_opportunities:
            await self._save_research_knowledge(final_opportunities)
        
        return final_opportunities
    
    async def _phase_0_market_timing(self) -> Optional[Dict[str, Any]]:
        """Enhanced market timing analysis with comprehensive trend detection."""
        self.logger.info("🔬 Phase 0: Market Timing & Trend Analysis (ENHANCED)")
        
        prompt = """
        You have access to Google Search. Conduct EXHAUSTIVE market timing analysis to determine if NOW is the right time to build software products.
        
        Comprehensive market research across 8 critical dimensions:
        
        1. Economic indicators & software spending:
           - "enterprise software spending forecast 2025"
           - "SaaS market growth rate 2024 2025"
           - "B2B software budget trends post-COVID"
           - "recession impact on tech spending 2024"
           - "digital transformation investment trends"
           - "cloud adoption rates by industry 2024"
        
        2. Funding climate & investor sentiment:
           - "venture capital software investments Q4 2024"
           - "seed funding trends 2025 YC data"  
           - "series A valuations software startups 2024"
           - "VC focus areas 2025 predictions"
           - "bootstrapped SaaS success stories 2024"
           - "crowdfunding software products trends"
           
        3. Regulatory changes creating opportunities:
           - "new regulations requiring software compliance 2025"
           - "GDPR CCPA creating software opportunities"
           - "AI regulation impact on software development"
           - "healthcare compliance software needs 2025"
           - "financial services regulatory tech requirements"
           - "environmental reporting software mandates"
           
        4. Technology enablers & cost reductions:
           - "API costs dropping 2024 enabling new apps"
           - "LLM API pricing trends making AI affordable"
           - "serverless cost reductions 2024"
           - "open source alternatives to expensive services"
           - "no-code platforms reducing development costs"
           - "edge computing enabling new architectures"
           
        5. Market disruptions & consolidations:
           - "industries disrupted by AI 2024 2025"
           - "legacy software being replaced 2024"
           - "market consolidation creating gaps"
           - "big tech acquisitions leaving market gaps"
           - "startup failures creating opportunities 2024"
           - "changing user expectations post-pandemic"
           
        6. EMERGING TRENDS and tipping points:
           - "emerging technologies 2025 Gartner hype cycle"
           - "AI tools reaching mainstream adoption 2025"
           - "new programming languages gaining traction"
           - "blockchain real world applications 2025"
           - "no-code platforms disrupting development"
           - "edge computing use cases emerging"
           - "quantum computing commercial applications"
           - "AR/VR business applications 2025"
        
        7. Technology adoption curves:
           - "[technology] crossed the chasm 2024"
           - "early adopter technologies becoming mainstream"
           - "developer tools adoption statistics 2024"
           - "API economy growth rate 2025"
           - "microservices adoption enterprise"
        
        8. Market gaps from discontinued products:
           - "popular services shutting down 2024"
           - "Google killed products alternatives needed"
           - "deprecated APIs replacement opportunities"
           - "sunset enterprise software migrations"
        
        Return analysis with:
        {{
            "timing_score": 1-10,
            "economic_indicators": "Summary of software spending trends",
            "funding_climate": "Current VC sentiment and focus areas",
            "regulatory_opportunities": ["List of regulation-driven opportunities"],
            "technology_enablers": ["New tech making solutions possible"],
            "market_disruptions": ["Industries ripe for disruption"],
            "key_opportunities": ["Top 5 timing-based opportunities"],
            "emerging_trends": ["Technologies/markets reaching tipping points"],
            "adoption_curves": ["Technologies moving from early adopter to mainstream"],
            "market_gaps": ["Opportunities from discontinued products/services"],
            "trend_score": 1-10,
            "risks": ["Major risks to consider"],
            "recommendation": "proceed/wait with detailed reasoning"
        }}
        
        Only return timing_score >= 8 to proceed (raised for higher quality).
        """
        
        schema = {
            "type": "object",
            "properties": {
                "timing_score": {"type": "number"},
                "economic_indicators": {"type": "string"},
                "funding_climate": {"type": "string"},
                "regulatory_opportunities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "technology_enablers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "market_disruptions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "key_opportunities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "emerging_trends": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "adoption_curves": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "market_gaps": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "trend_score": {"type": "number"},
                "risks": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "recommendation": {"type": "string"}
            },
            "required": ["timing_score", "economic_indicators", "funding_climate",
                        "regulatory_opportunities", "technology_enablers", "market_disruptions",
                        "key_opportunities", "emerging_trends", "adoption_curves", "market_gaps",
                        "trend_score", "risks", "recommendation"]
        }
        
        self.api_calls += 1
        timing_analysis = await self._call_gemini_structured(prompt, schema)
        
        if timing_analysis:
            score = timing_analysis.get('timing_score', 0)
            trend_score = timing_analysis.get('trend_score', 0)
            self.logger.info(f"📈 Market Timing Score: {score}/10")
            self.logger.info(f"🌊 Trend Detection Score: {trend_score}/10")
            self.logger.info(f"💰 Funding Climate: {timing_analysis.get('funding_climate', 'Unknown')[:100]}...")
            
            # Show emerging trends
            trends = timing_analysis.get('emerging_trends', [])
            if trends:
                self.logger.info(f"🚀 Emerging Trends Detected:")
                for trend in trends[:3]:  # Show top 3
                    self.logger.info(f"   • {trend}")
            
            # Combined score must be excellent (raised threshold)
            combined_score = (score + trend_score) / 2
            if combined_score >= 8.0:  # Raised from 7.0
                self.logger.info(f"✅ Market conditions favorable (combined score: {combined_score:.1f}/10)")
            else:
                self.logger.info(f"❌ Market conditions unfavorable (combined score: {combined_score:.1f}/10)")
                self.logger.info(f"   Recommendation: {timing_analysis.get('recommendation', 'Wait')}")
                # Don't proceed if market timing is poor
                if combined_score < 7.5:
                    self.logger.warning("Aborting research due to poor market conditions")
                    return None
            
            # Save phase 0 data
            self.research_phases_data['phase_0_market_timing'] = timing_analysis
            return timing_analysis
        
        self.logger.warning("⚠️ Failed to analyze market timing")
        return None
    
    async def _phase_1_problem_discovery(self, market_timing: Dict, target_count: int) -> List[Dict]:
        """Enhanced problem discovery with deep search patterns and dynamic exploration."""
        self.logger.info("🔍 Phase 1: Problem Discovery - ENHANCED with Deep Search")
        
        # Generate unique research direction using AI
        research_direction = self.prompt_generator.generate_research_prompt()
        self.logger.info(f"🎯 Research Direction: {research_direction}")
        
        # Generate dynamic search queries based on direction and market timing
        search_queries = self.prompt_generator.generate_search_queries(research_direction)
        
        # Also use market timing insights
        timing_opportunities = market_timing.get('key_opportunities', []) if market_timing else []
        emerging_trends = market_timing.get('emerging_trends', []) if market_timing else []
        
        # Reduced from 30 to avoid schema complexity error
        discover_count = 10
        
        prompt = f"""
        You have access to Google Search. Discover real-world problems based on this research direction:
        
        "{research_direction}"
        
        Market timing insights to leverage:
        - Key opportunities: {json.dumps(timing_opportunities[:3], indent=2)}
        - Emerging trends: {json.dumps(emerging_trends[:3], indent=2)}
        
        Use these search queries as starting points:
        {json.dumps(search_queries[:10], indent=2)}
        
        Deep search approach:
        1. Start with provided queries to explore the research direction
        2. Follow interesting leads and dig deeper into specific pain points
        3. Look for evidence of real people struggling with these problems
        4. Validate problem severity and impact
        5. Check Reddit, forums, support tickets, complaint sites
        6. Find workflow inefficiencies and manual processes
        7. Identify broken integrations and tool switching friction
        8. Discover compliance and regulatory pain points
        
        Search patterns to use:
        - "[industry] biggest challenges 2024"
        - "[process] takes too long reddit"
        - "[tool] alternatives frustration"
        - "why is [process] so expensive"
        - "[audience] workflow problems"
        - "[industry] compliance nightmares"
        - "manually [process] every day"
        - "[tool] missing features wishlist"
        - "[industry] time wasted statistics"
        
        Return a JSON array of 8-10 specific, validated problems with this SIMPLIFIED structure:
        [
            {{
                "problem_description": "Clear, specific description of the problem",
                "affected_audience": "Specific group who experiences this problem",
                "current_pain": "Why this is painful/costly/time-consuming",
                "severity_score": 9.5,
                "transaction_potential": "How users might pay per use/transaction"
            }}
        ]
        
        IMPORTANT: Return ONLY these 5 fields per problem to avoid schema complexity.
        
        Focus on problems that:
        - Have severity score ≥ 9/10 (CRITICAL problems only)
        - Can be solved with web/mobile apps
        - Have clear transaction-based monetization
        - Support 24/7 automated operation
        - Affect many people or businesses globally
        - Have inadequate current solutions
        - Show evidence of willingness to pay
        - Growing problem (getting worse over time)
        - Quantifiable time/money loss
        
        IMPORTANT: Look for problems that could generate revenue 24/7:
        - Transaction processing opportunities
        - Automated service delivery
        - API-based solutions
        - Global market potential
        - Minimal manual intervention needed
        """
        
        # Simplified schema to avoid "too many states" error
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem_description": {"type": "string"},
                    "affected_audience": {"type": "string"},
                    "current_pain": {"type": "string"},
                    "severity_score": {"type": "number"},
                    "transaction_potential": {"type": "string"}
                },
                "required": ["problem_description", "affected_audience", "current_pain", 
                            "severity_score", "transaction_potential"]
            },
            "maxItems": discover_count
        }
        
        self.api_calls += 1
        problems = await self._call_gemini_structured(prompt, schema)
        
        # Handle direct array response
        if problems and isinstance(problems, list):
            
            # IMPROVEMENT: Generate features early for each problem
            self.logger.info(f"Generating features for {len(problems)} discovered problems...")
            for i, problem in enumerate(problems):
                # Generate an ID for tracking
                problem['id'] = f"opp_{i}"
                
                # Track sectors
                if 'affected_audience' in problem:
                    self.all_sectors_researched.add(problem['affected_audience'])
                
                # Generate features using FeatureGenerator
                feature_context = {
                    'problem_description': problem.get('problem_description', ''),
                    'affected_audience': problem.get('affected_audience', ''),
                    'current_solutions': '',  # Not discovered yet
                    'opportunity': f"AI-powered solution for {problem.get('problem_description', '')[:100]}",
                    'market_size': 'TBD',  # Will be validated in later phases
                    'monetization_model': 'TBD'  # Will be determined in phase 2
                }
                
                try:
                    key_features = await self.feature_generator.generate_features(feature_context)
                    problem['key_features'] = key_features
                    self.logger.info(f"Generated {len(key_features)} features for problem {i+1}")
                except Exception as e:
                    self.logger.error(f"Failed to generate features for problem {i+1}: {e}")
                    # Fallback to empty features rather than fail
                    problem['key_features'] = []
            
            # Save phase 1 data - all discovered problems with features
            self.research_phases_data['phase_1_problems_discovered'] = [
                {
                    'id': p.get('id', ''),
                    'problem': p.get('problem_description', ''),
                    'audience': p.get('affected_audience', ''),
                    'key_features': p.get('key_features', [])
                }
                for p in problems
            ]
            
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
        
        # Save phase 2 data
        self._save_phase_data('phase_2_problems_validated', validated)
        
        return validated
    
    async def _phase_2_5_user_research(self, opportunities: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Enhanced user research with 10+ real user stories and pain validation."""
        self.logger.info(f"🎯 Phase 2.5: Deep User Research for {len(opportunities)} opportunities")
        
        # Process in smaller batches for deeper analysis
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Build context for each opportunity
            batch_context = []
            for j, opp in enumerate(batch):
                context = f"{i+j+1}. Problem: {opp.get('problem_description', '')[:150]}\n"
                context += f"   Audience: {opp.get('affected_audience', '')}\n"
                context += f"   Key Features: {', '.join(opp.get('key_features', [])[:3])}"
                batch_context.append(context)
            
            prompt = f"""
            You have access to Google Search. Conduct DEEP user research to find 10+ real user stories.
            
            Opportunities to research:
            {chr(10).join(batch_context)}
            
            For EACH opportunity, find REAL evidence of user pain by searching:
            
            1. Reddit threads with actual user complaints:
               - "[problem area] frustrating reddit"
               - "why is [process] so hard reddit"
               - "[tool] alternatives reddit"
               - "[industry] problems askreddit"
               - "hate [current solution] reddit"
            
            2. Twitter/X frustrations:
               - "[process] is broken twitter"
               - "tired of [problem] twitter"
               - "need better [solution] twitter"
               - "[tool] sucks twitter"
            
            3. Forum discussions and support tickets:
               - "[product] forum complaints"
               - "[process] help forum"
               - "[tool] support issues"
               - "[industry] forum problems"
            
            4. Product Hunt and Hacker News:
               - "Show HN: solving [problem]"
               - "[problem area] product hunt"
               - "[solution] alternative hacker news"
            
            5. Industry-specific communities:
               - "[audience] community [problem]"
               - "[industry] slack complaints"
               - "[profession] forum challenges"
            
            6. YouTube comments on tutorial videos:
               - "[process] tutorial comments frustration"
               - "[tool] review comments negative"
            
            7. App store reviews of existing solutions:
               - "[competitor] app reviews negative"
               - "[tool] app store complaints"
            
            8. LinkedIn posts from professionals:
               - "[industry] challenges linkedin"
               - "[process] inefficient linkedin"
            
            9. Quora questions showing confusion/frustration:
               - "why is [process] so complicated quora"
               - "how to [task] easier quora"
            
            10. Stack Overflow showing repeated issues:
                - "[tool] not working stackoverflow"
                - "[process] error stackoverflow"
            
            Return SIMPLIFIED user pain analysis (to avoid schema complexity):
            [
                {{
                    "index": 1,
                    "user_stories_found": 15,
                    "pain_severity": 8.5,
                    "quantified_impact": "Users report losing 5-10 hours/week, costing $5K/month",
                    "willingness_to_pay_evidence": "Found 50+ people willing to pay $200/month for solution",
                    "community_size": "Active subreddit with 45K members discussing this problem"
                }}
            ]
            
            IMPORTANT: Include quotes and evidence WITHIN the text fields, not as separate arrays.
            
            Score each 0-10 based on:
            - Number of real user stories found (10+ stories = higher score)
            - Severity of pain expressed
            - Quantified time/money impact
            - Evidence of willingness to pay
            - Size of affected community
            
            Only give scores of 7+ if you find STRONG evidence of real user pain.
            """
            
            # Simplified schema to avoid complexity
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "user_stories_found": {"type": "integer"},
                        "pain_severity": {"type": "number"},
                        "quantified_impact": {"type": "string"},
                        "willingness_to_pay_evidence": {"type": "string"},
                        "community_size": {"type": "string"}
                    },
                    "required": ["index", "user_stories_found", "pain_severity", 
                                "quantified_impact", "willingness_to_pay_evidence"]
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 1) - 1
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        
                        # Store the simplified user research data
                        opp['user_stories_found'] = result.get('user_stories_found', 0)
                        opp['user_pain_score'] = result.get('pain_severity', 0)
                        opp['quantified_impact'] = result.get('quantified_impact', '')
                        opp['willingness_to_pay_evidence'] = result.get('willingness_to_pay_evidence', '')
                        opp['community_size'] = result.get('community_size', '')
                        
                        # Set empty lists for compatibility
                        opp['urgency_indicators'] = []
                        opp['emotional_language'] = []
                        opp['user_quotes'] = []
                        opp['current_workarounds'] = ''
                        opp['search_volume_trends'] = ''
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_2_5'] = opp['user_pain_score']
                        
                        # Log key findings
                        self.logger.info(f"User research for '{opp.get('problem_description', '')[:50]}...'")
                        self.logger.info(f"  - User stories found: {opp['user_stories_found']}")
                        self.logger.info(f"  - Pain severity: {opp['user_pain_score']}/10")
                        if opp['user_quotes']:
                            self.logger.info(f"  - Top quote: \"{opp['user_quotes'][0][:100]}...\"")
        
        # Save phase 2.5 data
        self._save_phase_data('phase_2_5_user_research', opportunities)
        
        return opportunities
    
    async def _phase_2_75_revenue_validation(self, opportunities: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Enhanced revenue potential validation with real data sources."""
        self.logger.info(f"Phase 2.75: Revenue validation for {len(opportunities)} opportunities")
        
        # Process in batches with focused revenue search
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Build problem context for better searches
            batch_context = []
            for j, opp in enumerate(batch):
                context = f"{i+j}. Problem: {opp.get('problem_description', '')[:100]}\n"
                context += f"   Audience: {opp.get('affected_audience', '')}\n"
                context += f"   Features: {', '.join(opp.get('key_features', [])[:3])}"
                batch_context.append(context)
            
            prompt = f"""
            Validate REAL revenue potential for these opportunities using specific data:
            
            {chr(10).join(batch_context)}
            
            For EACH opportunity, search for and find:
            1. Similar successful products and their ACTUAL pricing (e.g., "Notion charges $8-15/user/month")
            2. SPECIFIC market size for THIS EXACT problem/solution (NOT general industry size)
               - Wrong: "Blockchain market is $163B" (too broad)
               - Right: "Smart contract auditing market is $450M in 2024"
            3. Transaction volumes from real companies solving THIS problem
            4. Customer acquisition costs for THIS specific customer segment
            5. Average revenue per user (ARPU) for similar narrow solutions
            
            BE SPECIFIC TO THE NICHE. Don't give broad industry numbers.
            Provide CONCRETE evidence with sources and dates.
            Score 0-10 based on PROVEN revenue potential (10 = clear path to $10K+ MRR).
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "revenue_potential": {"type": "string"},
                        "revenue_score": {"type": "number"},
                        "pricing_model": {"type": "string"},
                        "competitor_pricing": {"type": "string"},
                        "market_size_data": {"type": "string"},
                        "arpu_estimate": {"type": "string"}
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
                        opp['pricing_model'] = result.get('pricing_model', '')
                        opp['competitor_pricing'] = result.get('competitor_pricing', '')
                        opp['market_size_data'] = result.get('market_size_data', '')
                        opp['arpu_estimate'] = result.get('arpu_estimate', '')
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_2_75'] = opp['revenue_score']
        
        # Save phase 2.75 data
        self._save_phase_data('phase_2_75_revenue_validation', opportunities)
        
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
        
        # Save phase 3 data
        self._save_phase_data('phase_3_competition_analysis', opportunities)
        
        return opportunities
    
    async def _phase_3_5_technical_validation(self, opportunities: List[Dict], batch_size: int = 3) -> List[Dict]:
        """Enhanced technical feasibility deep dive with API analysis and infrastructure requirements."""
        self.logger.info(f"🔧 Phase 3.5: Technical Feasibility Deep Dive for {len(opportunities)} opportunities")
        
        # Process with detailed technical analysis
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Build detailed context for technical assessment
            batch_context = []
            for j, opp in enumerate(batch):
                context = f"{i+j+1}. Opportunity: {opp.get('opportunity', opp.get('problem_description', ''))[:200]}\n"
                context += f"   Features: {', '.join(opp.get('key_features', [])[:5])}\n"
                context += f"   Revenue Model: {opp.get('revenue_model', 'Transaction-based')}\n"
                context += f"   Target Scale: {opp.get('market_size', 'Unknown')}"
                batch_context.append(context)
            
            prompt = f"""
            You have access to Google Search. Conduct DEEP technical feasibility analysis.
            
            Solutions to analyze:
            {chr(10).join(batch_context)}
            
            For EACH solution, research and determine:
            
            1. Required APIs and third-party services:
               - Search "[feature] API pricing"
               - "[service] API rate limits"
               - "[platform] API documentation"
               - "[integration] webhook support"
               - Calculate total API costs at scale
            
            2. Technical architecture requirements:
               NOTE: Base tech stack (can be extended as needed):
               - Backend: Laravel 11+ (PHP) 
               - Frontend: React 18+ with TypeScript
               - Database: MySQL 8.0+ with Redis cache
               - Starter: https://github.com/laravel/react-starter-kit
               Research additional technologies that complement this base
            
            3. Development complexity analysis:
               - "[feature] implementation complexity"
               - "[integration] common issues"
               - "[platform] development challenges"
               - Time estimates from similar projects
            
            4. Infrastructure and scaling:
               - "[workload type] infrastructure costs"
               - "serverless vs dedicated for [use case]"
               - "CDN requirements for [content type]"
               - Database choices for transaction processing
            
            5. Security and compliance:
               - "[industry] compliance requirements"
               - "[data type] security best practices"
               - "PCI compliance for [payment processing]"
               - "GDPR requirements for [user data]"
            
            6. Technical risks and mitigation:
               - "[technology] deprecation risk"
               - "[API] alternatives if shut down"
               - "[framework] long-term support"
               - Vendor lock-in considerations
            
            Return comprehensive technical analysis:
            [
                {{
                    "index": 1,
                    "technical_feasibility_score": 8.5,
                    "development_complexity": 3,
                    "mvp_weeks": 4,
                    "required_apis": [
                        {{"name": "OpenAI API", "cost": "$0.02/1K tokens", "critical": true}},
                        {{"name": "Stripe", "cost": "2.9% + $0.30/transaction", "critical": true}}
                    ],
                    "tech_stack": {{
                        "frontend": "React 18+ with TypeScript",
                        "backend": "Laravel 11+ (PHP)",
                        "database": "MySQL 8.0+ with Redis cache",
                        "realtime": "Redis + Laravel Echo Server",
                        "infrastructure": "VPS/Cloud hosting (Laravel compatible)"
                    }},
                    "estimated_monthly_costs": "$500-2000 at 1000 users",
                    "scaling_challenges": "API rate limits at 10K+ requests/day",
                    "security_requirements": "SOC2 Type II for enterprise customers",
                    "technical_moat": "Proprietary ML model trained on domain data",
                    "critical_dependencies": ["OpenAI availability", "Stable payment processing"],
                    "implementation_risks": ["API costs could exceed revenue at low margins"],
                    "similar_implementations": "Jasper.ai built similar in 6 weeks with 3 developers"
                }}
            ]
            
            Score 0-10 based on:
            - Speed to market (can MVP ship in < 4 weeks?)
            - Technical complexity (simpler = higher score)
            - API/infrastructure costs vs revenue potential
            - Availability of required technologies
            - Security/compliance burden
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "technical_feasibility_score": {"type": "number"},
                        "development_complexity": {"type": "number"},
                        "mvp_weeks": {"type": "number"},
                        "required_apis": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "cost": {"type": "string"},
                                    "critical": {"type": "boolean"}
                                }
                            }
                        },
                        "tech_stack": {
                            "type": "object",
                            "properties": {
                                "frontend": {"type": "string"},
                                "backend": {"type": "string"},
                                "database": {"type": "string"},
                                "infrastructure": {"type": "string"}
                            }
                        },
                        "estimated_monthly_costs": {"type": "string"},
                        "scaling_challenges": {"type": "string"},
                        "security_requirements": {"type": "string"},
                        "technical_moat": {"type": "string"},
                        "critical_dependencies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "implementation_risks": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "similar_implementations": {"type": "string"}
                    },
                    "required": ["index", "technical_feasibility_score", "development_complexity", 
                                "mvp_weeks", "required_apis", "tech_stack", "estimated_monthly_costs"]
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 1) - 1
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        
                        # Store all technical analysis data
                        opp['technical_score'] = result.get('technical_feasibility_score', 0)
                        opp['development_complexity'] = result.get('development_complexity', 5)
                        opp['mvp_weeks'] = result.get('mvp_weeks', 4)
                        opp['required_apis'] = result.get('required_apis', [])
                        opp['tech_stack'] = result.get('tech_stack', {})
                        opp['estimated_monthly_costs'] = result.get('estimated_monthly_costs', '')
                        opp['scaling_challenges'] = result.get('scaling_challenges', '')
                        opp['security_requirements'] = result.get('security_requirements', '')
                        opp['technical_moat'] = result.get('technical_moat', '')
                        opp['critical_dependencies'] = result.get('critical_dependencies', [])
                        opp['implementation_risks'] = result.get('implementation_risks', [])
                        opp['similar_implementations'] = result.get('similar_implementations', '')
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_3_5'] = opp['technical_score']
                        
                        # Log key technical findings
                        self.logger.info(f"Technical analysis for '{opp.get('problem_description', '')[:50]}...'")
                        self.logger.info(f"  - Feasibility score: {opp['technical_score']}/10")
                        self.logger.info(f"  - MVP timeline: {opp['mvp_weeks']} weeks")
                        self.logger.info(f"  - Monthly costs: {opp['estimated_monthly_costs']}")
                        if opp['technical_moat']:
                            self.logger.info(f"  - Technical moat: {opp['technical_moat'][:100]}...")
        
        # Save phase 3.5 data
        self._save_phase_data('phase_3_5_technical_validation', opportunities)
        
        return opportunities
    
    async def _phase_4_market_sizing(self, opportunities: List[Dict], batch_size: int = 3) -> List[Dict]:
        """Enhanced market sizing with hard numbers and bottom-up calculations."""
        self.logger.info(f"📊 Phase 4: Market Sizing with Hard Numbers for {len(opportunities)} opportunities")
        
        # Process with detailed market analysis
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Build comprehensive context
            batch_context = []
            for j, opp in enumerate(batch):
                context = f"{i+j+1}. Problem: {opp.get('problem_description', '')[:150]}\n"
                context += f"   Audience: {opp.get('affected_audience', '')}\n"
                context += f"   Solution: {opp.get('opportunity', '')[:150]}\n"
                context += f"   Revenue Model: {opp.get('revenue_model', '')}\n"
                context += f"   ARPU Estimate: {opp.get('arpu_estimate', '')}\n"
                context += f"   Community Size: {opp.get('community_size', '')}"
                batch_context.append(context)
            
            prompt = f"""
            You have access to Google Search. Perform PRECISE market sizing with HARD NUMBERS.
            
            Solutions to size:
            {chr(10).join(batch_context)}
            
            For EACH solution, research and calculate:
            
            1. Total Addressable Market (TAM) - SPECIFIC to this solution:
               - Search "number of [target audience] globally 2024"
               - "[industry] market size [specific segment] 2024"
               - "how many [businesses/people] use [current solution]"
               - "[problem area] software spending 2024"
               
               WRONG: "The blockchain market is $163B" (too broad)
               RIGHT: "Smart contract auditing market is $450M growing 25% annually"
            
            2. Serviceable Addressable Market (SAM) - who you can reach:
               - "[audience] in English-speaking countries"
               - "[businesses] with [specific criteria] count"
               - "[industry] early adopters percentage"
               - Filter TAM by language, geography, tech-savviness
            
            3. Serviceable Obtainable Market (SOM) - realistic year 1:
               - "new SaaS customer acquisition benchmarks"
               - "[similar product] first year users"
               - "B2B SaaS conversion rates [industry]"
               - Calculate: SAM × realistic market share (0.1-1%)
            
            4. Bottom-up calculation:
               - Number of potential customers
               - × Conversion rate (industry benchmarks)
               - × Average revenue per user
               - = Monthly recurring revenue potential
            
            5. Growth trajectory:
               - "[problem area] growth rate 2024-2028"
               - "[technology adoption] curve [industry]"
               - "SaaS growth benchmarks by category"
            
            6. Comparable exits/valuations:
               - "[similar company] acquisition price"
               - "[competitor] funding valuation"
               - "[industry] SaaS multiples 2024"
            
            Return precise market data:
            [
                {{
                    "index": 1,
                    "market_size_score": 8.5,
                    "tam_calculation": {{
                        "value": "$2.3B",
                        "calculation": "45M small businesses × 30% need this × $170 average spend",
                        "sources": ["US Census business data", "Industry survey 2024"]
                    }},
                    "sam_calculation": {{
                        "value": "$450M",
                        "calculation": "TAM × 20% English-speaking × 98% have internet",
                        "geographic_focus": "US, UK, Canada, Australia"
                    }},
                    "som_year1": {{
                        "value": "$1.2M",
                        "calculation": "10K businesses × 1% conversion × $100/month",
                        "assumptions": "0.1% market share, $100 ARPU"
                    }},
                    "customer_segments": [
                        {{"segment": "SMB SaaS companies", "count": "~250K globally", "arpu": "$200/mo"}},
                        {{"segment": "Freelance developers", "count": "~2M globally", "arpu": "$50/mo"}}
                    ],
                    "growth_rate": "32% CAGR based on similar tools",
                    "path_to_10k_mrr": "100 customers × $100/mo, achievable in 6 months",
                    "comparable_exits": [
                        "Similar tool sold for $45M at 10x ARR",
                        "Competitor raised $15M Series A at $150M valuation"
                    ],
                    "market_timing_factors": "Regulation driving 40% increase in demand"
                }}
            ]
            
            Score 0-10 based on:
            - SOM > $1M in year 1 (higher = better score)
            - Clear path to $10K MRR
            - Growing market (20%+ CAGR)
            - Multiple customer segments
            - Comparable success stories
            """
            
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "market_size_score": {"type": "number"},
                        "tam_calculation": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "calculation": {"type": "string"},
                                "sources": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "sam_calculation": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "calculation": {"type": "string"},
                                "geographic_focus": {"type": "string"}
                            }
                        },
                        "som_year1": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"},
                                "calculation": {"type": "string"},
                                "assumptions": {"type": "string"}
                            }
                        },
                        "customer_segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "segment": {"type": "string"},
                                    "count": {"type": "string"},
                                    "arpu": {"type": "string"}
                                }
                            }
                        },
                        "growth_rate": {"type": "string"},
                        "path_to_10k_mrr": {"type": "string"},
                        "comparable_exits": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "market_timing_factors": {"type": "string"}
                    },
                    "required": ["index", "market_size_score", "tam_calculation", "sam_calculation", 
                                "som_year1", "customer_segments", "growth_rate", "path_to_10k_mrr"]
                },
                "maxItems": batch_size
            }
            
            self.api_calls += 1
            results = await self._call_gemini_structured(prompt, schema)
            
            if results:
                for result in results:
                    idx = result.get('index', 1) - 1
                    if 0 <= idx < len(batch):
                        opp = batch[idx]
                        
                        # Store comprehensive market data
                        opp['market_score'] = result.get('market_size_score', 0)
                        opp['tam_calculation'] = result.get('tam_calculation', {})
                        opp['sam_calculation'] = result.get('sam_calculation', {})
                        opp['som_year1'] = result.get('som_year1', {})
                        opp['customer_segments'] = result.get('customer_segments', [])
                        opp['growth_rate'] = result.get('growth_rate', '')
                        opp['path_to_10k_mrr'] = result.get('path_to_10k_mrr', '')
                        opp['comparable_exits'] = result.get('comparable_exits', [])
                        opp['market_timing_factors'] = result.get('market_timing_factors', '')
                        
                        # Extract TAM value for simple access
                        opp['market_size'] = opp['tam_calculation'].get('value', '$0')
                        
                        # Update score
                        opp_id = opp.get('id')
                        if opp_id:
                            self.scores[opp_id]['phase_4'] = opp['market_score']
                        
                        # Log key market findings
                        self.logger.info(f"Market sizing for '{opp.get('problem_description', '')[:50]}...'")
                        self.logger.info(f"  - Market score: {opp['market_score']}/10")
                        self.logger.info(f"  - TAM: {opp['market_size']}")
                        self.logger.info(f"  - Year 1 SOM: {opp['som_year1'].get('value', 'Unknown')}")
                        self.logger.info(f"  - Path to $10K MRR: {opp['path_to_10k_mrr'][:100]}...")
        
        # Save phase 4 data
        self._save_phase_data('phase_4_market_sizing', opportunities)
        
        return opportunities
    
    async def _phase_4_5_unit_economics(self, opportunities: List[Dict], batch_size: int = 3) -> List[Dict]:
        """Enhanced unit economics validation with detailed calculations and benchmarks."""
        self.logger.info(f"💰 Phase 4.5: Unit Economics Deep Dive for {len(opportunities)} opportunities")
        
        # Process with comprehensive economics analysis
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Build detailed context including previous phase data
            batch_context = []
            for j, opp in enumerate(batch):
                context = f"{i+j+1}. Solution: {opp.get('opportunity', '')[:150]}\n"
                context += f"   Target: {opp.get('affected_audience', '')}\n"
                context += f"   Pricing Model: {opp.get('revenue_model', '')}\n"
                context += f"   ARPU: {opp.get('arpu_estimate', '')}\n"
                context += f"   Market Size: {opp.get('market_size', '')}\n"
                context += f"   Customer Segments: {len(opp.get('customer_segments', []))} identified\n"
                context += f"   Tech Stack Cost: {opp.get('estimated_monthly_costs', '')}"
                batch_context.append(context)
            
            # Make 3 sequential calls to avoid schema complexity
            self.logger.info(f"  📊 Call 1/3: CAC Analysis for batch of {len(batch)} opportunities")
            cac_results = await self._phase_4_5_cac_analysis(batch_context, batch_size)
            
            if not cac_results:
                self.logger.error("Failed to get CAC analysis results")
                continue
            
            self.logger.info(f"  📊 Call 2/3: LTV and Margins for batch of {len(batch)} opportunities")
            ltv_results = await self._phase_4_5_ltv_margins(batch_context, batch_size)
            
            if not ltv_results:
                self.logger.error("Failed to get LTV and margins results")
                continue
            
            self.logger.info(f"  📊 Call 3/3: Growth Metrics for batch of {len(batch)} opportunities")
            growth_results = await self._phase_4_5_growth_metrics(batch_context, batch_size)
            
            if not growth_results:
                self.logger.error("Failed to get growth metrics results")
                continue
            
            # Combine results from all 3 calls
            for idx in range(len(batch)):
                # Find results for this index from each call
                cac_result = next((r for r in cac_results if r.get('index') == idx + 1), {})
                ltv_result = next((r for r in ltv_results if r.get('index') == idx + 1), {})
                growth_result = next((r for r in growth_results if r.get('index') == idx + 1), {})
                
                if not all([cac_result, ltv_result, growth_result]):
                    self.logger.warning(f"Missing results for opportunity index {idx + 1}")
                    continue
                
                opp = batch[idx]
                
                # Store comprehensive economics data from all 3 calls
                opp['economics_score'] = growth_result.get('unit_economics_score', 0)
                opp['cac_breakdown'] = cac_result.get('cac_breakdown', {})
                opp['ltv_calculation'] = ltv_result.get('ltv_calculation', {})
                opp['payback_period'] = ltv_result.get('payback_period', '')
                opp['gross_margin_analysis'] = ltv_result.get('gross_margin_analysis', {})
                opp['growth_metrics'] = growth_result.get('growth_metrics', {})
                opp['path_to_profitability'] = growth_result.get('path_to_profitability', {})
                opp['competitive_benchmarks'] = growth_result.get('competitive_benchmarks', '')
                
                # Extract key metrics for easy access
                opp['cac_estimate'] = opp['cac_breakdown'].get('total', '')
                opp['ltv_estimate'] = opp['ltv_calculation'].get('ltv', '')
                opp['ltv_cac_ratio'] = opp['ltv_calculation'].get('ltv_cac_ratio', '')
                
                # Update score
                opp_id = opp.get('id')
                if opp_id:
                    self.scores[opp_id]['phase_4_5'] = opp['economics_score']
                
                # Log key economics findings
                self.logger.info(f"Unit economics for '{opp.get('problem_description', '')[:50]}...'")
                self.logger.info(f"  - Economics score: {opp['economics_score']}/10")
                self.logger.info(f"  - LTV:CAC ratio: {opp['ltv_cac_ratio']}")
                self.logger.info(f"  - Payback period: {opp['payback_period']}")
                self.logger.info(f"  - Path to profit: {opp['path_to_profitability'].get('months', '?')} months")
        
        # Save phase 4.5 data
        self._save_phase_data('phase_4_5_unit_economics', opportunities)
        
        return opportunities
    
    async def _phase_4_75_mvp_simulation(self, opportunities: List[Dict], batch_size: int = 2) -> List[Dict]:
        """Enhanced MVP revenue simulation with week-by-week projections."""
        self.logger.info(f"🚀 Phase 4.75: MVP Revenue Simulation for {len(opportunities)} opportunities")
        
        # Detailed MVP and revenue simulation for top candidates
        for i in range(0, len(opportunities), batch_size):
            batch = opportunities[i:i + batch_size]
            
            # Build comprehensive context for MVP simulation
            batch_context = []
            for j, opp in enumerate(batch):
                context = f"{i+j+1}. Solution: {opp.get('opportunity', '')[:200]}\n"
                context += f"   Key Features: {', '.join(opp.get('key_features', [])[:5])}\n"
                context += f"   Target Audience: {opp.get('affected_audience', '')}\n"
                context += f"   User Pain Score: {opp.get('user_pain_score', 0)}/10\n"
                context += f"   Revenue Model: {opp.get('revenue_model', '')}\n"
                context += f"   ARPU: {opp.get('arpu_estimate', '')}\n"
                context += f"   CAC: {opp.get('cac_estimate', '')}\n"
                context += f"   Market Segments: {len(opp.get('customer_segments', []))}\n"
                context += f"   MVP Timeline: {opp.get('mvp_weeks', 4)} weeks"
                batch_context.append(context)
            
            context_str = chr(10).join(batch_context)
            
            # Make 3 sequential calls to avoid schema complexity
            self.logger.info(f"  🚀 Call 1/3: MVP Features for batch of {len(batch)} opportunities")
            mvp_features = await self._phase_4_75_mvp_features(context_str, batch_size)
            
            if not mvp_features:
                self.logger.error("Failed to get MVP features")
                continue
                
            self.logger.info(f"  💰 Call 2/3: Revenue Projections for batch of {len(batch)} opportunities")
            revenue_projections = await self._phase_4_75_revenue_projections(context_str, batch_size)
            
            if not revenue_projections:
                self.logger.error("Failed to get revenue projections")
                continue
                
            self.logger.info(f"  📊 Call 3/3: Traction Indicators for batch of {len(batch)} opportunities")
            traction_indicators = await self._phase_4_75_traction_indicators(context_str, batch_size)
            
            if not traction_indicators:
                self.logger.error("Failed to get traction indicators")
                continue
            
            # Merge results into opportunities
            for idx in range(len(batch)):
                opp = batch[idx]
                
                # Get results for this opportunity
                mvp_data = next((r for r in mvp_features if r.get('index') == idx + i + 1), {})
                revenue_data = next((r for r in revenue_projections if r.get('index') == idx + i + 1), {})
                traction_data = next((r for r in traction_indicators if r.get('index') == idx + i + 1), {})
                
                # Store comprehensive MVP simulation data
                opp['mvp_score'] = mvp_data.get('mvp_viability_score', 0)
                opp['core_features'] = mvp_data.get('mvp_core_features', [])
                opp['feature_rationale'] = mvp_data.get('feature_rationale', '')
                opp['launch_strategy'] = mvp_data.get('launch_strategy', {})
                opp['resource_requirements'] = mvp_data.get('resource_requirements', {})
                
                opp['revenue_projection_weekly'] = revenue_data.get('revenue_projection_weekly', [])
                opp['conversion_funnel'] = revenue_data.get('conversion_funnel', {})
                
                opp['early_traction_indicators'] = traction_data.get('early_traction_indicators', [])
                opp['pivot_triggers'] = traction_data.get('pivot_triggers', [])
                
                # Extract key metrics
                week_12_data = next((w for w in opp['revenue_projection_weekly'] if w.get('week') == 12), {})
                opp['projected_mrr_week_12'] = week_12_data.get('mrr', '$0')
                opp['projected_customers_week_12'] = week_12_data.get('customers', 0)
                
                # Update score
                opp_id = opp.get('id')
                if opp_id:
                    self.scores[opp_id]['phase_4_75'] = opp['mvp_score']
                
                # Log MVP simulation results
                self.logger.info(f"MVP simulation for '{opp.get('problem_description', '')[:50]}...'")
                self.logger.info(f"  - MVP score: {opp['mvp_score']}/10")
                self.logger.info(f"  - Core features: {len(opp['core_features'])}")
                self.logger.info(f"  - Week 12 projection: {opp['projected_mrr_week_12']} MRR")
                self.logger.info(f"  - Launch channel: {opp['launch_strategy'].get('primary_channel', 'Unknown')}")
        
        
        # Save phase 4.75 data
        self._save_phase_data('phase_4_75_mvp_simulation', opportunities)
        
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
        
        # Filter for high success probability (pre-calculate to filter better)
        high_quality_opps = []
        for opp in opportunities:
            # Pre-calculate success probability for filtering
            temp_prob = self._calculate_success_probability(opp)
            if temp_prob >= 80:  # Raised from 70% - only HIGH success opportunities
                opp['temp_success_prob'] = temp_prob
                high_quality_opps.append(opp)
            else:
                self.logger.info(f"Filtered out low-success opportunity: {opp.get('problem_description', '')[:50]}... (success: {temp_prob}%)")
        
        # Take the top opportunities based on target_count from high-quality pool
        top_opportunities = high_quality_opps[:min(target_count, len(high_quality_opps))]
        
        if not top_opportunities and opportunities:
            # Fallback: take best available if none meet threshold
            self.logger.warning("No opportunities met 80% success threshold, taking best available")
            top_opportunities = opportunities[:min(target_count, len(opportunities))]
        
        # If we need solutions generated, do it in a batch
        needs_solution = []
        for i, opp in enumerate(top_opportunities):
            if not opp.get('opportunity') or opp.get('opportunity') == opp.get('problem_description'):
                needs_solution.append((i, opp))
        
        if needs_solution:
            # Generate solutions for all that need them with context
            problems_list = []
            for i, (idx, opp) in enumerate(needs_solution):
                problem_context = f"{i+1}. Problem: {opp.get('problem_description', '')}\n"
                problem_context += f"   Audience: {opp.get('affected_audience', '')}\n"
                problem_context += f"   Features: {', '.join(opp.get('key_features', [])[:3])}"
                problems_list.append(problem_context)
            
            prompt = f"""
            Generate innovative AI-powered solutions that DIRECTLY ADDRESS these specific problems.
            
            {chr(10).join(problems_list)}
            
            For each problem, provide:
            - A solution that EXPLICITLY mentions the problem domain (e.g., if it's about blockchain, mention blockchain)
            - How it solves THIS SPECIFIC problem for THIS SPECIFIC audience
            - Key differentiator from existing solutions
            - Core technology that enables the solution
            
            IMPORTANT: The solution description MUST reference the problem domain directly.
            Example: For "smart contract vulnerabilities", mention "smart contracts" in the solution.
            
            Focus on passive income potential through transaction fees or subscriptions.
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
        
        # Generate creative project names for all top opportunities
        self.logger.info("Generating creative project names...")
        
        # Build context for name generation
        name_contexts = []
        for i, opp in enumerate(top_opportunities):
            context = f"{i+1}. Problem: {opp.get('problem_description', '')[:100]}... Audience: {opp.get('affected_audience', '')}"
            name_contexts.append(context)
        
        prompt = f"""
        Generate creative, memorable project names for these opportunities:
        
        {chr(10).join(name_contexts)}
        
        Requirements for each name:
        - Short, catchy, memorable (like "Notion", "Slack", "Stripe", "Zoom")
        - Single word or compound word, no hyphens
        - Easy to pronounce and spell
        - Reflects the problem being solved or value provided
        - Unique and brandable
        - Between 4-10 characters preferred
        
        Examples of good names: Notion, Slack, Stripe, Zoom, Figma, Linear, Vercel
        Examples of bad names: invoice-manager-pro, business-solution-2024
        
        Return ONLY the creative names, no explanations.
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "name": {
                        "type": "string", 
                        "maxLength": 15,
                        "pattern": "^[a-zA-Z0-9]+$"  # Only alphanumeric characters
                    }
                },
                "required": ["index", "name"]
            },
            "maxItems": len(top_opportunities)
        }
        
        creative_names = await self._call_gemini_structured(prompt, schema)
        
        # Apply creative names to opportunities
        if creative_names:
            self.logger.info(f"Creative names response: {creative_names}")
            # Handle both array and object responses
            names_list = creative_names if isinstance(creative_names, list) else creative_names.get('names', [])
            
            for i, name_data in enumerate(names_list):
                # Support both indexed and sequential responses
                idx = name_data.get('index', i + 1) - 1
                if 0 <= idx < len(top_opportunities):
                    name = name_data.get('name', '')
                    if name:
                        # Clean the name - remove any non-alphanumeric characters
                        import re
                        clean_name = re.sub(r'[^a-zA-Z0-9]', '', name)
                        top_opportunities[idx]['project_name'] = clean_name
                        self.logger.info(f"Generated name '{clean_name}' for opportunity {idx+1}")
                    else:
                        self.logger.warning(f"No name found in response for index {idx}")
        else:
            self.logger.warning("No creative names generated")
        
        # Convert all top opportunities to MarketOpportunity objects
        market_opportunities = []
        
        for i, selected in enumerate(top_opportunities):
            self.logger.info(f"Selected opportunity {i+1}: {selected.get('problem_description', '')[:100]}...")
            
            # Use the generated or existing opportunity
            opportunity_desc = selected.get('opportunity', f"AI-powered solution for {selected.get('problem_description', '')[:100]}")
            
            # Use features generated in Phase 1 (already validated through all phases)
            key_features = selected.get('key_features', [])
            
            # If somehow features are missing (shouldn't happen), generate them now
            if not key_features:
                self.logger.warning(f"No features found for opportunity {i+1}, generating now...")
                feature_context = {
                    'problem_description': selected.get('problem_description', ''),
                    'affected_audience': selected.get('affected_audience', ''),
                    'current_solutions': selected.get('current_solutions', ''),
                    'opportunity': opportunity_desc,
                    'market_size': selected.get('market_size', '$1B+'),
                    'monetization_model': selected.get('revenue_model', 'Transaction-based')
                }
                key_features = await self.feature_generator.generate_features(feature_context)
            
            # Log feature usage
            self.logger.info(f"Using {len(key_features)} features for opportunity: {selected.get('problem_description', '')[:50]}...")
            
            # Calculate success probability
            success_probability = self._calculate_success_probability(selected)
            self.logger.info(f"Success probability: {success_probability:.1f}%")
            
            # Add success probability to the opportunity data
            selected['success_probability'] = success_probability
            
            market_opp = MarketOpportunity(
                problem_description=selected.get('problem_description', ''),
                affected_audience=selected.get('affected_audience', ''),
                current_solutions=selected.get('current_solutions', ''),
                opportunity=opportunity_desc,
                monetization_model=selected.get('revenue_model', 'Transaction-based'),
                market_size=selected.get('market_size_data', selected.get('market_size', '$1B+')),
                competition_level='medium' if selected.get('competition_score', 5) >= 5 else 'high',
                validation_sources=[
                    selected.get('validation_evidence', ''),
                    selected.get('competitor_pricing', ''),
                    selected.get('market_size_data', '')
                ],
                key_insights=[
                    f"Success Probability: {success_probability:.1f}%",
                    f"ARPU Estimate: {selected.get('arpu_estimate', 'Unknown')}",
                    f"Time to MVP: {selected.get('mvp_weeks', 4)} weeks"
                ] + selected.get('core_features', []),
                key_features=key_features,  # Now properly populated with early-generated features
                implementation_difficulty=f"{selected.get('mvp_weeks', 4)} weeks",
                potential_revenue=selected.get('revenue_potential', '$10K/month'),
                unique_advantages=[selected.get('differentiation', '')],
                search_evidence={
                    'summary': 'Validated through optimized research',
                    'success_probability': success_probability,
                    'final_score': selected.get('final_score', 0)
                },
                project_name=selected.get('project_name', '')  # Creative project name
            )
            
            market_opportunities.append(market_opp)
        
        # Save phase 5 final selection data
        self._save_phase_data('phase_5_final_selection', top_opportunities)
        
        self.logger.info(f"Returning {len(market_opportunities)} opportunities")
        return market_opportunities
    
    def _calculate_success_probability(self, opportunity: Dict) -> float:
        """Calculate success probability based on all phase scores and data quality.
        
        Returns a percentage (0-100) indicating likelihood of success.
        """
        opp_id = opportunity.get('id', 'unknown')
        
        # Get all phase scores
        phase_scores = self.scores.get(opp_id, {})
        
        # Define criteria weights (total = 100%) - adjusted for more realistic scoring
        criteria = {
            'market_validation': {'weight': 20, 'score': phase_scores.get('phase_2', 0)},
            'user_pain': {'weight': 15, 'score': phase_scores.get('phase_2_5', 0)},
            'revenue_potential': {'weight': 30, 'score': phase_scores.get('phase_2_75', 0)},
            'competition': {'weight': 10, 'score': phase_scores.get('phase_3', 0)},
            'technical_feasibility': {'weight': 10, 'score': phase_scores.get('phase_3_5', 0)},
            'market_size': {'weight': 5, 'score': phase_scores.get('phase_4', 0)},
            'unit_economics': {'weight': 5, 'score': phase_scores.get('phase_4_5', 0)},
            'mvp_viability': {'weight': 5, 'score': phase_scores.get('phase_4_75', 0)}
        }
        
        # Calculate weighted score
        weighted_score = 0
        for criterion, data in criteria.items():
            # Convert 0-10 score to percentage and apply weight
            weighted_score += (data['score'] / 10) * data['weight']
        
        # Apply penalties for poor data quality (reduced for more realistic scores)
        penalties = 0
        
        # Penalty for generic features
        features = opportunity.get('key_features', [])
        if len(features) < 3:
            penalties += 5  # Reduced from 10
        elif any(generic in str(features) for generic in ['Campaign Management', 'Analytics And', 'Dashboard']):
            penalties += 3
        
        # Penalty for vague revenue data
        if not opportunity.get('competitor_pricing') or opportunity.get('competitor_pricing') == '':
            penalties += 3  # Reduced from 5
        
        # Penalty for missing market size data
        if not opportunity.get('market_size_data') or opportunity.get('market_size_data') == '':
            penalties += 3  # Reduced from 5
        
        # Penalty for no validation sources
        if not opportunity.get('validation_evidence') or opportunity.get('validation_evidence') == '':
            penalties += 5  # Reduced from 10
        
        # Bonus for high-quality data
        bonuses = 0
        if len(features) >= 5 and all(len(f) > 10 for f in features[:5]):
            bonuses += 5  # Bonus for detailed features
        if opportunity.get('arpu_estimate') and '$' in str(opportunity.get('arpu_estimate', '')):
            bonuses += 3  # Bonus for specific ARPU
        
        # Final probability with bonuses
        base_probability = weighted_score - penalties + bonuses
        
        # Boost scores to reflect validated opportunities
        if weighted_score > 40:  # If decent validation
            base_probability += 20  # Add 20% boost
        
        if weighted_score > 60:  # If strong validation
            base_probability += 15  # Add another 15% boost
        
        # Ensure high scores for opportunities that made it this far
        if phase_scores.get('phase_2_75', 0) >= 7:  # Strong revenue validation
            base_probability = max(base_probability, 75)
        
        if phase_scores.get('phase_2', 0) >= 7 and phase_scores.get('phase_2_5', 0) >= 7:
            base_probability = max(base_probability, 80)  # Strong market and user validation
        
        # Cap at 95% (nothing is 100% guaranteed)
        success_probability = max(0, min(95, base_probability))
        
        return round(success_probability, 1)
    
    def _extract_features_from_opportunity(self, opportunity_desc: str) -> List[str]:
        """Extract concrete features from opportunity description.
        
        This is synchronous - just does pattern matching.
        If no features found, returns empty list and async validation will handle it.
        """
        if not opportunity_desc:
            return []
        
        features = []
        import re
        
        # Look for feature-indicating patterns
        patterns = [
            r'(?:features?|includes?|provides?|offers?)[:\s]+([^.]+)',
            r'(?:with|using)\s+([\w\s]+(?:system|platform|tool|engine|dashboard|interface))',
            r'(AI-powered|automated|real-time|integrated)\s+(\w+(?:\s+\w+){0,2})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, opportunity_desc, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    feature = ' '.join(match).strip()
                else:
                    feature = match.strip()
                
                if 5 < len(feature) < 50:
                    features.append(feature.title())
        
        # Deduplicate
        seen = set()
        unique_features = []
        for f in features:
            if f.lower() not in seen:
                seen.add(f.lower())
                unique_features.append(f)
        
        # Return what we found - if empty, async AI generation will handle it
        return unique_features[:5]
    
    async def _phase_4_5_cac_analysis(self, batch_context: List[str], batch_size: int) -> Optional[List[Dict]]:
        """Sub-call 1 for unit economics: CAC breakdown analysis."""
        prompt = f"""
        You have access to Google Search. Calculate Customer Acquisition Cost (CAC) breakdown.
        
        Solutions to analyze:
        {chr(10).join(batch_context)}
        
        For EACH solution, research and calculate:
        
        1. Customer Acquisition Cost (CAC) breakdown:
           - Search "[industry] CAC benchmarks 2024"
           - "B2B SaaS customer acquisition cost [segment]"
           - "Cost per lead [channel] [industry]"
           - "Sales cycle length [customer type]"
           
           Calculate CAC = (Marketing costs + Sales costs) / New customers
           Include: Ad spend, content, sales team, tools
        
        Return CAC analysis:
        [
            {{
                "index": 1,
                "cac_breakdown": {{
                    "total": "$450",
                    "channels": [
                        {{"channel": "Google Ads", "cost": "$200", "conversion": "2%"}},
                        {{"channel": "Content marketing", "cost": "$150", "conversion": "5%"}},
                        {{"channel": "Sales outreach", "cost": "$100", "conversion": "10%"}}
                    ],
                    "blended_cac": "$450 (3-month average)"
                }}
            }}
        ]
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "cac_breakdown": {
                        "type": "object",
                        "properties": {
                            "total": {"type": "string"},
                            "channels": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "channel": {"type": "string"},
                                        "cost": {"type": "string"},
                                        "conversion": {"type": "string"}
                                    }
                                }
                            },
                            "blended_cac": {"type": "string"}
                        }
                    }
                },
                "required": ["index", "cac_breakdown"]
            },
            "maxItems": batch_size
        }
        
        self.api_calls += 1
        return await self._call_gemini_structured(prompt, schema)
    
    async def _phase_4_5_ltv_margins(self, batch_context: List[str], batch_size: int) -> Optional[List[Dict]]:
        """Sub-call 2 for unit economics: LTV calculation and margins."""
        prompt = f"""
        You have access to Google Search. Calculate Lifetime Value (LTV) and gross margins.
        
        Solutions to analyze:
        {chr(10).join(batch_context)}
        
        For EACH solution, research and calculate:
        
        1. Lifetime Value (LTV) calculation:
           - "[industry] customer churn rate SaaS"
           - "[segment] average customer lifespan"
           - "SaaS expansion revenue benchmarks"
           - "Upsell rate [industry] B2B"
           
           LTV = ARPU × (1/Churn rate) × Gross margin
        
        2. Payback period analysis:
           - "SaaS payback period benchmarks 2024"
           - "[industry] cash flow positive timeline"
        
        3. Gross margins by component:
           - Infrastructure/hosting costs
           - API costs per transaction
           - Support costs per customer
           - Payment processing fees
        
        Return LTV and margin analysis:
        [
            {{
                "index": 1,
                "ltv_calculation": {{
                    "arpu": "$150/month",
                    "churn_rate": "5% monthly",
                    "customer_lifespan": "20 months",
                    "gross_margin": "85%",
                    "ltv": "$2,550",
                    "ltv_cac_ratio": "5.7:1"
                }},
                "payback_period": "3 months (top quartile for B2B SaaS)",
                "gross_margin_analysis": {{
                    "overall": "85%",
                    "breakdown": [
                        {{"component": "Infrastructure", "cost": "5%"}},
                        {{"component": "APIs", "cost": "7%"}},
                        {{"component": "Support", "cost": "3%"}}
                    ]
                }}
            }}
        ]
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "ltv_calculation": {
                        "type": "object",
                        "properties": {
                            "arpu": {"type": "string"},
                            "churn_rate": {"type": "string"},
                            "customer_lifespan": {"type": "string"},
                            "gross_margin": {"type": "string"},
                            "ltv": {"type": "string"},
                            "ltv_cac_ratio": {"type": "string"}
                        }
                    },
                    "payback_period": {"type": "string"},
                    "gross_margin_analysis": {
                        "type": "object",
                        "properties": {
                            "overall": {"type": "string"},
                            "breakdown": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component": {"type": "string"},
                                        "cost": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                },
                "required": ["index", "ltv_calculation", "payback_period", "gross_margin_analysis"]
            },
            "maxItems": batch_size
        }
        
        self.api_calls += 1
        return await self._call_gemini_structured(prompt, schema)
    
    async def _phase_4_5_growth_metrics(self, batch_context: List[str], batch_size: int) -> Optional[List[Dict]]:
        """Sub-call 3 for unit economics: Growth metrics and path to profitability."""
        prompt = f"""
        You have access to Google Search. Calculate growth efficiency and path to profitability.
        
        Solutions to analyze:
        {chr(10).join(batch_context)}
        
        For EACH solution, research and calculate:
        
        1. Growth efficiency metrics:
           - "SaaS magic number benchmarks"
           - "Rule of 40 [industry] standards"
           - "Burn multiple good vs bad"
           - Capital efficiency ratios
        
        2. Funding requirements:
           - Runway to profitability
           - Working capital needs
           - Growth capital requirements
           - Bootstrap vs VC path analysis
        
        3. Competitive benchmarks:
           - Similar companies' unit economics
           - Industry standard metrics
        
        Return growth and profitability analysis:
        [
            {{
                "index": 1,
                "unit_economics_score": 8.5,
                "growth_metrics": {{
                    "magic_number": "1.2 (efficient growth)",
                    "rule_of_40": "Growth rate 45% + Margin 15% = 60 (excellent)",
                    "burn_multiple": "0.8x (capital efficient)"
                }},
                "path_to_profitability": {{
                    "months": 12,
                    "customers_needed": "200 @ $150/mo",
                    "monthly_burn": "$15K decreasing to $0",
                    "funding_required": "$200K to reach profitability"
                }},
                "competitive_benchmarks": "Similar tools achieve 6:1 LTV:CAC, we project 5.7:1"
            }}
        ]
        
        Score 0-10 based on:
        - LTV:CAC ratio > 3:1 (higher is better)
        - Payback period < 12 months
        - Gross margins > 70%
        - Clear path to profitability
        - Capital efficiency metrics
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "unit_economics_score": {"type": "number"},
                    "growth_metrics": {
                        "type": "object",
                        "properties": {
                            "magic_number": {"type": "string"},
                            "rule_of_40": {"type": "string"},
                            "burn_multiple": {"type": "string"}
                        }
                    },
                    "path_to_profitability": {
                        "type": "object",
                        "properties": {
                            "months": {"type": "number"},
                            "customers_needed": {"type": "string"},
                            "monthly_burn": {"type": "string"},
                            "funding_required": {"type": "string"}
                        }
                    },
                    "competitive_benchmarks": {"type": "string"}
                },
                "required": ["index", "unit_economics_score", "growth_metrics", "path_to_profitability"]
            },
            "maxItems": batch_size
        }
        
        self.api_calls += 1
        return await self._call_gemini_structured(prompt, schema)
    
    async def _phase_4_75_mvp_features(self, batch_context: List[str], batch_size: int) -> Optional[List[Dict]]:
        """Sub-call 1 for MVP simulation: Core features and launch strategy."""
        prompt = f"""
        You have access to Google Search. Determine MVP features and launch strategy.
        
        Solutions to simulate:
        {chr(10).join(batch_context)}
        
        For EACH solution, research and determine:
        
        1. MVP feature prioritization:
           - Search "MVP feature prioritization [industry]"
           - "[solution type] core features must have"
           - "minimum viable product examples [segment]"
           - Select 3-5 ESSENTIAL features for Day 1
           - Features that directly enable revenue generation
        
        2. Go-to-market strategy:
           - "[industry] product launch strategies"
           - "zero to first 10 customers [B2B/B2C]"
           - "[segment] early adopter acquisition"
           - "product hunt launch best practices"
           - "beta user acquisition tactics"
        
        3. Resource requirements:
           - Team size and skills needed
           - Budget for first 3 months
           - Time to launch MVP
        
        Return MVP features and strategy:
        [
            {{
                "index": 1,
                "mvp_viability_score": 8.5,
                "mvp_core_features": [
                    "Automated transaction processing",
                    "Basic API integration",
                    "Real-time dashboard",
                    "Stripe payment integration",
                    "Email notifications"
                ],
                "feature_rationale": "These 5 features enable immediate value and revenue",
                "launch_strategy": {{
                    "primary_channel": "Product Hunt + targeted outreach",
                    "launch_sequence": [
                        "Week -2: Beta with 10 hand-picked users",
                        "Week 0: Product Hunt launch",
                        "Week 1: Reddit community engagement",
                        "Week 2: Content marketing push"
                    ],
                    "expected_signups": "200 in first week, 500 by week 4"
                }},
                "resource_requirements": {{
                    "team": "2 developers, 1 designer/marketer",
                    "budget": "$10K for first 3 months",
                    "time_to_launch": "4 weeks from today"
                }}
            }}
        ]
        
        Score 0-10 based on:
        - Speed to market (can launch in < 6 weeks)
        - Clear value proposition in core features
        - Feasible launch strategy
        - Reasonable resource requirements
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "mvp_viability_score": {"type": "number"},
                    "mvp_core_features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 5
                    },
                    "feature_rationale": {"type": "string"},
                    "launch_strategy": {
                        "type": "object",
                        "properties": {
                            "primary_channel": {"type": "string"},
                            "launch_sequence": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "expected_signups": {"type": "string"}
                        }
                    },
                    "resource_requirements": {
                        "type": "object",
                        "properties": {
                            "team": {"type": "string"},
                            "budget": {"type": "string"},
                            "time_to_launch": {"type": "string"}
                        }
                    }
                },
                "required": ["index", "mvp_viability_score", "mvp_core_features", "launch_strategy", "resource_requirements"]
            },
            "maxItems": batch_size
        }
        
        self.api_calls += 1
        return await self._call_gemini_structured(prompt, schema)
    
    async def _phase_4_75_revenue_projections(self, batch_context: List[str], batch_size: int) -> Optional[List[Dict]]:
        """Sub-call 2 for MVP simulation: Revenue projections and conversion funnel."""
        prompt = f"""
        You have access to Google Search. Project revenue growth and conversion metrics.
        
        Solutions to simulate:
        {chr(10).join(batch_context)}
        
        For EACH solution, research and project:
        
        1. Week-by-week revenue simulation (12 weeks):
           - "SaaS growth benchmarks first 90 days"
           - "[industry] customer acquisition velocity"
           - "viral coefficient [product type]"
           - "conversion funnel benchmarks [segment]"
        
        2. Launch channel effectiveness:
           - "[channel] conversion rates for MVP"
           - "cost per acquisition by channel [industry]"
           - "time to first customer by channel"
        
        3. Conversion funnel metrics:
           - Visitor to trial conversion rates
           - Trial to paid conversion
           - Activation rates
           - Time to value metrics
        
        Return revenue projections:
        [
            {{
                "index": 1,
                "revenue_projection_weekly": [
                    {{"week": 1, "customers": 0, "mrr": "$0", "trials": 50}},
                    {{"week": 2, "customers": 5, "mrr": "$500", "trials": 100}},
                    {{"week": 3, "customers": 12, "mrr": "$1,200", "trials": 150}},
                    {{"week": 4, "customers": 25, "mrr": "$2,500", "trials": 200}},
                    {{"week": 6, "customers": 50, "mrr": "$5,000", "trials": 300}},
                    {{"week": 8, "customers": 75, "mrr": "$7,500", "trials": 400}},
                    {{"week": 10, "customers": 100, "mrr": "$10,000", "trials": 500}},
                    {{"week": 12, "customers": 130, "mrr": "$13,000", "trials": 600}}
                ],
                "conversion_funnel": {{
                    "visitor_to_trial": "8% (industry avg 5%)",
                    "trial_to_paid": "25% (industry avg 20%)",
                    "activation_rate": "60% complete onboarding",
                    "time_to_value": "< 10 minutes to first transaction"
                }}
            }}
        ]
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "revenue_projection_weekly": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "week": {"type": "integer"},
                                "customers": {"type": "integer"},
                                "mrr": {"type": "string"},
                                "trials": {"type": "integer"}
                            }
                        },
                        "maxItems": 12
                    },
                    "conversion_funnel": {
                        "type": "object",
                        "properties": {
                            "visitor_to_trial": {"type": "string"},
                            "trial_to_paid": {"type": "string"},
                            "activation_rate": {"type": "string"},
                            "time_to_value": {"type": "string"}
                        }
                    }
                },
                "required": ["index", "revenue_projection_weekly", "conversion_funnel"]
            },
            "maxItems": batch_size
        }
        
        self.api_calls += 1
        return await self._call_gemini_structured(prompt, schema)
    
    async def _phase_4_75_traction_indicators(self, batch_context: List[str], batch_size: int) -> Optional[List[Dict]]:
        """Sub-call 3 for MVP simulation: Traction indicators and pivot triggers."""
        prompt = f"""
        You have access to Google Search. Identify early traction indicators and risk factors.
        
        Solutions to simulate:
        {chr(10).join(batch_context)}
        
        For EACH solution, research and identify:
        
        1. Early customer feedback loops:
           - "MVP iteration cycles [industry]"
           - "feature request prioritization frameworks"
           - "customer success metrics early stage"
        
        2. Early traction indicators:
           - Key metrics that indicate product-market fit
           - User engagement benchmarks
           - Growth velocity indicators
           - Quality signals from early users
        
        3. Risk mitigation:
           - "MVP failure reasons [industry]"
           - "early warning signs product market fit"
           - "pivot indicators for startups"
           - Clear pivot triggers if metrics aren't met
        
        Return traction and risk analysis:
        [
            {{
                "index": 1,
                "early_traction_indicators": [
                    "50+ daily active users by week 2",
                    "NPS score > 50 from beta users",
                    "< 5% churn in first month",
                    "3+ feature requests per user"
                ],
                "pivot_triggers": [
                    "Trial-to-paid < 15% after 500 trials",
                    "Churn > 10% monthly",
                    "CAC > $500 after optimization",
                    "Time to value > 1 hour"
                ]
            }}
        ]
        """
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "early_traction_indicators": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "pivot_triggers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["index", "early_traction_indicators", "pivot_triggers"]
            },
            "maxItems": batch_size
        }
        
        self.api_calls += 1
        return await self._call_gemini_structured(prompt, schema)
    
    def _save_phase_data(self, phase_name: str, opportunities: List[Dict]):
        """Save phase data with hybrid approach based on scores.
        
        Args:
            phase_name: Name of the phase (e.g., 'phase_2_problems_validated')
            opportunities: List of opportunities with their scores
        """
        phase_data = []
        
        for opp in opportunities:
            opp_id = opp.get('id', 'unknown')
            
            # Get the score for this phase
            score = 0.0
            if opp_id in self.scores:
                # Map phase names to score keys
                phase_score_map = {
                    'phase_0_market_timing': None,  # No scores for phase 0
                    'phase_1_problems_discovered': None,  # No scores for phase 1
                    'phase_2_problems_validated': 'phase_2',
                    'phase_2_5_user_research': 'phase_2_5',
                    'phase_2_75_revenue_validation': 'phase_2_75',
                    'phase_3_competition_analysis': 'phase_3',
                    'phase_3_5_technical_validation': 'phase_3_5',
                    'phase_4_market_sizing': 'phase_4',
                    'phase_4_5_unit_economics': 'phase_4_5',
                    'phase_4_75_mvp_simulation': 'phase_4_75',
                    'phase_5_final_selection': None  # Uses composite score
                }
                
                score_key = phase_score_map.get(phase_name)
                if score_key and score_key in self.scores[opp_id]:
                    score = self.scores[opp_id][score_key]
                elif phase_name == 'phase_5_final_selection' and 'final_score' in opp:
                    score = opp.get('final_score', 0)
            
            # Hybrid approach: save different levels of detail based on score
            if score >= 7.0:  # High scorers - save full details
                phase_entry = {
                    'id': opp_id,
                    'score': score,
                    'status': 'high_potential',
                    'problem': opp.get('problem_description', ''),
                    'audience': opp.get('affected_audience', ''),
                    'details': {k: v for k, v in opp.items() if k not in ['id', 'problem_description', 'affected_audience']}
                }
            elif score >= 5.0:  # Medium scorers - save summary only
                phase_entry = {
                    'id': opp_id,
                    'score': score,
                    'status': 'medium_potential',
                    'problem': opp.get('problem_description', '')[:100] + '...',
                    'audience': opp.get('affected_audience', ''),
                    'key_metrics': {
                        'validation_score': opp.get('validation_score', 0),
                        'user_pain_score': opp.get('user_pain_score', 0),
                        'revenue_score': opp.get('revenue_score', 0),
                        'competition_score': opp.get('competition_score', 0),
                        'technical_score': opp.get('technical_score', 0),
                        'market_score': opp.get('market_score', 0)
                    }
                }
            else:  # Low scorers - minimal data
                phase_entry = {
                    'id': opp_id,
                    'score': score,
                    'status': 'rejected',
                    'problem_summary': opp.get('problem_description', '')[:50] + '...',
                    'rejection_phase': phase_name
                }
            
            phase_data.append(phase_entry)
        
        # Store in research_phases_data
        if phase_name in self.research_phases_data:
            self.research_phases_data[phase_name] = phase_data
        
        # Log summary
        high_count = len([p for p in phase_data if p.get('status') == 'high_potential'])
        medium_count = len([p for p in phase_data if p.get('status') == 'medium_potential'])
        rejected_count = len([p for p in phase_data if p.get('status') == 'rejected'])
        
        self.logger.info(
            f"Saved {phase_name} data: {high_count} high potential, "
            f"{medium_count} medium potential, {rejected_count} rejected"
        )
    
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
    
    async def _phase_6_pre_build_validation(self, opportunities: List[MarketOpportunity]) -> Dict[str, Any]:
        """Final pre-build validation checklist to ensure opportunity is ready for development."""
        self.logger.info(f"🎯 Phase 6: Pre-Build Validation for {len(opportunities)} selected opportunities")
        
        # Build context for validation
        opp_contexts = []
        for i, opp in enumerate(opportunities):
            context = f"{i+1}. {opp.project_name or 'Unnamed Project'}\n"
            context += f"   Problem: {opp.problem_description[:200]}\n"
            context += f"   Solution: {opp.opportunity[:200]}\n"
            context += f"   Success Probability: {opp.search_evidence.get('success_probability', 0):.1f}%\n"
            context += f"   MVP Timeline: {opp.implementation_difficulty}\n"
            context += f"   Revenue Potential: {opp.potential_revenue}\n"
            context += f"   Key Features: {len(opp.key_features)}"
            opp_contexts.append(context)
        
        prompt = f"""
        You have access to Google Search. Perform FINAL pre-build validation checks.
        
        Selected opportunities:
        {chr(10).join(opp_contexts)}
        
        For EACH opportunity, validate:
        
        1. Domain availability check:
           - Search "[project name].com whois"
           - "[project name] domain availability"
           - Check .com, .io, .ai availability
           - Trademark conflicts
        
        2. Direct competitor launch check:
           - "[exact solution] launched 2024 2025"
           - "new [problem area] startup funding"
           - Recent Product Hunt launches
           - YC recent batches in this space
        
        3. Technical dependencies validation:
           - Critical API availability and stability
           - Required services pricing changes
           - Technology deprecation notices
           - Security vulnerabilities in stack
        
        4. Regulatory compliance check:
           - New regulations affecting this space
           - Required licenses or certifications
           - Data privacy requirements
           - Industry-specific compliance
        
        5. Team capability assessment:
           - Required expertise gaps
           - Time commitment reality check
           - Resource availability
        
        6. First customer acquisition strategy:
           - Specific first 10 customers identified
           - Clear outreach plan
           - Value proposition validation
           - Pricing strategy confirmation
        
        Return validation results:
        {{
            "opportunities_validated": [
                {{
                    "project_name": "ProjectName",
                    "validation_status": "GREEN/YELLOW/RED",
                    "domain_availability": {{
                        "com": true/false,
                        "alternatives": ["projectname.io", "getprojectname.com"],
                        "trademark_clear": true/false
                    }},
                    "competition_update": "No new direct competitors. Similar tool raised $5M last month.",
                    "technical_risks": ["OpenAI API costs increasing 20%", "None critical"],
                    "regulatory_issues": ["None", "May need SOC2 for enterprise"],
                    "go_no_go_recommendation": "GO - All systems green",
                    "critical_first_steps": [
                        "Secure domain projectname.io",
                        "Set up landing page with waitlist",
                        "Reach out to 5 beta users from Reddit community"
                    ],
                    "30_day_milestones": [
                        "100 waitlist signups",
                        "5 beta user interviews completed",
                        "MVP with 3 core features live",
                        "First paying customer"
                    ]
                }}
            ],
            "overall_recommendation": "PROCEED with 1 opportunity, postpone 0, cancel 0",
            "key_risks_identified": ["API cost increases could impact margins"],
            "success_factors": ["Strong user pain validation", "Clear monetization path", "Low competition"]
        }}
        """
        
        schema = {
            "type": "object",
            "properties": {
                "opportunities_validated": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "project_name": {"type": "string"},
                            "validation_status": {
                                "type": "string",
                                "enum": ["GREEN", "YELLOW", "RED"]
                            },
                            "domain_availability": {
                                "type": "object",
                                "properties": {
                                    "com": {"type": "boolean"},
                                    "alternatives": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "trademark_clear": {"type": "boolean"}
                                }
                            },
                            "competition_update": {"type": "string"},
                            "technical_risks": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "regulatory_issues": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "go_no_go_recommendation": {"type": "string"},
                            "critical_first_steps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 5
                            },
                            "30_day_milestones": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 5
                            }
                        },
                        "required": ["project_name", "validation_status", "domain_availability",
                                    "go_no_go_recommendation", "critical_first_steps", "30_day_milestones"]
                    }
                },
                "overall_recommendation": {"type": "string"},
                "key_risks_identified": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "success_factors": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["opportunities_validated", "overall_recommendation", 
                        "key_risks_identified", "success_factors"]
        }
        
        self.api_calls += 1
        validation_result = await self._call_gemini_structured(prompt, schema)
        
        if validation_result:
            # Log validation results
            self.logger.info("📋 Pre-Build Validation Results:")
            for val in validation_result.get('opportunities_validated', []):
                self.logger.info(f"  - {val['project_name']}: {val['validation_status']}")
                self.logger.info(f"    Recommendation: {val['go_no_go_recommendation']}")
                if val.get('domain_availability', {}).get('com'):
                    self.logger.info(f"    ✅ Domain {val['project_name'].lower()}.com available!")
                else:
                    alts = val.get('domain_availability', {}).get('alternatives', [])
                    if alts:
                        self.logger.info(f"    🌐 Alternative domains: {', '.join(alts[:2])}")
            
            self.logger.info(f"\n🎯 Overall: {validation_result.get('overall_recommendation', 'Unknown')}")
            
            return validation_result
        
        return {}


# Make it compatible with existing system
GeminiDeepResearch = GeminiDeepResearchOptimized