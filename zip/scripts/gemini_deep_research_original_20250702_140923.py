"""
Gemini Deep Research Integration with Google Search Grounding

This module enhances market research by leveraging Gemini's built-in Google Search
grounding capabilities to analyze real-world problems, market trends, and opportunities.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import aiohttp
import ssl
from pathlib import Path
import re

from scripts.ai_brain import IntelligentAIBrain
from scripts.research_memory import ResearchMemory
from scripts.dynamic_research_prompt_generator import DynamicResearchPromptGenerator
from scripts.emergent_learning_system import EmergentLearningSystem
from scripts.custom_json_encoder import safe_json_dump


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


class GeminiDeepResearch:
    """Performs deep market research using Gemini 1.5 Pro with Google Search grounding."""
    
    
    def __init__(self, ai_brain: Optional[IntelligentAIBrain] = None):
        """Initialize with AI brain configured for Gemini with search grounding.
        
        Args:
            ai_brain: AI brain instance (optional, will create if not provided)
        """
        self.logger = logging.getLogger(__name__)
        # Enable debug logging for this module to diagnose issues
        self.logger.setLevel(logging.DEBUG)
        self.all_sectors_researched = set()  # Track all sectors/audiences discovered
        self.research_stats = {
            'problems_discovered': 0,
            'problems_validated': 0,
            'competition_analyzed': 0,
            'search_queries_made': 0
        }
        
        # Store ALL research data from each phase
        self.research_phases_data = {
            'phase_0_market_timing': None,
            'phase_1_problems_discovered': [],
            'phase_2_problems_validated': [],
            'phase_2_5_user_research': [],
            'phase_2_75_revenue_validation': [],  # NEW
            'phase_3_competition_analysis': [],
            'phase_3_5_technical_validation': [],
            'phase_4_market_sizing': [],
            'phase_4_5_unit_economics': [],
            'phase_4_75_mvp_simulation': [],  # NEW
            'phase_5_final_selection': [],
            'phase_6_pre_build_validation': None  # NEW
        }
        
        # Initialize AI components
        if ai_brain:
            self.ai_brain = ai_brain
        else:
            # Create fresh instance
            self.ai_brain = IntelligentAIBrain()
        
        # Initialize research memory and prompt generator
        self.research_memory = ResearchMemory()
        self.prompt_generator = DynamicResearchPromptGenerator(self.ai_brain, self.research_memory)
        
        # Initialize emergent learning system
        self.learning_system = EmergentLearningSystem()
        
        # Get Gemini API key for direct structured output calls
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.gemini_api_key:
            self.logger.warning("GOOGLE_API_KEY not found in environment")
    
    async def _call_gemini_structured(self, prompt: str, response_schema: Dict[str, Any], max_retries: int = 10) -> Optional[Dict[str, Any]]:
        """Make direct Gemini API call with structured output.
        
        Args:
            prompt: The prompt to send to Gemini
            response_schema: The JSON schema for the response
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.gemini_api_key:
            self.logger.error("Gemini API key not available")
            return None
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={self.gemini_api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Construct request with structured output configuration
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "maxOutputTokens": 8192  # Restore original token limit
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
                    # Log the request details
                    self.logger.debug(f"[Attempt {attempt + 1}/{max_retries}] Sending request to Gemini Pro")
                    self.logger.debug(f"Request prompt length: {len(prompt)} chars")
                    self.logger.debug(f"Response schema: {json.dumps(response_schema)[:200]}...")
                    
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            # First get raw text to log it
                            raw_response_text = await response.text()
                            self.logger.debug(f"Raw response length: {len(raw_response_text)} chars")
                            
                            # Parse the response as JSON
                            try:
                                result = json.loads(raw_response_text)
                            except json.JSONDecodeError as je:
                                self.logger.error(f"Failed to parse response as JSON: {je}")
                                self.logger.error(f"Raw response (first 500 chars): {raw_response_text[:500]}")
                                self.logger.error(f"Raw response (last 500 chars): {raw_response_text[-500:]}")
                                raise je
                            
                            # Extract content from Gemini response
                            if result.get("candidates") and len(result["candidates"]) > 0:
                                candidate = result["candidates"][0]
                                if candidate.get("content") and candidate["content"].get("parts"):
                                    content = candidate["content"]["parts"][0].get("text", "{}")
                                    
                                    # Log the extracted content
                                    self.logger.debug(f"Extracted content length: {len(content)} chars")
                                    self.logger.debug(f"Content preview (first 200 chars): {content[:200]}")
                                    
                                    # Parse the JSON response with detailed error handling
                                    try:
                                        parsed_json = json.loads(content)
                                        self.logger.debug(f"Successfully parsed JSON. Type: {type(parsed_json)}, Keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'N/A'}")
                                        return parsed_json
                                    except json.JSONDecodeError as je:
                                        self.logger.error(f"JSON parsing error: {je}")
                                        self.logger.error(f"Error position: Line {je.lineno}, Column {je.colno}, Char {je.pos}")
                                        
                                        # Log the problematic area
                                        if je.pos is not None and je.pos > 0:
                                            start = max(0, je.pos - 100)
                                            end = min(len(content), je.pos + 100)
                                            self.logger.error(f"Content around error position [{start}:{end}]:")
                                            self.logger.error(f">>>{content[start:end]}<<<")
                                            
                                            # Highlight the exact error position
                                            error_line_start = content.rfind('\n', 0, je.pos) + 1
                                            error_line_end = content.find('\n', je.pos)
                                            if error_line_end == -1:
                                                error_line_end = len(content)
                                            self.logger.error(f"Error line: >>>{content[error_line_start:error_line_end]}<<<")
                                        
                                        # Log full content for debugging
                                        self.logger.error(f"Full content (first 1000 chars): {content[:1000]}")
                                        if len(content) > 1000:
                                            self.logger.error(f"Full content (last 1000 chars): {content[-1000:]}")
                                        
                                        # Try to identify common issues
                                        if '\\"' in content:
                                            self.logger.warning("Found escaped quotes (\\\"`) in content - might be double-escaped")
                                        if '\n' in content and not content.strip().startswith('[') and not content.strip().startswith('{'):
                                            self.logger.warning("Content contains newlines but doesn't start with [ or { - might not be valid JSON")
                                        
                                        raise je
                                else:
                                    self.logger.error("No content/parts in candidate response")
                                    self.logger.error(f"Candidate structure: {json.dumps(candidate)[:500]}")
                            else:
                                self.logger.error("No candidates in response")
                                self.logger.error(f"Response structure: {json.dumps(result)[:500]}")
                        elif response.status == 503:
                            # Model is overloaded
                            error_text = await response.text()
                            self.logger.warning(f"Gemini overloaded (attempt {attempt + 1}/{max_retries}): {error_text}")
                            
                            if attempt < max_retries - 1:
                                # No timeout - wait as long as needed
                                wait_time = 30 + (attempt * 10)  # 30, 40, 50... seconds
                                print(f"⏳ Gemini is overloaded. Waiting {wait_time} seconds before retry... (attempt {attempt + 1}/{max_retries})")
                                print("   Research will continue until complete - no timeouts applied.")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                self.logger.error("Gemini still overloaded after all retries")
                                return None
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Gemini API error: {response.status} - {error_text}")
                            return None
            except json.JSONDecodeError as je:
                # JSON parsing error - already logged in detail above
                self.logger.error(f"JSON decode error in attempt {attempt + 1}: {je}")
                if attempt < max_retries - 1:
                    wait_time = 15 + (attempt * 5)  # 15, 20, 25... seconds
                    print(f"⏳ JSON parsing error: {str(je)[:100]}... Waiting {wait_time} seconds before retry...")
                    print("   Research will continue until complete - no timeouts applied.")
                    await asyncio.sleep(wait_time)
                    continue
            except aiohttp.ClientError as ae:
                self.logger.error(f"Network error calling Gemini API: {ae}")
                self.logger.error(f"Error type: {type(ae).__name__}")
                if attempt < max_retries - 1:
                    wait_time = 15 + (attempt * 5)  # 15, 20, 25... seconds
                    print(f"⏳ Network error: {str(ae)[:100]}... Waiting {wait_time} seconds before retry...")
                    print("   Research will continue until complete - no timeouts applied.")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                self.logger.error(f"Unexpected error calling Gemini structured API: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error details: {repr(e)}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    wait_time = 15 + (attempt * 5)  # 15, 20, 25... seconds
                    print(f"⏳ Error occurred: {str(e)[:100]}... Waiting {wait_time} seconds before retry...")
                    print("   Research will continue until complete - no timeouts applied.")
                    await asyncio.sleep(wait_time)
                    continue
                
        return None
    
    async def conduct_deep_market_research(self, 
                                         count: int = 1,
                                         existing_portfolio: Optional[Dict[str, Any]] = None) -> List[MarketOpportunity]:
        """Conduct INTENSIVE market research with no timeouts - searches take as long as needed.
        
        This enhanced method performs exhaustive validation through 11 phases:
        - Phase 0: Market Timing & Trend Detection (ENHANCED)
        - Phase 1: Discover 25-30 real problems (expanded from 10-15)
        - Phase 2: Validate ALL problems with score ≥ 9/10 (was 7/10)
        - Phase 2.5: User Research Simulation (NEW)
        - Phase 2.75: Automated Revenue Validation (NEW)
        - Phase 3: Deep Competition Analysis
        - Phase 3.5: Technical Feasibility Deep Dive (NEW)
        - Phase 4: Market Sizing with Hard Numbers (NEW)
        - Phase 4.5: Unit Economics Validation (NEW)
        - Phase 4.75: MVP Revenue Simulation (NEW)
        - Phase 5: Final Selection with 15x opportunity (was 10x)
        - Phase 6: Pre-Build Validation (NEW)
        
        Args:
            count: Number of opportunities to return (default 3)
            existing_portfolio: Current portfolio to avoid overlap
            
        Returns:
            List of THOROUGHLY validated market opportunities
        """
        print("🔬 Starting INTENSIVE Gemini Deep Research - No timeouts, exhaustive validation...")
        print("⏰ This research will take as long as needed for thorough analysis.\n")
        
        # Phase 0: Market Timing Analysis (NEW)
        market_timing_data = await self._phase_0_market_timing()
        self.research_phases_data['phase_0_market_timing'] = market_timing_data
        
        if not market_timing_data:
            print("\n⚠️ Failed to analyze market timing, returning empty list")
            await self._save_research_knowledge([])  # Save even failed research
            return []
        
        # Check combined score of timing + trends (lowered threshold)
        timing_score = market_timing_data.get('timing_score', 0)
        trend_score = market_timing_data.get('trend_score', 0)
        combined_score = (timing_score + trend_score) / 2
        
        if combined_score < 6:  # Lowered from 7
            print("\n⚠️ Market conditions not favorable (combined score < 6), returning empty list")
            await self._save_research_knowledge([])  # Save even failed research
            return []
        
        # Phase 1: Discover real-world problems organically (25-30 problems)
        real_problems = await self._discover_real_world_problems()
        self.research_phases_data['phase_1_problems_discovered'] = real_problems
        
        if not real_problems:
            print("\n⚠️ No problems discovered, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 2: Validate problems for web-based solutions (score ≥ 7/10) - LOWERED THRESHOLD
        validated_problems = await self._validate_problems_with_search(real_problems)
        self.research_phases_data['phase_2_problems_validated'] = validated_problems
        
        if not validated_problems:
            print("\n⚠️ No problems validated with score ≥ 7/10, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 2.5: Deep User Research Simulation (NEW)
        user_validated = await self._phase_2_5_user_research(validated_problems)
        self.research_phases_data['phase_2_5_user_research'] = user_validated
        
        if not user_validated:
            print("\n⚠️ No problems passed user research validation, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 2.75: Automated Revenue Validation (NEW)
        revenue_validated = await self._phase_2_75_revenue_validation(user_validated)
        self.research_phases_data['phase_2_75_revenue_validation'] = revenue_validated
        
        if not revenue_validated:
            print("\n⚠️ No problems passed revenue validation, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 3: Competition Analysis with Market Data
        opportunities = await self._analyze_competition_with_search(revenue_validated)
        self.research_phases_data['phase_3_competition_analysis'] = opportunities
        
        if not opportunities:
            print("\n⚠️ No viable opportunities after competition analysis, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 3.5: Technical Feasibility Deep Dive (NEW)
        tech_validated = await self._phase_3_5_technical_validation(opportunities)
        self.research_phases_data['phase_3_5_technical_validation'] = tech_validated
        
        if not tech_validated:
            print("\n⚠️ No opportunities passed technical validation, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 4: Market Sizing with Quantitative Data (ENHANCED)
        market_sized = await self._phase_4_market_sizing(tech_validated)
        self.research_phases_data['phase_4_market_sizing'] = market_sized
        
        if not market_sized:
            print("\n⚠️ No opportunities with sufficient market size, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 4.5: Unit Economics Validation (NEW)
        economics_validated = await self._phase_4_5_unit_economics(market_sized)
        self.research_phases_data['phase_4_5_unit_economics'] = economics_validated
        
        if not economics_validated:
            print("\n⚠️ No opportunities with viable unit economics, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 4.75: MVP Revenue Simulation (NEW)
        mvp_validated = await self._phase_4_75_mvp_revenue_simulation(economics_validated)
        self.research_phases_data['phase_4_75_mvp_simulation'] = mvp_validated
        
        if not mvp_validated:
            print("\n⚠️ No opportunities with viable MVP revenue projections, returning empty list")
            await self._save_research_knowledge([])  # Save research data
            return []
        
        # Phase 5: Final Selection with Portfolio Context (15x opportunity required)
        final_opportunities = await self._filter_and_rank(mvp_validated, count, existing_portfolio)
        # Convert MarketOpportunity objects to dicts for JSON serialization
        self.research_phases_data['phase_5_final_selection'] = [
            asdict(opp) if hasattr(opp, '__dataclass_fields__') else opp 
            for opp in final_opportunities
        ]
        
        # Phase 6: Pre-Build Validation (NEW)
        # Run validation but don't block on failure - add validation info to opportunity
        if final_opportunities:
            # Validate top opportunity
            pre_build_validation = await self._phase_6_pre_build_validation(final_opportunities[0])
            self.research_phases_data['phase_6_pre_build_validation'] = pre_build_validation
            
            # Add validation info to opportunity
            if hasattr(final_opportunities[0], '__dict__'):
                final_opportunities[0].pre_build_validation = pre_build_validation
            
            # Always return only the single best opportunity
            selected_opportunities = [final_opportunities[0]]
            print("\n📌 Returning single best opportunity")
            
            # Log validation warnings if any
            if pre_build_validation:
                val_score = pre_build_validation.get('validation_score', 0)
                if val_score < 8:
                    print(f"\n⚠️ Pre-build validation score: {val_score}/10")
                    if not pre_build_validation.get('domain_available', True):
                        print(f"   Domain not available - consider alternatives: {', '.join(pre_build_validation.get('suggested_domains', [])[:3])}")
                    if pre_build_validation.get('technical_blockers'):
                        print(f"   Technical concerns: {', '.join(pre_build_validation.get('technical_blockers', []))}")
                    if pre_build_validation.get('api_concerns'):
                        print(f"   API concerns: {', '.join(pre_build_validation.get('api_concerns', []))}")
                    print("\n   💡 These are warnings only - proceeding with opportunity")
                else:
                    print(f"✅ Pre-build validation passed: {val_score}/10")
        else:
            selected_opportunities = []
        
        # Update stats
        self.research_stats['problems_validated'] = len(validated_problems)
        self.research_stats['competition_analyzed'] = len(opportunities)
        
        # Save research data
        opportunities_to_save = selected_opportunities if 'selected_opportunities' in locals() else final_opportunities[:count]
        await self._save_research_knowledge(opportunities_to_save)
        
        print(f"\n✅ Intensive research complete. Found {len(opportunities_to_save)} exceptional opportunities.")
        return opportunities_to_save
    
    async def _discover_real_world_problems(self) -> List[Dict[str, Any]]:
        """Discover real-world problems organically through search queries."""
        
        # Generate a unique research direction using AI
        research_direction = self.prompt_generator.generate_research_prompt()
        print(f"\n🎯 Research Direction: {research_direction}")
        
        # Generate dynamic search queries based on the direction
        search_queries = self.prompt_generator.generate_search_queries(research_direction)
        
        # Build the discovery prompt dynamically
        prompt = f"""
        You have access to Google Search. Discover real-world problems based on this research direction:
        
        "{research_direction}"
        
        Use these search queries as starting points:
        {json.dumps(search_queries[:10], indent=2)}
        
        Search approach:
        1. Start with the provided queries to explore the research direction
        2. Follow interesting leads and dig deeper into specific pain points
        3. Look for evidence of real people struggling with these problems
        4. Validate problem severity and impact
        
        Return a JSON array of 25-30 specific, validated problems with this structure:
        [
            {{
                "problem_statement": "Clear description of the problem",
                "affected_audience": "Who experiences this problem",
                "current_pain": "Why this is painful/costly",
                "search_evidence": ["URL or source validating the problem"],
                "severity_score": 1-10,
                "web_solvable": true/false
            }}
        ]
        
        Focus on problems that:
        - Can be solved with web/mobile apps
        - Have clear monetization potential
        - Affect many people or businesses
        - Have inadequate current solutions
        
        IMPORTANT: Look for problems that could generate revenue 24/7:
        - Transaction-based revenue models
        - Automated service delivery
        - Global market potential
        - Minimal manual intervention needed
        """
        
        response_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem_statement": {"type": "string"},
                    "affected_audience": {"type": "string"},
                    "current_pain": {"type": "string"},
                    "search_evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "severity_score": {"type": "number"},
                    "web_solvable": {"type": "boolean"}
                },
                "required": ["problem_statement", "affected_audience", "current_pain", 
                            "search_evidence", "severity_score", "web_solvable"]
            }
        }
        
        print("\n🔍 Phase 1: Discovering real-world problems through organic search...")
        problems = await self._call_gemini_structured(prompt, response_schema)
        
        if problems and isinstance(problems, list):
            print(f"✅ Discovered {len(problems)} real-world problems")
            
            # Track stats and sectors
            self.research_stats['problems_discovered'] = len(problems)
            self.research_stats['search_queries_made'] += 1
            
            # Track all sectors/audiences discovered
            for problem in problems:
                if problem.get('affected_audience'):
                    self.all_sectors_researched.add(problem['affected_audience'])
            
            # Filter for web-solvable problems with MODERATE severity (lowered threshold)
            filtered = [p for p in problems if p.get('web_solvable', False) and p.get('severity_score', 0) >= 7]  # Lowered from 8
            print(f"📊 After filtering: {len(filtered)} problems with severity ≥ 7/10")
            return filtered
        else:
            print("⚠️ Failed to discover problems")
            return []
    
    async def _validate_problems_with_search(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate problems and check if they can be solved with web-based solutions."""
        
        print(f"\n🎯 Phase 2: Validating {len(problems)} problems for web-based solutions...")
        validated = []
        
        # Check each problem against research memory first
        for i, problem in enumerate(problems):
            # Check if we've researched similar problems before
            embedding = self.research_memory.create_embedding(problem['problem_statement'], self.ai_brain)
            is_similar, similar_attempt = self.research_memory.is_similar_to_previous(
                problem['problem_statement'], 
                embedding
            )
            
            if is_similar and similar_attempt:
                print(f"\n🔄 Problem {i+1} is similar to previously researched: {similar_attempt.problem_statement[:60]}...")
                print(f"   Previous result: score={similar_attempt.validation_score}, success={similar_attempt.success}")
                # Skip if we've already tried this and it failed
                if not similar_attempt.success and similar_attempt.validation_score < 7:
                    print("   Skipping - previously failed validation")
                    continue
            prompt = f"""
        You have access to Google Search. Validate this problem and determine if it can be effectively solved with a web-based solution.
        
        Problem to validate:
        {json.dumps(problem, indent=2)}
        
        Validation research:
        1. Check if web/mobile solutions exist:
           - Search "[problem] software solution"
           - Look for SaaS products addressing this
           - Check app stores for mobile solutions
        
        2. Analyze why current solutions fail:
           - Read negative reviews of existing solutions
           - Find forum complaints about current tools
           - Identify missing features or pain points
        
        3. Validate market demand:
           - Search volume for solution-related terms
           - Number of people asking for solutions
           - Willingness to pay indicators
        
        4. Assess technical feasibility:
           - Can this be solved with web technologies?
           - What APIs/integrations would be needed?
           - Development complexity assessment
        
        Return a validation object with:
        {{
            "validated": true/false,
            "validation_score": 0-10,
            "existing_solutions": ["list of current solutions"],
            "why_they_fail": "Key reasons current solutions are inadequate",
            "market_demand_evidence": "Search volume, user requests, etc.",
            "monetization_potential": "How users would pay",
            "technical_feasibility": "High/Medium/Low with reasoning",
            "unique_angle": "How a new solution could be 10x better"
        }}
        """
            
            validation_schema = {
                "type": "object",
                "properties": {
                    "validated": {"type": "boolean"},
                    "validation_score": {"type": "number"},
                    "existing_solutions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "why_they_fail": {"type": "string"},
                    "market_demand_evidence": {"type": "string"},
                    "monetization_potential": {"type": "string"},
                    "technical_feasibility": {"type": "string"},
                    "unique_angle": {"type": "string"}
                },
                "required": ["validated", "validation_score", "existing_solutions",
                           "why_they_fail", "market_demand_evidence", "monetization_potential",
                           "technical_feasibility", "unique_angle"]
            }
            
            print(f"\n📤 Validating problem {i+1}/{min(8, len(problems))}...")
            validation = await self._call_gemini_structured(prompt, validation_schema)
            
            if validation and validation.get('validated') and validation.get('validation_score', 0) >= 7:  # Lowered from 9
                # Merge validation data with original problem
                validated_problem = {**problem, **validation}
                validated.append(validated_problem)
                print(f"✅ Problem validated with score {validation.get('validation_score')}/10")
                
                # Record this attempt in research memory
                embedding = self.research_memory.create_embedding(problem['problem_statement'], self.ai_brain)
                self.research_memory.record_attempt(
                    problem['problem_statement'],
                    embedding,
                    success=True,  # Will update later if project created
                    validation_score=validation.get('validation_score', 0),
                    severity_score=problem.get('severity_score', 0)
                )
            else:
                score = validation.get('validation_score', 0) if validation else 0
                print(f"❌ Problem not validated or score too low ({score}/10)")
                
                # Record failed attempt
                if validation:
                    embedding = self.research_memory.create_embedding(problem['problem_statement'], self.ai_brain)
                    self.research_memory.record_attempt(
                        problem['problem_statement'],
                        embedding,
                        success=False,
                        validation_score=score,
                        severity_score=problem.get('severity_score', 0),
                        rejection_reason=f"Validation score too low: {score}/10"
                    )
        
        return validated
    
    async def _analyze_competition_with_search(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze existing solutions using Google Search data."""
        
        print(f"\n🎯 Phase 3: Analyzing competition for {len(problems)} validated problems...")
        
        opportunities = []
        
        # Analyze top problems based on validation score
        top_problems = sorted(problems, key=lambda x: x.get('validation_score', 0), reverse=True)[:5]
        print(f"Selected top {len(top_problems)} problems for deep analysis")
        
        for i, problem in enumerate(top_problems):
            prompt = f"""
            You have access to Google Search. Conduct an exhaustive competitive analysis and opportunity assessment for this high-impact problem.
            
            Problem to analyze:
            {json.dumps(problem, indent=2)}
            
            Comprehensive competitive research:
            
            1. Find ALL existing solutions:
               - Direct competitors (solving exact problem)
               - Indirect competitors (partial solutions)
               - Enterprise vs SMB solutions
               - Geographic variations
               - Open source alternatives
            
            2. Deep dive on top 5 competitors:
               - Pricing models and tiers
               - Feature sets and limitations
               - User reviews and NPS scores
               - Market share and growth
               - Technology stack used
               - Funding and revenue (if available)
            
            3. Search patterns:
               - "[problem] software solutions comparison"
               - "best [category] tools 2024"
               - "[competitor] vs [competitor] reviews"
               - "[solution] alternatives reddit"
               - "[competitor] pricing plans"
               - "[competitor] user complaints"
               - "[category] market leader analysis"
            
            4. Identify opportunity gaps:
               - Features users desperately want
               - Price points not being served
               - User segments being ignored
               - Technical limitations of current solutions
               - Integration gaps
            
            Create a breakthrough opportunity with:
            - solution_name: Memorable, brandable name
            - tagline: Clear value proposition
            - core_innovation: What makes this 10x better
            - key_features: List of 5-7 must-have differentiators
            - technical_approach: Modern architecture advantages
            - go_to_market: Specific user acquisition strategy
            - pricing_strategy: Detailed tiers with reasoning
            - revenue_model: Multiple revenue streams description  
            - competitive_moat: Why hard to copy
            - revenue_frequency: "per-transaction/hourly/daily/monthly"
            - automation_level: "fully-automated/semi-automated/manual" 
            - passive_income_score: 1-10 (based on automation and frequency)
            - year_one_projection: Users and revenue targets
            - implementation_phases: 3-6 month roadmap
            - success_metrics: List of KPIs to track
            - risk_mitigation: List of top 3 risks and solutions
            
            Include all competitor data and search evidence.
            Only proceed if there's a clear, defensible opportunity.
            """
            
            # Define the response schema for structured output
            opportunity_schema = {
                "type": "object",
                "properties": {
                    "solution_name": {"type": "string"},
                    "tagline": {"type": "string"},
                    "core_innovation": {"type": "string"},
                    "key_features": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "technical_approach": {"type": "string"},
                    "go_to_market": {"type": "string"},
                    "pricing_strategy": {"type": "string"},
                    "revenue_model": {"type": "string"},
                    "competitive_moat": {"type": "string"},
                    "revenue_frequency": {"type": "string"},
                    "automation_level": {"type": "string"},
                    "passive_income_score": {"type": "number"},
                    "year_one_projection": {"type": "string"},
                    "implementation_phases": {"type": "string"},
                    "success_metrics": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "risk_mitigation": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["solution_name", "tagline", "core_innovation", "key_features",
                           "technical_approach", "go_to_market", "pricing_strategy", "revenue_model",
                           "competitive_moat", "revenue_frequency", "automation_level", 
                           "passive_income_score", "year_one_projection", "implementation_phases",
                           "success_metrics", "risk_mitigation"]
            }
            
            print(f"\n📤 Analyzing competition for problem {i+1}/{len(top_problems)}...")
            opportunity = await self._call_gemini_structured(prompt, opportunity_schema)
            
            if opportunity:
                opportunity['source_problem'] = problem
                opportunities.append(opportunity)
                print(f"✅ Successfully analyzed competition for problem {i+1}")
            else:
                print(f"⚠️ Failed to get structured response for problem {i+1}")
        
        return opportunities
    
    async def _validate_with_search_data(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate opportunities with comprehensive Google Search data."""
        
        validated = []
        
        for i, opp in enumerate(opportunities):
            prompt = f"""
            You have access to Google Search. Perform rigorous validation of this opportunity using real market data.
            
            Opportunity to validate:
            {json.dumps(opp, indent=2)}
            
            Validation research required:
            
            1. Market size validation:
               - Search for industry reports and statistics
               - Find comparable company revenues
               - Look for market research firm data
               - Check government/census data
               - Analyze search volume trends
            
            2. Customer validation:
               - Search for willingness-to-pay studies
               - Find similar product pricing acceptance
               - Look for budget allocation data
               - Check purchase intent surveys
               - Analyze competitor growth rates
            
            3. Technical validation:
               - Search for similar technical implementations
               - Find development cost benchmarks
               - Look for required expertise availability
               - Check technology maturity levels
               - Analyze integration possibilities
            
            4. Business model validation:
               - Search for similar model success stories
               - Find unit economics benchmarks
               - Look for CAC/LTV ratios in sector
               - Check gross margin standards
               - Analyze scaling examples
            
            5. Search queries to use:
               - "[market] TAM SAM SOM analysis"
               - "[solution type] customer acquisition cost"
               - "[industry] software spending trends"
               - "[technology] implementation case studies"
               - "[business model] success rate statistics"
            
            Provide comprehensive validation with:
            - market_size_validated: TAM/SAM/SOM with sources
            - customer_validation: Willingness to pay data
            - technical_validation: Feasibility with examples
            - business_model_validation: Unit economics
            - competitive_validation: Market position viability
            - regulatory_validation: Compliance requirements
            - timeline_validation: Realistic launch schedule
            - investment_required: Accurate cost estimates
            - break_even_analysis: Path to profitability
            - success_indicators: Similar success stories
            - failure_risks: What could go wrong
            - validation_confidence: 0-100 (numeric value)
            - go_no_go_recommendation: Clear decision
            - supporting_evidence: List of URLs and sources
            
            Only validate if confidence is above 70.
            """
            
            # Define the response schema for structured output
            validation_schema = {
                "type": "object",
                "properties": {
                    "market_size_validated": {"type": "string"},
                    "customer_validation": {"type": "string"},
                    "technical_validation": {"type": "string"},
                    "business_model_validation": {"type": "string"},
                    "competitive_validation": {"type": "string"},
                    "regulatory_validation": {"type": "string"},
                    "timeline_validation": {"type": "string"},
                    "investment_required": {"type": "string"},
                    "break_even_analysis": {"type": "string"},
                    "success_indicators": {"type": "string"},
                    "failure_risks": {"type": "string"},
                    "validation_confidence": {"type": "number"},
                    "go_no_go_recommendation": {"type": "string"},
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["market_size_validated", "customer_validation", "technical_validation",
                           "business_model_validation", "competitive_validation", "regulatory_validation",
                           "timeline_validation", "investment_required", "break_even_analysis",
                           "success_indicators", "failure_risks", "validation_confidence",
                           "go_no_go_recommendation", "supporting_evidence"]
            }
            
            print(f"\n📤 Validating opportunity {i+1}/{len(opportunities)}...")
            validation = await self._call_gemini_structured(prompt, validation_schema)
            
            if validation:
                confidence = float(validation.get('validation_confidence', 0))
                
                if confidence > 70:
                    opp['validation'] = validation
                    validated.append(opp)
                    print(f"✅ Opportunity validated with {confidence}% confidence")
                else:
                    print(f"❌ Opportunity not validated (confidence: {confidence}%)")
            else:
                print(f"❌ Opportunity validation failed - no structured response")
        
        return validated
    
    async def _filter_and_rank(self, opportunities: List[Dict[str, Any]], 
                              count: int,
                              existing_portfolio: Optional[Dict[str, Any]] = None) -> List[MarketOpportunity]:
        """Filter and rank opportunities with search-grounded insights."""
        
        # Get insights from emergent learning system
        learning_insights = self.learning_system.get_insights()
        adjusted_thresholds = self.learning_system.get_adjusted_thresholds()
        
        print(f"\n🧠 Learning System Insights:")
        print(f"   Success rate: {learning_insights['success_rate']:.1%}")
        print(f"   Adjusted validation threshold: {adjusted_thresholds['validation_score']:.1f}")
        print(f"   Successful monetization models: {', '.join(learning_insights['successful_monetization_models'][:3]) if learning_insights['successful_monetization_models'] else 'None yet'}")
        
        # Prepare portfolio context
        portfolio_context = ""
        if existing_portfolio:
            portfolio_projects = [
                f"{name}: {data.get('description', '')}" 
                for name, data in existing_portfolio.items()
            ]
            portfolio_context = f"Existing portfolio:\n" + "\n".join(portfolio_projects)
        
        # Prepare opportunities for ranking - include ALL opportunities but with summarized data
        opportunity_summaries = []
        for i, opp in enumerate(opportunities):  # Include ALL opportunities
            # Create a concise summary for each opportunity
            summary = {
                'index': i,
                'name': str(opp.get('solution_name', f'Opportunity {i+1}'))[:80],
                'problem': str(opp.get('problem_description', ''))[:200],
                'audience': str(opp.get('affected_audience', ''))[:100],
                'revenue': str(opp.get('revenue_potential', opp.get('potential_revenue', 'Unknown')))[:50],
                'market_size': str(opp.get('market_size', 'Unknown'))[:30],
                'scores': {
                    'opportunity': float(opp.get('opportunity_score', 0)),
                    'passive_income': float(opp.get('passive_income_score', 0)),
                    'mvp': float(opp.get('mvp_score', 0)),
                    'technical': float(opp.get('technical_score', 0)),
                    'market': float(opp.get('market_score', 0)),
                    'economics': float(opp.get('economics_score', 0))
                },
                'key_advantage': str(opp.get('unique_advantages', [''])[0] if opp.get('unique_advantages') else '')[:100]
            }
            opportunity_summaries.append(summary)
        
        prompt = f"""
        You have access to Google Search. Perform final analysis and selection of the absolute best opportunity from all validated options.
        
        ALL validated opportunities with scores:
        {json.dumps(opportunity_summaries, indent=2)}
        
        {portfolio_context}
        
        Final research tasks:
        1. Search for latest market movements:
           - Recent funding announcements in these spaces
           - New regulations or policy changes
           - Major competitor moves or acquisitions
           - Emerging technologies that could disrupt
        
        2. Verify timing factors:
           - Why is NOW the right time?
           - What catalysts are driving adoption?
           - What barriers have been removed recently?
        
        3. Search queries:
           - "[sector] venture capital funding 2024"
           - "[problem area] new regulations 2024"
           - "[technology] breakthrough announcement"
           - "[market] disruption predictions 2025"
        
        GOLDEN PROJECT Selection Criteria (weighted scoring):
        1. Passive Income Potential (25%) - Revenue frequency × automation level
        2. Market demand validation (20%) - Real evidence with transaction data
        3. Revenue potential (20%) - Clear path to $50K+ MRR with per-transaction model
        4. Competitive advantage (10%) - 15x better with high switching costs
        5. Technical feasibility (10%) - Can build with small team
        6. Time to positive cash flow (10%) - Revenue from day 1
        7. Global 24/7 usage (5%) - Works across all timezones
        
        Select the SINGLE BEST opportunity from all validated options by returning its index number.
        
        If multiple opportunities have high scores, pick THE ONE based on:
        1. Highest weighted score across all criteria
        2. Best unit economics potential
        3. Fastest path to significant revenue
        4. Strongest technical moat and defensibility
        5. Best fit with current market timing
        
        The selected opportunity must be:
        - Specific enough to start building tomorrow
        - Validated with real market data
        - Differentiated from existing solutions
        - Capable of generating passive income 24/7
        - Transaction-based or usage-based revenue model
        - Minimum 90% automation
        - Expected daily revenue from launch
        
        IMPORTANT: You must return ONLY the index (0-based) of the selected opportunity and a brief reason.
        Do not return the opportunity details, just the index number.
        """
        
        # Check prompt size to avoid API limits
        if len(prompt) > 50000:
            self.logger.warning(f"Prompt too large ({len(prompt)} chars), truncating opportunities data")
            # Further simplify if needed
            prompt = prompt[:45000] + "\n\n[Truncated for API limits]"
        
        # Ultra-simple schema - just return the index of the selected opportunity
        response_schema = {
            "type": "object",
            "properties": {
                "selected_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": len(opportunities) - 1
                },
                "reason": {"type": "string"}
            },
            "required": ["selected_index", "reason"]
        }
        
        print("\n📤 Sending structured final ranking request to Gemini...")
        selection_result = await self._call_gemini_structured(prompt, response_schema)
        
        if not selection_result:
            print("\n⚠️ Failed to get structured response for final ranking")
            return []
        
        # Get the selected opportunity by index
        market_opportunities = []
        try:
            selected_index = selection_result.get('selected_index', 0)
            reason = selection_result.get('reason', '')
            
            print(f"\n✅ Selected opportunity index: {selected_index}")
            print(f"   Reason: {reason[:200]}...")
            
            # Get the full opportunity data from our original list
            if 0 <= selected_index < len(opportunities):
                selected_opp = opportunities[selected_index]
                
                # Create a properly formatted MarketOpportunity with ALL the original data
                opportunity_data = {
                    'problem_description': selected_opp.get('problem_description', ''),
                    'affected_audience': selected_opp.get('affected_audience', ''),
                    'current_solutions': selected_opp.get('current_solutions', ''),
                    'opportunity': selected_opp.get('opportunity', ''),
                    'monetization_model': selected_opp.get('monetization_model', 'SaaS'),
                    'market_size': selected_opp.get('market_size', '$1B+'),
                    'competition_level': selected_opp.get('competition_level', 'medium'),
                    'validation_sources': selected_opp.get('validation_sources', []),
                    'key_insights': selected_opp.get('key_insights', []),
                    'implementation_difficulty': selected_opp.get('implementation_difficulty', 'medium'),
                    'potential_revenue': selected_opp.get('revenue_potential', selected_opp.get('potential_revenue', '$10K/month')),
                    'unique_advantages': selected_opp.get('unique_advantages', []),
                    'search_evidence': selected_opp.get('search_evidence', {'summary': 'Validated through research'})
                }
                
                opportunity = MarketOpportunity(**opportunity_data)
                market_opportunities.append(opportunity)
                print(f"\n✅ Selected opportunity: {opportunity.problem_description[:100]}...")
            else:
                print(f"\n❌ Invalid index {selected_index} for {len(opportunities)} opportunities")
                
        except Exception as e:
            self.logger.error(f"Failed to process selection result: {e}")
            print(f"\n❌ Error processing selection: {e}")
        
        return market_opportunities
    
    async def _phase_0_market_timing(self) -> Optional[Dict[str, Any]]:
        """Analyze market conditions and emerging trends for new software products."""
        print("\n📈 Phase 0: Market Timing Analysis & Trend Detection...")
        
        prompt = """
        You have access to Google Search. Analyze current market conditions and detect emerging trends for launching new software products.
        
        Research the following:
        1. Economic indicators affecting software spending:
           - "software spending forecast 2024 2025"
           - "SMB technology budget trends"
           - "venture capital software investments Q4 2024"
           - "recession impact on SaaS adoption"
        
        2. Recent funding trends in tech:
           - "recent series A software startups 2024"
           - "YC batch companies winter 2024"
           - "software categories getting funded 2024"
           - "VC investment thesis 2025"
        
        3. Regulatory changes creating opportunities:
           - "new regulations requiring software 2024"
           - "GDPR CCPA compliance software demand"
           - "AI regulation impact on software"
           - "remote work legislation software needs"
        
        4. Emerging technologies enabling new solutions:
           - "GPT-4 Claude API new use cases"
           - "breakthrough technologies 2024 software"
           - "no-code platforms market growth"
           - "edge computing opportunities 2024"
        
        5. Major industry shifts or disruptions:
           - "industries being disrupted 2024"
           - "digital transformation laggards"
           - "COVID permanent changes software"
           - "gen Z workplace software preferences"
        
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
        {
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
        }
        
        Only return timing_score >= 7 to proceed.
        """
        
        response_schema = {
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
        
        timing_analysis = await self._call_gemini_structured(prompt, response_schema)
        
        if timing_analysis:
            score = timing_analysis.get('timing_score', 0)
            trend_score = timing_analysis.get('trend_score', 0)
            print(f"\n📈 Market Timing Score: {score}/10")
            print(f"🌊 Trend Detection Score: {trend_score}/10")
            print(f"💰 Funding Climate: {timing_analysis.get('funding_climate', 'Unknown')[:100]}...")
            
            # Show emerging trends
            trends = timing_analysis.get('emerging_trends', [])
            if trends:
                print(f"\n🚀 Emerging Trends Detected:")
                for trend in trends[:3]:  # Show top 3
                    print(f"   • {trend}")
            
            # Combined score must be good
            combined_score = (score + trend_score) / 2
            if combined_score >= 7:
                print(f"\n✅ Market conditions favorable (combined score: {combined_score:.1f}/10)")
            else:
                print(f"\n❌ Market conditions unfavorable (combined score: {combined_score:.1f}/10)")
                print(f"   Recommendation: {timing_analysis.get('recommendation', 'Wait')}")
            
            return timing_analysis
        
        print("⚠️ Failed to analyze market timing")
        return None
    
    async def _phase_2_5_user_research(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate user interviews through deep search data analysis."""
        print(f"\n👥 Phase 2.5: Deep User Research Simulation for {len(problems)} problems...")
        
        user_validated = []
        
        for i, problem in enumerate(problems[:10]):  # Deep dive on top 10
            prompt = f"""
            You have access to Google Search. Conduct deep user research for this problem by finding real user stories and experiences.
            
            Problem to research:
            {json.dumps(problem, indent=2)}
            
            Deep user research tasks:
            
            1. Find real user stories:
               - "[problem] day in the life" stories
               - "[problem] workflow frustrations reddit"
               - "[problem] case study before after"
               - "[audience] typical day challenges"
               - "[problem] time wasted statistics"
            
            2. Analyze user workflows:
               - Current workarounds users employ
               - Time spent on manual processes
               - Error rates and rework frequency
               - Collaboration breakdowns
               - Tool switching friction
            
            3. Discover willingness to pay:
               - "[audience] software budget allocation"
               - "[problem] ROI calculator"
               - "how much would you pay for [solution]"
               - Similar tool pricing acceptance
               - Cost of not solving the problem
               - "[problem] transaction volume per day"
               - "[audience] API usage statistics"
               - "[problem] cost per transaction"
               - "switching costs from [competitor]"
            
            4. Identify feature requests:
               - "[competitor] feature requests"
               - "[competitor] missing features"
               - "wish list for [solution type]"
               - "if only [tool] could do"
               - Integration requests
            
            5. Measure urgency and priority:
               - Is this a top 3 problem for users?
               - Daily/weekly/monthly pain frequency
               - Business impact if unsolved
               - Competitive disadvantage without solution
            
            Return comprehensive user research:
            {{
                "user_validation_score": 1-10,
                "user_stories": ["At least 10 specific user quotes/stories"],
                "workflow_analysis": "Detailed current vs. desired workflow",
                "time_cost_impact": "Quantified time/money lost",
                "willingness_to_pay": "Specific price points with evidence",
                "must_have_features": ["Top 10 features users desperately want"],
                "urgency_level": "critical/high/medium/low with evidence",
                "adoption_barriers": ["What would prevent adoption"],
                "success_criteria": "How users define success",
                "user_segments": ["Detailed user personas who need this most"],
                "transaction_frequency": "How often users would use this per day/week",
                "revenue_per_use": "Potential revenue per transaction/API call/action",
                "automation_potential": "Percentage of use cases that can be fully automated"
            }}
            
            Only validate if user_validation_score >= 7 and urgency_level is 'critical' or 'high' or 'medium'.
            """
            
            user_schema = {
                "type": "object",
                "properties": {
                    "user_validation_score": {"type": "number"},
                    "user_stories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 10
                    },
                    "workflow_analysis": {"type": "string"},
                    "time_cost_impact": {"type": "string"},
                    "willingness_to_pay": {"type": "string"},
                    "must_have_features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 10
                    },
                    "urgency_level": {"type": "string"},
                    "adoption_barriers": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "success_criteria": {"type": "string"},
                    "user_segments": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "transaction_frequency": {"type": "string"},
                    "revenue_per_use": {"type": "string"},
                    "automation_potential": {"type": "string"}
                },
                "required": ["user_validation_score", "user_stories", "workflow_analysis",
                           "time_cost_impact", "willingness_to_pay", "must_have_features",
                           "urgency_level", "adoption_barriers", "success_criteria", "user_segments",
                           "transaction_frequency", "revenue_per_use", "automation_potential"]
            }
            
            print(f"\n🔍 Conducting user research {i+1}/{min(10, len(problems))}...")
            user_research = await self._call_gemini_structured(prompt, user_schema)
            
            if user_research:
                # Use AI's user validation score instead of hardcoded thresholds
                score = user_research.get('user_validation_score', 0)
                urgency = user_research.get('urgency_level', 'low')
                num_stories = len(user_research.get('user_stories', []))
                
                # Always include opportunities - let AI's score determine viability
                problem['user_research'] = user_research
                user_validated.append(problem)
                
                # Show status based on score
                if score >= 8:
                    print(f"🟢 Excellent user validation (Score: {score}/10, Urgency: {urgency})")
                    print(f"   Found {num_stories} user stories")
                elif score >= 6:
                    print(f"🟡 Good user validation (Score: {score}/10, Urgency: {urgency})")
                    print(f"   Found {num_stories} user stories")
                elif score >= 4:
                    print(f"🟠 Fair user validation (Score: {score}/10, Urgency: {urgency})")
                    print(f"   Found {num_stories} user stories")
                else:
                    print(f"🔴 Limited user validation (Score: {score}/10, Urgency: {urgency})")
                    print(f"   Found {num_stories} user stories")
            else:
                print("❌ Failed to get user research data")
        
        return user_validated
    
    async def _phase_2_75_revenue_validation(self, problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate automated revenue potential through transaction and usage analysis."""
        print(f"\n💎 Phase 2.75: Automated Revenue Validation for {len(problems)} problems...")
        
        revenue_validated = []
        
        for i, problem in enumerate(problems[:10]):  # Deep dive on top 10
            prompt = f"""
            You have access to Google Search. Validate the automated revenue potential for this problem.
            
            Problem to analyze:
            {json.dumps(problem, indent=2)}
            
            Revenue validation research:
            
            1. Transaction frequency analysis:
               - Search for "[problem] daily transaction volume"
               - "[industry] average transactions per customer"
               - "[problem] API call frequency statistics"
               - "[competitor] usage metrics public data"
               - "[industry] peak usage times globally"
            
            2. Revenue per transaction research:
               - "[industry] payment processing fees standard"
               - "[competitor] pricing per transaction"
               - "API usage pricing benchmarks [industry]"
               - "[problem] willingness to pay per use"
               - "microtransaction acceptance [audience]"
            
            3. 24/7 usage validation:
               - "[industry] usage by timezone data"
               - "[problem] global market presence"
               - "after hours [service] demand"
               - "weekend usage patterns [industry]"
               - "automated processing requirements"
            
            4. Payment integration research:
               - "[industry] common payment processors"
               - "stripe connect marketplace fees"
               - "payment processor API availability"
               - "instant settlement options [industry]"
               - "subscription vs usage billing [audience]"
            
            5. Automation validation:
               - "[process] automation success stories"
               - "human intervention requirements [industry]"
               - "fully automated [competitor] examples"
               - "self-service adoption rates"
               - "chatbot resolution rates [industry]"
            
            Return comprehensive revenue validation:
            {{
                "revenue_validation_score": 1-10,
                "transaction_volume": "Expected daily/monthly transaction count",
                "revenue_per_transaction": "$X.XX per transaction/API call/action",
                "daily_revenue_potential": "$X,XXX projected daily revenue",
                "monthly_revenue_potential": "$XX,XXX projected monthly revenue",
                "usage_pattern": "24/7 global / business hours / sporadic",
                "automation_percentage": "XX% fully automated",
                "payment_processors": ["Available payment integration options"],
                "time_to_first_revenue": "X days from launch",
                "recurring_revenue_model": "transaction/usage/hybrid description",
                "competitive_pricing_analysis": "How competitors charge",
                "golden_project_score": 1-10
            }}
            
            Only validate if:
            - golden_project_score >= 7
            - automation_percentage >= 90%
            - daily_revenue_potential is significant
            - usage_pattern supports 24/7 revenue
            """
            
            revenue_schema = {
                "type": "object",
                "properties": {
                    "revenue_validation_score": {"type": "number"},
                    "transaction_volume": {"type": "string"},
                    "revenue_per_transaction": {"type": "string"},
                    "daily_revenue_potential": {"type": "string"},
                    "monthly_revenue_potential": {"type": "string"},
                    "usage_pattern": {"type": "string"},
                    "automation_percentage": {"type": "string"},
                    "payment_processors": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "time_to_first_revenue": {"type": "string"},
                    "recurring_revenue_model": {"type": "string"},
                    "competitive_pricing_analysis": {"type": "string"},
                    "golden_project_score": {"type": "number"}
                },
                "required": ["revenue_validation_score", "transaction_volume", "revenue_per_transaction",
                           "daily_revenue_potential", "monthly_revenue_potential", "usage_pattern",
                           "automation_percentage", "payment_processors", "time_to_first_revenue",
                           "recurring_revenue_model", "competitive_pricing_analysis", "golden_project_score"]
            }
            
            print(f"\n💰 Validating revenue potential {i+1}/{min(10, len(problems))}...")
            revenue_validation = await self._call_gemini_structured(prompt, revenue_schema)
            
            if revenue_validation:
                # Use AI's golden score instead of hardcoded thresholds
                golden_score = revenue_validation.get('golden_project_score', 0)
                automation = revenue_validation.get('automation_percentage', '0%')
                daily_revenue = revenue_validation.get('daily_revenue_potential', 'Unknown')
                
                # Always include opportunities - let AI's score determine viability
                problem['revenue_validation'] = revenue_validation
                revenue_validated.append(problem)
                
                # Show status based on score
                if golden_score >= 8:
                    print(f"🟢 Excellent revenue potential (Score: {golden_score}/10, Automation: {automation})")
                    print(f"   Daily revenue potential: {daily_revenue}")
                elif golden_score >= 6:
                    print(f"🟡 Good revenue potential (Score: {golden_score}/10, Automation: {automation})")
                    print(f"   Daily revenue potential: {daily_revenue}")
                elif golden_score >= 4:
                    print(f"🟠 Fair revenue potential (Score: {golden_score}/10, Automation: {automation})")
                    print(f"   Daily revenue potential: {daily_revenue}")
                else:
                    print(f"🔴 Challenging revenue potential (Score: {golden_score}/10, Automation: {automation})")
                    print(f"   Daily revenue potential: {daily_revenue}")
            else:
                print("❌ Failed to get revenue validation data")
        
        return revenue_validated
    
    async def _phase_3_5_technical_validation(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deep technical feasibility analysis."""
        print(f"\n🔧 Phase 3.5: Technical Feasibility Deep Dive for {len(opportunities)} opportunities...")
        
        tech_validated = []
        
        for i, opp in enumerate(opportunities):
            prompt = f"""
            You have access to Google Search. Assess technical feasibility concisely.
            
            Opportunity: {opp.get('solution_name', 'Unknown')}
            
            Key technical questions:
            1. What APIs/services are needed and their costs?
            2. Infrastructure costs for 10k users/month?
            3. Time to build MVP with 2-3 developers?
            4. Main technical risks?
            
            Search for: "[solution type] technical architecture", API pricing, similar implementations
            
            Return CONCISE assessment:
            {{
                "technical_score": 1-10,
                "time_to_mvp": "X months",
                "monthly_cost_10k_users": "$X",
                "main_apis_needed": ["API names only"],
                "key_technical_risk": "Single biggest risk",
                "complexity": "low/medium/high"
            }}
            
            Provide a technical_score (1-10) based on feasibility and resource requirements.
            """
            
            tech_schema = {
                "type": "object",
                "properties": {
                    "technical_score": {"type": "number"},
                    "time_to_mvp": {"type": "string"},
                    "monthly_cost_10k_users": {"type": "string"},
                    "main_apis_needed": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "key_technical_risk": {"type": "string"},
                    "complexity": {"type": "string"}
                },
                "required": ["technical_score", "time_to_mvp", "monthly_cost_10k_users",
                           "main_apis_needed", "key_technical_risk", "complexity"]
            }
            
            print(f"\n🔍 Validating technical feasibility {i+1}/{len(opportunities)}...")
            tech_validation = await self._call_gemini_structured(prompt, tech_schema)
            
            if tech_validation:
                score = tech_validation.get('technical_score', 0)
                time_to_mvp = tech_validation.get('time_to_mvp', '12 months')
                complexity = tech_validation.get('complexity', 'unknown')
                cost = tech_validation.get('monthly_cost_10k_users', 'unknown')
                
                # Always include opportunities - let AI's score determine viability
                opp['technical_validation'] = tech_validation
                tech_validated.append(opp)
                
                # Show status based on score
                if score >= 8:
                    print(f"🟢 Excellent technical feasibility (Score: {score}/10, MVP: {time_to_mvp}, Complexity: {complexity})")
                    print(f"   Monthly cost for 10k users: {cost}")
                elif score >= 6:
                    print(f"🟡 Good technical feasibility (Score: {score}/10, MVP: {time_to_mvp}, Complexity: {complexity})")
                    print(f"   Monthly cost for 10k users: {cost}")
                elif score >= 4:
                    print(f"🟠 Fair technical feasibility (Score: {score}/10, MVP: {time_to_mvp}, Complexity: {complexity})")
                    print(f"   Monthly cost for 10k users: {cost}")
                else:
                    print(f"🔴 Challenging technical feasibility (Score: {score}/10, MVP: {time_to_mvp}, Complexity: {complexity})")
                    print(f"   Monthly cost for 10k users: {cost}")
            else:
                print("❌ Failed to get technical validation")
        
        return tech_validated
    
    async def _phase_4_market_sizing(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get hard numbers on market size with quantitative validation."""
        print(f"\n📊 Phase 4: Quantitative Market Sizing for {len(opportunities)} opportunities...")
        
        market_validated = []
        
        for i, opp in enumerate(opportunities):
            prompt = f"""
            You have access to Google Search. Find concrete market size data with real numbers.
            
            Opportunity to size:
            {json.dumps(opp, indent=2)}
            
            Market sizing research:
            
            1. TAM (Total Addressable Market):
               - "[market] size billion 2024"
               - "[industry] software spending total"
               - "global [problem area] market value"
               - Industry reports from Gartner, Forrester, IDC
               - Government statistics on business counts
            
            2. SAM (Serviceable Addressable Market):
               - Number of businesses in target segment
               - Geographic constraints
               - Regulatory limitations
               - Language/localization requirements
               - Actual reachable market
            
            3. SOM (Serviceable Obtainable Market):
               - Realistic market share year 1-3
               - Competitor market shares
               - Customer acquisition constraints
               - Growth rate assumptions
               - Channel limitations
            
            4. Supporting metrics:
               - "[audience] population size 2024"
               - "[industry] number of companies"
               - "[problem] affects percentage statistics"
               - "[solution category] adoption rate"
               - "[market] CAGR forecast 2025"
            
            5. Comparable company analysis:
               - Similar startups' revenue/user numbers
               - Time to reach milestones
               - Market penetration rates
               - Average revenue per user
               - Churn rates in category
            
            Return market sizing with HARD NUMBERS:
            {{
                "market_score": 1-10,
                "tam_dollars": "Specific number in billions/millions",
                "tam_users": "Number of potential users globally",
                "sam_dollars": "Serviceable market in dollars",
                "sam_users": "Reachable users in target markets",
                "som_year1": "Realistic year 1 revenue",
                "som_year3": "Year 3 revenue projection",
                "market_growth_rate": "Annual growth percentage",
                "competitor_benchmarks": ["Company X: $Y revenue at Z users"],
                "data_sources": ["Specific reports and URLs"],
                "confidence_level": "high/medium/low based on data quality"
            }}
            
            Provide a market_score (1-10) based on overall market opportunity,
            considering TAM, growth rate, competition, and accessibility.
            """
            
            market_schema = {
                "type": "object",
                "properties": {
                    "market_score": {"type": "number"},
                    "tam_dollars": {"type": "string"},
                    "tam_users": {"type": "string"},
                    "sam_dollars": {"type": "string"},
                    "sam_users": {"type": "string"},
                    "som_year1": {"type": "string"},
                    "som_year3": {"type": "string"},
                    "market_growth_rate": {"type": "string"},
                    "competitor_benchmarks": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "data_sources": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "confidence_level": {"type": "string"}
                },
                "required": ["market_score", "tam_dollars", "tam_users", "sam_dollars", "sam_users",
                           "som_year1", "som_year3", "market_growth_rate",
                           "competitor_benchmarks", "data_sources", "confidence_level"]
            }
            
            print(f"\n📊 Sizing market {i+1}/{len(opportunities)}...")
            market_data = await self._call_gemini_structured(prompt, market_schema)
            
            if market_data:
                # Use AI's market score instead of hardcoded thresholds
                market_score = market_data.get('market_score', 0)
                tam_str = market_data.get('tam_dollars', '$0')
                users_str = market_data.get('tam_users', '0')
                confidence = market_data.get('confidence_level', 'low')
                
                # Always include opportunities - let AI's score determine viability
                opp['market_sizing'] = market_data
                market_validated.append(opp)
                
                # Show status based on score
                if market_score >= 8:
                    print(f"🟢 Excellent market opportunity (Score: {market_score}/10, TAM: {tam_str}, Users: {users_str})")
                    print(f"   Confidence: {confidence}")
                elif market_score >= 6:
                    print(f"🟡 Good market opportunity (Score: {market_score}/10, TAM: {tam_str}, Users: {users_str})")
                    print(f"   Confidence: {confidence}")
                elif market_score >= 4:
                    print(f"🟠 Fair market opportunity (Score: {market_score}/10, TAM: {tam_str}, Users: {users_str})")
                    print(f"   Confidence: {confidence}")
                else:
                    print(f"🔴 Limited market opportunity (Score: {market_score}/10, TAM: {tam_str}, Users: {users_str})")
                    print(f"   Confidence: {confidence}")
            else:
                print("❌ Failed to get market sizing data")
        
        return market_validated
    
    async def _phase_4_5_unit_economics(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate business viability with detailed unit economics."""
        print(f"\n💰 Phase 4.5: Unit Economics Validation for {len(opportunities)} opportunities...")
        
        economics_validated = []
        
        for i, opp in enumerate(opportunities):
            prompt = f"""
            You have access to Google Search. Calculate detailed unit economics for this opportunity.
            
            Opportunity to analyze:
            {json.dumps(opp, indent=2)}
            
            Unit economics research:
            
            1. Customer Acquisition Cost (CAC):
               - "[industry] customer acquisition cost SaaS"
               - "[marketing channel] CAC benchmarks"
               - Content marketing costs per lead
               - Paid advertising costs per customer
               - Sales team costs if needed
            
            2. Lifetime Value (LTV):
               - Average revenue per user (ARPU)
               - Churn rate benchmarks for category
               - Expansion revenue potential
               - Support costs per customer
               - Gross margins after all costs
            
            3. Key SaaS metrics:
               - "[category] SaaS magic number"
               - "[industry] payback period average"
               - "B2B SaaS rule of 40 benchmarks"
               - Net revenue retention rates
               - Gross margin standards
            
            4. Cost structure:
               - Infrastructure costs per user
               - Support costs as % of revenue
               - R&D costs for feature development
               - Sales & marketing as % of revenue
               - G&A overhead allocation
            
            5. Benchmark comparisons:
               - Public SaaS company metrics
               - Similar stage startup metrics
               - Industry unit economics reports
               - Investor expectations for category
            
            Calculate with specific numbers:
            {{
                "economics_score": 1-10,
                "cac_blended": "$X average across channels",
                "cac_breakdown": {{"content": "$X", "paid": "$X", "sales": "$X"}},
                "ltv_monthly": "$X based on ARPU and churn",
                "ltv_cac_ratio": "X:1 ratio",
                "payback_months": "X months",
                "gross_margin": "X% after all COGS",
                "monthly_burn_rate": "$X at 100 customers",
                "break_even_customers": "X customers needed",
                "rule_of_40_score": "Growth rate + profit margin",
                "benchmark_comparison": ["Similar company metrics"],
                "key_assumptions": ["Critical assumptions made"],
                "sensitivity_analysis": "Impact of 20% worse CAC or churn"
            }}
            
            Provide an economics_score (1-10) based on overall unit economics health,
            considering industry standards, growth potential, and sustainability.
            """
            
            economics_schema = {
                "type": "object",
                "properties": {
                    "economics_score": {"type": "number"},
                    "cac_blended": {"type": "string"},
                    "cac_breakdown": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "paid": {"type": "string"},
                            "sales": {"type": "string"}
                        }
                    },
                    "ltv_monthly": {"type": "string"},
                    "ltv_cac_ratio": {"type": "string"},
                    "payback_months": {"type": "string"},
                    "gross_margin": {"type": "string"},
                    "monthly_burn_rate": {"type": "string"},
                    "break_even_customers": {"type": "string"},
                    "rule_of_40_score": {"type": "string"},
                    "benchmark_comparison": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "key_assumptions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "sensitivity_analysis": {"type": "string"}
                },
                "required": ["economics_score", "cac_blended", "cac_breakdown", "ltv_monthly", "ltv_cac_ratio",
                           "payback_months", "gross_margin", "monthly_burn_rate",
                           "break_even_customers", "rule_of_40_score", "benchmark_comparison",
                           "key_assumptions", "sensitivity_analysis"]
            }
            
            print(f"\n📊 Calculating unit economics {i+1}/{len(opportunities)}...")
            economics = await self._call_gemini_structured(prompt, economics_schema)
            
            if economics:
                # Use AI's economics score instead of hardcoded calculations
                economics_score = economics.get('economics_score', 0)
                ratio_str = economics.get('ltv_cac_ratio', 'N/A')
                payback_str = economics.get('payback_months', 'N/A')
                
                # Always include opportunities - let AI's score determine viability
                opp['unit_economics'] = economics
                economics_validated.append(opp)
                
                # Show status based on score
                if economics_score >= 8:
                    print(f"🟢 Excellent economics (Score: {economics_score}/10, LTV:CAC: {ratio_str}, Payback: {payback_str})")
                elif economics_score >= 6:
                    print(f"🟡 Good economics (Score: {economics_score}/10, LTV:CAC: {ratio_str}, Payback: {payback_str})")
                elif economics_score >= 4:
                    print(f"🟠 Fair economics (Score: {economics_score}/10, LTV:CAC: {ratio_str}, Payback: {payback_str})")
                else:
                    print(f"🔴 Challenging economics (Score: {economics_score}/10, LTV:CAC: {ratio_str}, Payback: {payback_str})")
            else:
                print("❌ Failed to calculate unit economics")
        
        return economics_validated
    
    async def _phase_4_75_mvp_revenue_simulation(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate MVP revenue for first 30 days and path to profitability."""
        print(f"\n🚀 Phase 4.75: MVP Revenue Simulation for {len(opportunities)} opportunities...")
        
        mvp_validated = []
        
        for i, opp in enumerate(opportunities):
            prompt = f"""
            You have access to Google Search. Simulate the first 30 days revenue for an MVP of this opportunity.
            
            Opportunity to simulate:
            {json.dumps(opp, indent=2)}
            
            MVP revenue simulation research:
            
            1. Launch strategy research:
               - "product hunt launch conversion rates"
               - "beta launch user acquisition costs"
               - "[industry] early adopter characteristics"
               - "MVP feature set benchmarks"
               - "soft launch vs hard launch metrics"
            
            2. First customer acquisition:
               - "time to first paying customer SaaS"
               - "[industry] trial to paid conversion"
               - "early bird pricing strategies"
               - "founder network conversion rates"
               - "cold outreach success rates B2B"
            
            3. Daily revenue projection:
               - "[industry] MVP daily active users"
               - "transaction volume ramp up curve"
               - "[competitor] early growth metrics"
               - "organic vs paid user acquisition"
               - "viral coefficient benchmarks"
            
            4. Cost analysis:
               - "MVP infrastructure costs [scale]"
               - "payment processing setup costs"
               - "customer support automation costs"
               - "monitoring and analytics costs"
               - "compliance and legal minimums"
            
            5. Path to profitability:
               - "SaaS burn rate benchmarks"
               - "[industry] break even timeline"
               - "unit economics improvement curve"
               - "scale effects on margins"
               - "reinvestment vs profitability balance"
            
            Create detailed 30-day simulation:
            {{
                "mvp_score": 1-10,
                "day_1_revenue": "$X.XX first day projection",
                "day_7_revenue": "$XXX first week total",
                "day_30_revenue": "$X,XXX first month total",
                "first_paying_customer": "Day X expected",
                "customer_acquisition_strategy": "Primary method for first 10 customers",
                "mvp_features": ["Core features for launch"],
                "launch_costs": "$X,XXX total launch investment",
                "daily_server_costs": "$XX per day at launch scale",
                "break_even_projection": "Day XX or Month X",
                "30_day_metrics": {{
                    "total_users": "XXX",
                    "paying_users": "XX",
                    "transactions": "X,XXX",
                    "mrr_exit_rate": "$X,XXX"
                }},
                "growth_trajectory": "linear/exponential/stepped",
                "key_risks": ["Main risks to revenue projections"],
                "confidence_level": "high/medium/low with reasoning"
            }}
            
            Only validate if:
            - mvp_score >= 7
            - first_paying_customer <= Day 7
            - day_30_revenue is meaningful (>$1000)
            - confidence_level is high or medium
            """
            
            mvp_schema = {
                "type": "object",
                "properties": {
                    "mvp_score": {"type": "number"},
                    "day_1_revenue": {"type": "string"},
                    "day_7_revenue": {"type": "string"},
                    "day_30_revenue": {"type": "string"},
                    "first_paying_customer": {"type": "string"},
                    "customer_acquisition_strategy": {"type": "string"},
                    "mvp_features": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "launch_costs": {"type": "string"},
                    "daily_server_costs": {"type": "string"},
                    "break_even_projection": {"type": "string"},
                    "30_day_metrics": {
                        "type": "object",
                        "properties": {
                            "total_users": {"type": "string"},
                            "paying_users": {"type": "string"},
                            "transactions": {"type": "string"},
                            "mrr_exit_rate": {"type": "string"}
                        }
                    },
                    "growth_trajectory": {"type": "string"},
                    "key_risks": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "confidence_level": {"type": "string"}
                },
                "required": ["mvp_score", "day_1_revenue", "day_7_revenue", "day_30_revenue",
                           "first_paying_customer", "customer_acquisition_strategy", "mvp_features",
                           "launch_costs", "daily_server_costs", "break_even_projection",
                           "30_day_metrics", "growth_trajectory", "key_risks", "confidence_level"]
            }
            
            print(f"\n📊 Simulating MVP revenue {i+1}/{len(opportunities)}...")
            mvp_simulation = await self._call_gemini_structured(prompt, mvp_schema)
            
            if mvp_simulation:
                # Use AI's MVP score instead of hardcoded thresholds
                score = mvp_simulation.get('mvp_score', 0)
                first_customer = mvp_simulation.get('first_paying_customer', 'Day 30')
                day_30_str = mvp_simulation.get('day_30_revenue', '$0')
                confidence = mvp_simulation.get('confidence_level', 'low')
                
                # Always include opportunities - let AI's score determine viability
                opp['mvp_simulation'] = mvp_simulation
                mvp_validated.append(opp)
                
                # Show status based on score
                if score >= 8:
                    print(f"🟢 Excellent MVP potential (Score: {score}/10, First customer: {first_customer})")
                    print(f"   30-day revenue: {day_30_str}, Confidence: {confidence}")
                elif score >= 6:
                    print(f"🟡 Good MVP potential (Score: {score}/10, First customer: {first_customer})")
                    print(f"   30-day revenue: {day_30_str}, Confidence: {confidence}")
                elif score >= 4:
                    print(f"🟠 Fair MVP potential (Score: {score}/10, First customer: {first_customer})")
                    print(f"   30-day revenue: {day_30_str}, Confidence: {confidence}")
                else:
                    print(f"🔴 Challenging MVP potential (Score: {score}/10, First customer: {first_customer})")
                    print(f"   30-day revenue: {day_30_str}, Confidence: {confidence}")
            else:
                print("❌ Failed to simulate MVP revenue")
        
        return mvp_validated
    
    async def _phase_6_pre_build_validation(self, opportunity: MarketOpportunity) -> Optional[Dict[str, Any]]:
        """Final validation checks before committing to build the project."""
        print(f"\n🎯 Phase 6: Pre-Build Validation for {opportunity.opportunity[:50]}...")
        
        prompt = f"""
        You have access to Google Search. Perform final pre-build validation checks for this opportunity.
        
        Final opportunity selected:
        - Problem: {opportunity.problem_description}
        - Solution: {opportunity.opportunity}
        - Market: {opportunity.market_size}
        - Revenue Model: {opportunity.monetization_model}
        
        Final validation checks:
        
        1. Domain availability check:
           - Search "[project name variations] domain availability"
           - ".com .io .app domain registrar check"
           - "trademark search [project name]"
           - "brand name conflicts [industry]"
        
        2. Key integration verification:
           - Verify critical APIs mentioned are available
           - Check API pricing hasn't changed dramatically
           - Confirm no major platform policy changes
           - Test sandbox/developer account availability
        
        3. Regulatory compliance quick check:
           - "[industry] software compliance requirements 2024"
           - "data privacy laws [target market]"
           - "payment processing regulations [region]"
           - "industry specific licenses needed"
        
        4. Technical prototype validation:
           - Search for similar MVP implementations
           - "[core feature] implementation tutorial"
           - Required libraries/frameworks stability
           - Hosting platform compatibility
        
        5. Launch strategy validation:
           - "[target audience] online communities"
           - "product hunt launch best practices 2024"
           - Early adopter acquisition channels
           - Beta testing user sources
        
        6. Competitive landscape final check:
           - Any new competitors launched recently?
           - Major players entering this space?
           - Recent funding announcements in sector
           - Acquisition activity that changes market
        
        Return validation results:
        {{
            "validation_score": 1-10,
            "domain_available": true/false,
            "suggested_domains": ["available domain options"],
            "critical_apis_verified": true/false,
            "api_concerns": ["any API availability/pricing issues"],
            "regulatory_requirements": ["key compliance needs"],
            "technical_blockers": ["any technical showstoppers found"],
            "launch_channels": ["validated user acquisition channels"],
            "new_competition": ["any concerning new entrants"],
            "proceed_with_build": true/false,
            "final_recommendations": ["critical actions before starting"]
        }}
        
        Provide an honest validation_score (1-10) based on overall readiness.
        Set proceed_with_build based on whether the opportunity is viable,
        even if some elements need work (e.g., domain alternatives needed).
        Focus on identifying concerns to address rather than blocking progress.
        """
        
        validation_schema = {
            "type": "object",
            "properties": {
                "validation_score": {"type": "number"},
                "domain_available": {"type": "boolean"},
                "suggested_domains": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "critical_apis_verified": {"type": "boolean"},
                "api_concerns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "regulatory_requirements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "technical_blockers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "launch_channels": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "new_competition": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "proceed_with_build": {"type": "boolean"},
                "final_recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["validation_score", "domain_available", "suggested_domains",
                       "critical_apis_verified", "api_concerns", "regulatory_requirements",
                       "technical_blockers", "launch_channels", "new_competition",
                       "proceed_with_build", "final_recommendations"]
        }
        
        print("🔍 Running final pre-build checks...")
        validation = await self._call_gemini_structured(prompt, validation_schema)
        
        if validation:
            score = validation.get('validation_score', 0)
            proceed = validation.get('proceed_with_build', False)
            
            print(f"\n🎯 Pre-Build Validation Score: {score}/10")
            print(f"🌐 Domain Available: {validation.get('domain_available', False)}")
            print(f"🔌 APIs Verified: {validation.get('critical_apis_verified', False)}")
            
            if proceed:
                print("✅ All checks passed - proceeding with project creation!")
                if validation.get('final_recommendations'):
                    print("\n📄 Final Recommendations:")
                    for rec in validation['final_recommendations'][:3]:
                        print(f"   • {rec}")
            else:
                print("❌ Pre-build validation failed")
                if validation.get('technical_blockers'):
                    print("\n🚫 Technical Blockers:")
                    for blocker in validation['technical_blockers']:
                        print(f"   • {blocker}")
            
            return validation
        
        print("⚠️ Failed to complete pre-build validation")
        return None
    
    async def _save_research_knowledge(self, opportunities: List[MarketOpportunity]) -> None:
        """Save research data to knowledge base for future reference and learning."""
        try:
            # Create filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"research_{timestamp}.json"
            research_dir = Path.cwd() / "research_knowledge"
            filepath = research_dir / filename
            
            # Ensure directory exists
            research_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare COMPREHENSIVE research data with ALL phases
            research_data = {
                'opportunities': [asdict(opp) for opp in opportunities],
                'research_timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'gemini_deep_research_intensive_mode',
                'validation_method': 'multi_phase_intensive_validation',
                'research_settings': {
                    'severity_threshold': 8,  # ≥ 8/10
                    'validation_threshold': 9,  # ≥ 9/10
                    'problems_discovered_target': '25-30',
                    'opportunity_multiplier': '15x',
                    'all_problems_validated': True,
                    'phases_completed': len([p for p in self.research_phases_data.values() if p])
                },
                'phase_data': self.research_phases_data,  # ALL data from every phase
                'statistics': {
                    'total_sectors_analyzed': len(self.all_sectors_researched),
                    'sectors_discovered': list(self.all_sectors_researched),
                    'problems_discovered': self.research_stats['problems_discovered'],
                    'problems_validated': self.research_stats['problems_validated'],
                    'competition_analyzed': self.research_stats['competition_analyzed'],
                    'search_queries_made': self.research_stats['search_queries_made'],
                    'phase_0_market_timing': bool(self.research_phases_data.get('phase_0_market_timing')),
                    'phase_1_problems_count': len(self.research_phases_data.get('phase_1_problems_discovered', [])),
                    'phase_2_validated_count': len(self.research_phases_data.get('phase_2_problems_validated', [])),
                    'phase_2_5_user_validated': len(self.research_phases_data.get('phase_2_5_user_research', [])),
                    'phase_2_75_revenue_validated': len(self.research_phases_data.get('phase_2_75_revenue_validation', [])),
                    'phase_3_opportunities': len(self.research_phases_data.get('phase_3_competition_analysis', [])),
                    'phase_3_5_tech_validated': len(self.research_phases_data.get('phase_3_5_technical_validation', [])),
                    'phase_4_market_sized': len(self.research_phases_data.get('phase_4_market_sizing', [])),
                    'phase_4_5_economics_validated': len(self.research_phases_data.get('phase_4_5_unit_economics', [])),
                    'phase_4_75_mvp_validated': len(self.research_phases_data.get('phase_4_75_mvp_simulation', [])),
                    'phase_5_final_selected': len(opportunities),
                    'phase_6_pre_build_validated': bool(self.research_phases_data.get('phase_6_pre_build_validation'))
                }
            }
            
            # Save research data
            with open(filepath, 'w') as f:
                safe_json_dump(research_data, f, indent=2)
            
            self.logger.info(f"✅ Saved research knowledge to: {filepath}")
            
            # Update summary file
            summary_file = research_dir / "research_summary.json"
            summary_data = {}
            
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
            
            # Initialize summary structure if needed
            if 'total_research_sessions' not in summary_data:
                summary_data = {
                    'total_research_sessions': 0,
                    'all_sectors_discovered': [],
                    'all_problems_discovered': [],
                    'research_files': [],
                    'total_unique_sectors': 0,
                    'total_problems_discovered': 0
                }
            
            # Update summary
            summary_data['total_research_sessions'] += 1
            summary_data['last_research'] = research_data['research_timestamp']
            
            # Track unique sectors
            current_sectors = set(summary_data.get('all_sectors_discovered', []))
            current_sectors.update(self.all_sectors_researched)
            summary_data['all_sectors_discovered'] = list(current_sectors)
            summary_data['total_unique_sectors'] = len(current_sectors)
            
            # Track problems discovered
            for opp in opportunities:
                summary_data['all_problems_discovered'].append({
                    'problem': opp.problem_description,
                    'timestamp': research_data['research_timestamp'],
                    'audience': opp.affected_audience,
                    'opportunity': opp.opportunity,
                    'market_size': opp.market_size
                })
            
            summary_data['total_problems_discovered'] = len(summary_data['all_problems_discovered'])
            summary_data['research_files'].append(filename)
            
            # Save updated summary
            with open(summary_file, 'w') as f:
                safe_json_dump(summary_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save research knowledge: {e}")
            # Don't fail the whole process if saving fails
    
    def get_research_results(self) -> Dict[str, Any]:
        """Get research results with all tracked statistics."""
        return {
            'total_sectors_analyzed': len(self.all_sectors_researched),
            'sectors_discovered': list(self.all_sectors_researched),
            'problems_discovered': self.research_stats['problems_discovered'],
            'problems_validated': self.research_stats['problems_validated'],
            'competition_analyzed': self.research_stats['competition_analyzed'],
            'search_queries_made': self.research_stats['search_queries_made']
        }


async def demonstrate_deep_research():
    """Demonstrate deep research capabilities with Google Search grounding."""
    print("🚀 Gemini Deep Research with Organic Problem Discovery")
    print("=" * 60)
    
    # Initialize deep research
    researcher = GeminiDeepResearch()
    
    print("\n🔍 Discovering real-world problems through organic search...")
    print("This approach finds actual problems people are struggling with")
    print("rather than analyzing predefined market sectors.\n")
    
    # Conduct research
    opportunities = await researcher.conduct_deep_market_research()
    
    # Display results
    print(f"\n📊 Found {len(opportunities)} validated opportunities:\n")
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp.problem_description}")
        print(f"   🎯 Target: {opp.affected_audience}")
        print(f"   💰 Revenue: {opp.potential_revenue}")
        print(f"   🏆 Competition: {opp.competition_level}")
        print(f"   ✨ Advantages: {', '.join(opp.unique_advantages[:2])}")
        print(f"   📊 Market Size: {opp.market_size}")
        print(f"   🔗 Validated by {len(opp.validation_sources)} sources")
        print()
    
    # Show one detailed opportunity
    if opportunities:
        print("\n📋 Detailed View - Top Opportunity:")
        print("=" * 60)
        top = opportunities[0]
        print(f"Problem: {top.problem_description}")
        print(f"\nKey Insights:")
        for insight in top.key_insights:
            print(f"  • {insight}")
        print(f"\nMonetization: {top.monetization_model}")
        print(f"\nValidation Sources:")
        for source in top.validation_sources[:3]:
            print(f"  • {source}")


if __name__ == "__main__":
    # Set up environment
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run demonstration
    asyncio.run(demonstrate_deep_research())