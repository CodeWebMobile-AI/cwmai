#!/usr/bin/env python3
"""
Simple Problem-First Market Research
Focuses on discovering real-world problems without sector bias
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

from scripts.ai_brain import IntelligentAIBrain


@dataclass
class SimpleProblem:
    """Represents a simple real-world problem opportunity."""
    problem: str
    who: str
    impact: str
    current_solutions: str
    opportunity: str
    estimated_users: str
    revenue_potential: str
    validation_score: float


class SimpleProblemResearch:
    """Discovers real-world problems using a simplified approach."""
    
    def __init__(self, ai_brain: Optional[IntelligentAIBrain] = None):
        """Initialize with AI brain."""
        self.logger = logging.getLogger(__name__)
        self.ai_brain = ai_brain or IntelligentAIBrain()
    
    async def find_real_world_problems(self) -> List[SimpleProblem]:
        """Discover problems that everyday people face."""
        
        print("🔍 Searching for real-world problems people actually face...")
        
        # Phase 1: Discover problems
        problems = await self._discover_everyday_problems()
        
        # Phase 2: Quick validation
        if problems:
            validated = await self._validate_problem_demand(problems)
            return validated
        
        return []
    
    async def _discover_everyday_problems(self) -> List[Dict[str, Any]]:
        """Discover problems using focused search queries."""
        
        prompt = """
        You have access to Google Search. Find the top 10 most frustrating problems 
        that regular people (NOT developers or tech professionals) face in their daily lives.
        
        Use these search queries:
        - "biggest daily frustrations" site:reddit.com
        - "I wish there was an app for" -developer -coding -programming
        - "why is it so hard to" everyday life
        - "frustrated with" small business owners
        - "struggling with" parents elderly healthcare
        - "waste too much time on" daily tasks
        - "hate dealing with" life admin
        - "annoying things about" work life balance
        
        Focus on problems that:
        - Affect non-technical people (families, small businesses, elderly, students)
        - Waste significant time or money regularly
        - Have solutions that are too complex, expensive, or don't exist
        - Could be solved with a simple web/mobile app
        - People would gladly pay $10-50/month to solve
        
        AVOID:
        - Developer tools or coding problems
        - Technical infrastructure issues
        - Problems requiring hardware solutions
        - Problems already well-solved by major companies
        
        For each problem, provide:
        {
            "problem": "Clear description of the frustration",
            "who": "Specific group affected (e.g., 'working parents with young children')",
            "impact": "Time/money lost (e.g., '2 hours per week' or '$200/month')",
            "current_solutions": "What exists and why it fails",
            "opportunity": "Simple solution approach using web/mobile app",
            "estimated_users": "Realistic market size (e.g., '~2M working parents in US')",
            "search_evidence": "Key findings from search"
        }
        
        Return exactly 10 problems as a JSON array, sorted by impact.
        Focus on REAL frustrations that REAL people have RIGHT NOW.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
        return self._parse_json_array(response)
    
    async def _validate_problem_demand(self, problems: List[Dict[str, Any]]) -> List[SimpleProblem]:
        """Quickly validate that people would pay for solutions."""
        
        validated = []
        
        prompt = f"""
        For these real-world problems, validate which ones have the strongest demand 
        for a simple SaaS solution. Consider:
        
        1. Would people pay $10-50/month to solve this?
        2. Is the problem frequent enough (daily/weekly)?
        3. Is current solution genuinely inadequate?
        4. Can it be solved with Laravel/React web app?
        
        Problems to validate:
        {json.dumps(problems, indent=2)}
        
        For ALL 10 problems (sorted by demand), provide:
        {{
            "problem": "The specific problem",
            "who": "Target audience",
            "impact": "Quantified impact",
            "current_solutions": "What exists",
            "opportunity": "Our simple solution",
            "estimated_users": "Market size",
            "revenue_potential": "$X-Y MRR potential",
            "validation_score": 0.0-1.0 (based on willingness to pay)
        }}
        
        Return as JSON array of 10 validated problems.
        Prioritize problems where people are ALREADY trying to pay for bad solutions.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini-pro')
        problems_data = self._parse_json_array(response)
        
        # Convert to SimpleProblem objects
        for data in problems_data:
            try:
                # Ensure validation_score is float
                data['validation_score'] = float(data.get('validation_score', 0.5))
                validated.append(SimpleProblem(**data))
            except Exception as e:
                self.logger.error(f"Failed to create SimpleProblem: {e}")
        
        return validated
    
    def _parse_json_array(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse JSON array from AI response."""
        if not response or 'content' not in response:
            return []
        
        content = response['content']
        print(f"\n📝 Raw response preview (first 500 chars):\n{content[:500]}...\n")
        
        try:
            # Remove markdown code blocks
            content = content.replace('```json', '').replace('```', '').strip()
            
            # Try to parse entire content first
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    print(f"✅ Successfully parsed {len(parsed)} items")
                    return parsed
            except json.JSONDecodeError as e:
                print(f"⚠️ Direct parse failed: {e}")
            
            # Try to find JSON array - look for [ to ]
            start_idx = content.find('[')
            if start_idx != -1:
                # Find the matching closing bracket
                bracket_count = 0
                in_string = False
                escape_next = False
                end_idx = -1
                
                for i in range(start_idx, len(content)):
                    char = content[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\' and in_string:
                        escape_next = True
                        continue
                    
                    if char == '"':
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_idx = i + 1
                                break
                
                if end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    print(f"📍 Found JSON array ({end_idx - start_idx} chars)")
                    try:
                        parsed = json.loads(json_str)
                        print(f"✅ Successfully parsed {len(parsed)} items")
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                        # Try to fix common issues
                        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            parsed = json.loads(json_str)
                            print(f"✅ Fixed and parsed {len(parsed)} items")
                            return parsed
                        except:
                            print("❌ Could not fix JSON")
            
            print("❌ No JSON array found in response")
                
        except Exception as e:
            self.logger.error(f"JSON parsing failed: {e}")
            print(f"❌ Parsing error: {e}")
        
        return []


async def test_simple_approach():
    """Test the simplified problem-first approach."""
    print("🧪 Testing Simple Problem-First Research")
    print("=" * 60)
    
    researcher = SimpleProblemResearch()
    
    print("\nThis approach:")
    print("✓ No hardcoded sectors")
    print("✓ Focuses on everyday frustrations")
    print("✓ Excludes developer problems")
    print("✓ Simple 2-phase process\n")
    
    try:
        problems = await researcher.find_real_world_problems()
        
        print(f"\n✅ Found {len(problems)} validated problems:\n")
        
        for i, problem in enumerate(problems, 1):
            print(f"\n{'='*60}")
            print(f"Problem {i}: {problem.problem}")
            print(f"{'='*60}")
            print(f"  👥 Who: {problem.who}")
            print(f"  💸 Impact: {problem.impact}")
            print(f"  ❌ Current Solutions: {problem.current_solutions}")
            print(f"  💡 Opportunity: {problem.opportunity}")
            print(f"  📊 Market Size: {problem.estimated_users}")
            print(f"  💰 Revenue Potential: {problem.revenue_potential}")
            print(f"  ✅ Validation Score: {problem.validation_score:.1%}")
        
        # Analyze results
        print(f"\n\n📈 Analysis:")
        print("-" * 60)
        
        # Check for technical vs non-technical
        tech_keywords = ['developer', 'programmer', 'coding', 'api', 'debug']
        life_keywords = ['family', 'parent', 'elderly', 'small business', 'health', 
                        'home', 'personal', 'daily', 'life']
        
        tech_count = 0
        life_count = 0
        
        for problem in problems:
            full_text = f"{problem.problem} {problem.who} {problem.opportunity}".lower()
            
            if any(keyword in full_text for keyword in tech_keywords):
                tech_count += 1
            if any(keyword in full_text for keyword in life_keywords):
                life_count += 1
        
        print(f"  📊 Real-life focused problems: {life_count}/{len(problems)}")
        print(f"  💻 Technical problems: {tech_count}/{len(problems)}")
        if problems:
            print(f"  💰 Average validation score: {sum(p.validation_score for p in problems)/len(problems):.1%}")
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "approach": "simple_problem_first",
            "total_problems": len(problems),
            "life_focused": life_count,
            "tech_focused": tech_count,
            "problems": [
                {
                    "problem": p.problem,
                    "who": p.who,
                    "impact": p.impact,
                    "opportunity": p.opportunity,
                    "revenue": p.revenue_potential,
                    "validation": p.validation_score
                }
                for p in problems
            ]
        }
        
        with open("simple_approach_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to simple_approach_results.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Testing simplified problem-first approach...")
    print("No sectors, no bias, just real problems.\n")
    asyncio.run(test_simple_approach())