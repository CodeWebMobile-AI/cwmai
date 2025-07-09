#!/usr/bin/env python3
"""Convert markdown files to structured JSON using Gemini with structured output."""

import json
import os
import asyncio
import aiohttp
import ssl
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

class GeminiMarkdownParser:
    """Use Gemini to parse markdown into structured format."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get Gemini API key for direct structured output calls
        self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.gemini_api_key:
            self.logger.warning("GOOGLE_API_KEY not found in environment")
    
    async def _call_gemini_structured(self, prompt: str, response_schema: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make direct Gemini API call with structured output.
        
        Args:
            prompt: The prompt to send to Gemini
            response_schema: The JSON schema for the response
            max_retries: Number of retries for failed requests
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.gemini_api_key:
            self.logger.error("No Gemini API key configured")
            return None
            
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        request_body = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.95,
                "topK": 20,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }
        
        for attempt in range(max_retries):
            try:
                # Create SSL context that doesn't verify certificates (for local development)
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, 
                        json=request_body, 
                        headers=headers, 
                        ssl=ssl_context,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            # Extract the generated content
                            if 'candidates' in result and result['candidates']:
                                content = result['candidates'][0]['content']['parts'][0]['text']
                                
                                # Parse the JSON response
                                try:
                                    return json.loads(content)
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"Failed to parse JSON response: {e}")
                                    self.logger.debug(f"Raw content: {content}")
                                    return None
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Gemini API error (status {response.status}): {error_text}")
                            
                            if response.status == 429 and attempt < max_retries - 1:
                                wait_time = (attempt + 1) * 2
                                self.logger.info(f"Rate limited, waiting {wait_time}s before retry...")
                                await asyncio.sleep(wait_time)
                                continue
                                
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error calling Gemini API: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                    
        return None
    
    async def parse_readme_to_dict(self, content: str) -> Dict[str, Any]:
        """Parse README.md content into structured dictionary using Gemini."""
        
        # Define the expected structure
        response_schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "problem_statement": {"type": "string"},
                "target_audience": {"type": "string"},
                "value_proposition": {"type": "string"},
                "features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["name", "description"]
                    }
                },
                "tech_stack": {
                    "type": "object",
                    "properties": {
                        "backend": {"type": "string"},
                        "frontend": {"type": "string"},
                        "database": {"type": "string"},
                        "cache": {"type": "string"},
                        "other": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                },
                "architecture_overview": {"type": "string"},
                "installation_steps": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "prerequisites": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "development_workflow": {"type": "string"},
                "testing_approach": {"type": "string"},
                "deployment_strategy": {"type": "string"},
                "design_system": {
                    "type": "object",
                    "properties": {
                        "font": {"type": "string"},
                        "color_palette": {
                            "type": "object",
                            "properties": {
                                "primary": {"type": "string"},
                                "secondary": {"type": "string"},
                                "accent": {"type": "string"},
                                "background": {"type": "string"},
                                "text": {"type": "string"},
                                "success": {"type": "string"},
                                "warning": {"type": "string"},
                                "danger": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "required": ["title", "description", "features", "tech_stack"]
        }
        
        prompt = f"""Parse this README.md content into a structured format.
Extract all the key information and organize it properly.

README Content:
{content}

Instructions:
1. Extract the project title and description
2. Identify the problem statement and target audience
3. List all features with their names and descriptions
4. Extract tech stack components (backend, frontend, database, etc.)
5. Summarize the architecture overview
6. Extract installation steps as an array
7. List prerequisites
8. Include any design system details if present

Return a well-structured JSON object with all this information."""

        try:
            response = await self._call_gemini_structured(prompt, response_schema)
            return response or {}
            
        except Exception as e:
            self.logger.error(f"Error parsing README with Gemini: {e}")
            return {}
    
    async def parse_monetization_to_dict(self, content: str) -> Dict[str, Any]:
        """Parse MONETIZATION_STRATEGY.md content into structured dictionary using Gemini."""
        
        # Define the expected structure
        response_schema = {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string"},
                "revenue_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "required": ["name", "description"]
                    }
                },
                "pricing_tiers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "string"},
                            "features": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "target_market": {"type": "string"}
                        },
                        "required": ["name", "price", "features"]
                    }
                },
                "market_analysis": {
                    "type": "object",
                    "properties": {
                        "market_size": {"type": "string"},
                        "growth_rate": {"type": "string"},
                        "competitors": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "competitive_advantage": {"type": "string"}
                    }
                },
                "customer_acquisition": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "growth_projections": {
                    "type": "object",
                    "properties": {
                        "year_1": {"type": "string"},
                        "year_2": {"type": "string"},
                        "year_3": {"type": "string"}
                    }
                },
                "implementation_roadmap": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phase": {"type": "string"},
                            "timeline": {"type": "string"},
                            "objectives": {"type": "string"}
                        }
                    }
                },
                "key_metrics": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["executive_summary", "revenue_models", "pricing_tiers"]
        }
        
        prompt = f"""Parse this MONETIZATION_STRATEGY.md content into a structured format.
Extract all monetization and business strategy information.

Monetization Content:
{content}

Instructions:
1. Extract the executive summary
2. List all revenue models with descriptions
3. Extract pricing tiers with their features and prices
4. Summarize market analysis including competitors
5. List customer acquisition strategies
6. Extract growth projections for each year
7. Parse implementation roadmap with timelines
8. List key metrics to track

Return a well-structured JSON object with all this information."""

        try:
            response = await self._call_gemini_structured(prompt, response_schema)
            return response or {}
            
        except Exception as e:
            self.logger.error(f"Error parsing monetization with Gemini: {e}")
            return {}
    
    async def update_repository_metadata(self, repo_path: Path) -> bool:
        """Update a repository's metadata.json with Gemini-parsed content."""
        metadata_path = repo_path / 'metadata.json'
        readme_path = repo_path / 'README.md'
        monetization_path = repo_path / 'MONETIZATION_STRATEGY.md'
        
        if not metadata_path.exists():
            print(f"  ❌ No metadata.json found")
            return False
        
        try:
            # Load existing metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            updated = False
            
            # Parse README.md if it exists
            if readme_path.exists():
                print(f"  📖 Parsing README.md with Gemini...")
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                readme_structured = await self.parse_readme_to_dict(readme_content)
                if readme_structured:
                    metadata['readme_content'] = readme_structured
                    print(f"  ✅ Parsed README.md into structured format")
                    updated = True
                else:
                    print(f"  ⚠️  Failed to parse README.md")
            
            # Parse MONETIZATION_STRATEGY.md if it exists
            if monetization_path.exists():
                print(f"  💰 Parsing MONETIZATION_STRATEGY.md with Gemini...")
                with open(monetization_path, 'r', encoding='utf-8') as f:
                    monetization_content = f.read()
                
                monetization_structured = await self.parse_monetization_to_dict(monetization_content)
                if monetization_structured:
                    metadata['monetization_content'] = monetization_structured
                    print(f"  ✅ Parsed MONETIZATION_STRATEGY.md into structured format")
                    updated = True
                else:
                    print(f"  ⚠️  Failed to parse MONETIZATION_STRATEGY.md")
            
            # Save updated metadata if changes were made
            if updated:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                return True
            
            return False
            
        except Exception as e:
            print(f"  ❌ Error updating metadata: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Parse markdown files using Gemini for all repositories."""
    # Path to local repositories
    local_repos_path = Path("/Users/fortressarigo/PhpstormProjects/cwmai/local_repos/repositories/CodeWebMobile-AI")
    
    if not local_repos_path.exists():
        print(f"Local repositories directory not found: {local_repos_path}")
        return
    
    print(f"🔍 Processing repositories in: {local_repos_path}")
    print("🤖 Using Gemini AI to parse .md files into structured JSON format")
    print("📝 Using direct Gemini API calls with structured output\n")
    
    # Initialize parser
    parser = GeminiMarkdownParser()
    
    if not parser.gemini_api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please set: export GOOGLE_API_KEY='your-api-key'")
        return
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # Process each repository
    for repo_path in local_repos_path.iterdir():
        if repo_path.is_dir() and not repo_path.name.startswith('.'):
            print(f"\n📁 Processing: {repo_path.name}")
            result = await parser.update_repository_metadata(repo_path)
            if result is True:
                success_count += 1
            elif result is False:
                skip_count += 1
            else:
                error_count += 1
            
            # Add a small delay to avoid rate limiting
            await asyncio.sleep(1)
    
    print(f"\n📊 Summary:")
    print(f"  - Successfully updated: {success_count} repositories")
    print(f"  - Skipped: {skip_count} repositories")
    print(f"  - Errors: {error_count} repositories")
    
    # Show example of structured output
    if success_count > 0:
        example_repo = next(local_repos_path.iterdir())
        example_metadata = example_repo / 'metadata.json'
        if example_metadata.exists():
            with open(example_metadata, 'r') as f:
                data = json.load(f)
            
            print("\n📋 Example of structured content:")
            if 'readme_content' in data and isinstance(data['readme_content'], dict):
                print("\n  README structure:")
                for key in list(data['readme_content'].keys())[:5]:
                    value = data['readme_content'][key]
                    if isinstance(value, list):
                        print(f"    - {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"    - {key}: {len(value)} fields")
                    else:
                        print(f"    - {key}: {type(value).__name__}")
            
            if 'monetization_content' in data and isinstance(data['monetization_content'], dict):
                print("\n  Monetization structure:")
                for key in list(data['monetization_content'].keys())[:5]:
                    value = data['monetization_content'][key]
                    if isinstance(value, list):
                        print(f"    - {key}: {len(value)} items")
                    elif isinstance(value, dict):
                        print(f"    - {key}: {len(value)} fields")
                    else:
                        print(f"    - {key}: {type(value).__name__}")


if __name__ == "__main__":
    # Set up environment variables
    os.environ['PYTHONPATH'] = '/Users/fortressarigo/PhpstormProjects/cwmai'
    
    # Run the async main function
    asyncio.run(main())