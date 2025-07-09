"""
Feature Generator - Single Source of Truth

Generates concrete, implementable features for projects based on market research.
This is the ONLY place where features should be generated to avoid duplication.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from scripts.http_ai_client import HTTPAIClient


class FeatureGenerator:
    """Single source of truth for AI-powered feature generation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_client = HTTPAIClient()
    
    async def generate_features(self, context: Dict[str, Any]) -> List[str]:
        """Generate 5-7 concrete, implementable features based on problem context.
        
        Args:
            context: Dictionary containing problem details:
                - problem_description: The problem being solved
                - affected_audience: Who experiences this problem
                - current_solutions: Existing solutions and their gaps
                - opportunity: The business opportunity
                - market_size: Size of the market (optional)
                - monetization_model: How to make money (optional)
                
        Returns:
            List of 5-7 feature names (3-5 words each)
        """
        self.logger.info("🎯 Generating features for problem context")
        
        # Build comprehensive context
        prompt = self._build_feature_prompt(context)
        
        try:
            # Use AI to generate features
            response = await self.ai_client.generate_enhanced_response(
                prompt,
                model='gemini'
            )
            
            # Parse features from response
            features = self._parse_features_from_response(response.get('content', ''))
            
            # Validate features
            if len(features) < 5:
                self.logger.warning(f"Only got {len(features)} features, generating more...")
                additional_features = await self._generate_additional_features(context, features)
                features.extend(additional_features)
            
            # Ensure quality
            features = self._validate_and_clean_features(features, context)
            
            self.logger.info(f"✅ Generated {len(features)} high-quality features")
            return features[:7]  # Return max 7 features
            
        except Exception as e:
            self.logger.error(f"Failed to generate features: {e}")
            # Emergency fallback - extract from context
            return self._extract_features_from_context(context)
    
    def _build_feature_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for feature generation."""
        problem = context.get('problem_description', '')
        audience = context.get('affected_audience', '')
        current = context.get('current_solutions', '')
        opportunity = context.get('opportunity', '')
        market_size = context.get('market_size', '')
        monetization = context.get('monetization_model', '')
        
        # Extract industry/domain from context
        industry_hints = self._extract_industry_hints(context)
        
        prompt = f"""You are a product architect designing features for a new solution. 
Analyze this market research and generate 5-7 specific, implementable features.

PROBLEM CONTEXT:
- Problem: {problem}
- Affected Users: {audience}
- Market Size: {market_size}
- Current Solutions: {current}
- Business Opportunity: {opportunity}
- Revenue Model: {monetization}
- Industry Indicators: {industry_hints}

REQUIREMENTS:
1. Each feature must directly solve aspects of the stated problem
2. Features should be concrete, buildable components (not vague concepts)
3. Feature names should be 3-5 words, specific to this domain
4. Avoid generic terms like "Dashboard", "API", "Database", "Login"
5. Focus on unique value propositions that differentiate from current solutions

GOOD EXAMPLES for different domains:
- Healthcare: "Automated Insurance Verification", "Smart Appointment Routing", "Clinical Decision Support"
- E-commerce: "Dynamic Pricing Engine", "Abandoned Cart Recovery", "Personalized Product Recommendations"
- Education: "Adaptive Learning Pathways", "Real-time Progress Analytics", "Peer Collaboration Tools"

BAD EXAMPLES (too generic):
- "User Management System"
- "Analytics Dashboard"
- "API Integration"
- "Mobile App"

Generate exactly 5-7 features. Return ONLY a JSON array of feature names, nothing else:
["Feature One Name", "Feature Two Name", ...]"""
        
        return prompt
    
    def _extract_industry_hints(self, context: Dict[str, Any]) -> str:
        """Extract industry/domain hints from context."""
        text = ' '.join(str(v) for v in context.values() if v)
        
        # Industry keywords
        industries = {
            'healthcare': ['health', 'medical', 'patient', 'doctor', 'clinic', 'hospital'],
            'education': ['student', 'teacher', 'learning', 'course', 'education'],
            'finance': ['payment', 'banking', 'investment', 'finance', 'money'],
            'retail': ['shop', 'store', 'product', 'inventory', 'retail'],
            'logistics': ['shipping', 'delivery', 'warehouse', 'supply chain'],
            'real estate': ['property', 'rental', 'housing', 'tenant', 'landlord'],
            'entertainment': ['media', 'content', 'streaming', 'gaming', 'entertainment']
        }
        
        found_industries = []
        text_lower = text.lower()
        
        for industry, keywords in industries.items():
            if any(keyword in text_lower for keyword in keywords):
                found_industries.append(industry)
        
        return ', '.join(found_industries) if found_industries else 'general business'
    
    def _parse_features_from_response(self, response: str) -> List[str]:
        """Parse features from AI response with multiple strategies."""
        features = []
        
        # Strategy 1: Try to parse as JSON array
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                features = json.loads(json_match.group())
                if isinstance(features, list) and all(isinstance(f, str) for f in features):
                    return [f.strip() for f in features if f.strip()]
        except:
            pass
        
        # Strategy 2: Look for numbered list
        numbered_pattern = r'^\d+\.\s*(.+)$'
        for line in response.split('\n'):
            match = re.match(numbered_pattern, line.strip())
            if match:
                feature = match.group(1).strip()
                # Remove quotes if present
                feature = feature.strip('"\'')
                if feature and len(feature.split()) <= 6:
                    features.append(feature)
        
        if features:
            return features
        
        # Strategy 3: Look for bullet points
        bullet_pattern = r'^[-•*]\s*(.+)$'
        for line in response.split('\n'):
            match = re.match(bullet_pattern, line.strip())
            if match:
                feature = match.group(1).strip()
                feature = feature.strip('"\'')
                if feature and len(feature.split()) <= 6:
                    features.append(feature)
        
        return features
    
    def _validate_and_clean_features(self, features: List[str], context: Dict[str, Any]) -> List[str]:
        """Validate and clean features for quality."""
        cleaned = []
        generic_terms = {
            'dashboard', 'api', 'database', 'platform', 'system', 
            'interface', 'module', 'component', 'service', 'tool'
        }
        
        for feature in features:
            # Clean the feature
            feature = feature.strip().strip('"\'').strip()
            
            # Skip if too short or too long
            words = feature.split()
            if len(words) < 2 or len(words) > 6:
                continue
            
            # Skip if it's just generic terms
            feature_lower = feature.lower()
            if all(word in generic_terms for word in feature_lower.split()):
                continue
            
            # Skip if it starts with generic articles
            if feature_lower.startswith(('the ', 'a ', 'an ')):
                continue
            
            cleaned.append(feature)
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for f in cleaned:
            if f.lower() not in seen:
                seen.add(f.lower())
                unique.append(f)
        
        return unique
    
    async def _generate_additional_features(self, context: Dict[str, Any], existing: List[str]) -> List[str]:
        """Generate additional features if we don't have enough."""
        existing_str = ', '.join(existing)
        prompt = f"""Based on this context, generate 3 more UNIQUE features that are different from these existing ones: {existing_str}

Context: {context.get('problem_description', '')}
Audience: {context.get('affected_audience', '')}

Return ONLY a JSON array of 3 new feature names:"""
        
        try:
            response = await self.ai_client.generate_enhanced_response(prompt, model='gemini')
            return self._parse_features_from_response(response.get('content', ''))
        except:
            return []
    
    def _extract_features_from_context(self, context: Dict[str, Any]) -> List[str]:
        """Emergency fallback: extract features from problem description."""
        features = []
        opportunity = context.get('opportunity', '')
        problem = context.get('problem_description', '')
        
        # Pattern matching for feature-like phrases
        patterns = [
            r'automated\s+(\w+\s+\w+)',
            r'intelligent\s+(\w+\s+\w+)',
            r'smart\s+(\w+\s+\w+)',
            r'real-time\s+(\w+\s+\w+)',
            r'(\w+\s+tracking)',
            r'(\w+\s+management)',
            r'(\w+\s+optimization)',
            r'(\w+\s+analytics)'
        ]
        
        text = f"{opportunity} {problem}"
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                feature = match.strip().title()
                if len(feature.split()) >= 2:
                    features.append(feature)
        
        # Ensure we have at least 5 features
        if len(features) < 5:
            # Generate based on audience
            audience = context.get('affected_audience', '').lower()
            if 'healthcare' in audience or 'medical' in audience:
                features.extend([
                    "Patient Record Automation",
                    "Clinical Workflow Optimization",
                    "Insurance Claim Processing",
                    "Appointment Scheduling System",
                    "Medical Compliance Tracking"
                ])
            elif 'business' in audience or 'company' in audience:
                features.extend([
                    "Customer Relationship Manager",
                    "Revenue Analytics Dashboard",
                    "Workflow Automation Engine",
                    "Team Collaboration Hub",
                    "Performance Metrics Tracker"
                ])
        
        return features[:7]