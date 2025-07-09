"""
Simple Architecture Enhancement Tool

Enhances existing ARCHITECTURE.md files with admin/user separation and additional sections.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

# Import required modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.http_ai_client import HTTPAIClient
from scripts.logger import Logger


class ArchitectureEnhancer:
    """Enhance ARCHITECTURE.md files to comprehensive standards."""
    
    def __init__(self):
        self.logger = Logger()
        self.ai_client = HTTPAIClient()
        
    async def enhance_architecture(self, project_path: str) -> Dict[str, Any]:
        """Enhance a single ARCHITECTURE.md file."""
        project_path = Path(project_path)
        arch_file = project_path / "ARCHITECTURE.md"
        
        if not arch_file.exists():
            return {
                'success': False,
                'error': f'No ARCHITECTURE.md found at {arch_file}'
            }
            
        # Read existing content
        existing_content = arch_file.read_text()
        
        # Check if already comprehensive and not truncated
        if self._is_comprehensive(existing_content) and not self._is_truncated(existing_content):
            return {
                'success': True,
                'status': 'already_comprehensive',
                'message': 'Architecture file already meets comprehensive standards'
            }
            
        # Extract project name and existing sections
        project_name = project_path.name
        core_entities = self._extract_section(existing_content, "Core Entities")
        design_system = self._extract_section(existing_content, "Design System")
        
        # Check if content appears truncated
        if self._is_truncated(existing_content):
            self.logger.info(f"Detected truncated content for {project_name}, will use chunked approach")
            return await self._enhance_with_chunks(project_path, existing_content, project_name, core_entities, design_system)
        
        # Create enhancement prompt
        # Try chunked approach if response might be too large
        if len(existing_content) > 20000:
            self.logger.info(f"Large file detected ({len(existing_content)} chars), using chunked approach")
            return await self._enhance_with_chunks(project_path, existing_content, project_name, core_entities, design_system)
        
        prompt = f"""
You are enhancing an ARCHITECTURE.md file.

Project: {project_name}

Your task is to generate a COMPLETE enhanced architecture that:
1. Preserves the existing content structure and context
2. Completes any truncated sections
3. Adds the new enhancements requested below

PLEASE OUTPUT THE COMPLETE ENHANCED ARCHITECTURE.MD FILE IN MARKDOWN FORMAT.

Enhancements to add:

1. **Admin/User System Separation**:
   - Separate admin and user portals architecture
   - Role-based access control (RBAC) with roles: super_admin, admin, moderator, support, user
   - Admin-specific endpoints and controllers
   - User-specific endpoints and controllers
   - Permission management system

2. **Comprehensive API Documentation**:
   - Admin API endpoints (at least 15-20 endpoints)
     - User management (CRUD, role assignment, permissions)
     - System configuration and settings
     - Analytics and reporting endpoints
     - Content moderation endpoints
   - User API endpoints (at least 15-20 endpoints)
     - Authentication and profile management
     - Core feature endpoints
     - User-specific actions

3. **Frontend Structure Separation**:
   - Admin portal structure:
     - components/admin/ directory structure
     - Admin-specific layouts and navigation
     - Admin dashboard components
     - User management interfaces
   - User portal structure:
     - components/user/ directory structure
     - User-specific layouts
     - Feature components

4. **Database Schema Enhancements**:
   - Role and permission tables
   - Admin action audit logs
   - User activity tracking
   - System configuration tables

5. **Security Architecture**:
   - Admin authentication flow (2FA required)
   - User authentication flow
   - API rate limiting by role
   - Admin action logging and auditing

6. **Redis Configuration**:
   - Admin session management
   - User session management
   - Real-time admin notifications
   - Cache strategies by role

7. **Comprehensive Features List**:
   Based on the project type ({project_name}), include ALL features that should be implemented:
   - Core Features (10-15 features minimum)
   - Admin Features (5-10 features)
   - Advanced Features (5-10 features)
   - Integration Features (3-5 features)
   - Each feature should have:
     - Clear description
     - User story
     - Technical requirements
     - Priority level (High/Medium/Low)

IMPORTANT INSTRUCTIONS:
1. Output the COMPLETE enhanced file in proper Markdown format
2. Do not use code blocks or wrap the output in backticks
3. Start directly with the # heading
4. Keep existing Core Entities: {core_entities if core_entities else 'Create new Core Entities section'}
5. Keep existing Design System: {design_system if design_system else 'Create new Design System section'}
6. Add a "## Features Overview" section listing 30-40 features
7. Complete the Feature Implementation Roadmap with ALL features (don't truncate)
8. Make sure the document is comprehensive and complete

Here is the current content (which may be truncated):

{existing_content}

GENERATE THE COMPLETE ENHANCED FILE NOW:
"""

        # Get enhancement from AI
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        if not response:
            return {
                'success': False,
                'error': 'Failed to get AI response'
            }
            
        # Extract content from response
        if isinstance(response, dict):
            enhanced_content = response.get('content', '')
        else:
            enhanced_content = str(response)
            
        # Clean up any code block wrappers
        if enhanced_content.startswith('```'):
            # Remove code block markers
            lines = enhanced_content.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            enhanced_content = '\n'.join(lines)
        
        # Validate enhancement
        if not self._is_comprehensive(enhanced_content):
            # Try again with more specific instructions
            retry_prompt = f"""
The previous enhancement was not comprehensive enough. 

IMPORTANT: You must work with the EXISTING {project_name} project architecture.
Keep the current project context and enhance it - do not create a different project!

Please enhance the EXISTING architecture document to be at least 30,000 characters long.

Start with the EXISTING content and ADD these sections:

1. Overview (with Problem Statement and Target Audience)
2. Core Entities (keep existing if provided)
3. Design System (keep existing if provided)
4. System Architecture with:
   - High-level architecture diagram
   - Admin Portal Architecture
   - User Portal Architecture
   - API Gateway design
   - Service separation
5. Database Schema (with all tables for admin/user separation)
6. Comprehensive API Design:
   - At least 20 Admin endpoints
   - At least 20 User endpoints
   - Each with request/response examples
7. Frontend Structure:
   - Detailed admin component structure
   - Detailed user component structure
8. Redis and Real-time architecture
9. Authentication & Authorization (separate flows for admin/user)
10. Security & Hardening
11. Feature Implementation Roadmap (at least 5 features)

Make it VERY detailed and comprehensive. Include code examples, configuration details, and implementation notes.
"""
            
            retry_response = await self.ai_client.generate_enhanced_response(retry_prompt)
            
            if retry_response and retry_response.get('content'):
                enhanced_content = retry_response.get('content', '')
        
        # Save enhanced content
        arch_file.write_text(enhanced_content)
        
        return {
            'success': True,
            'status': 'enhanced',
            'original_size': len(existing_content),
            'enhanced_size': len(enhanced_content),
            'improvement_ratio': len(enhanced_content) / len(existing_content)
        }
    
    def _is_comprehensive(self, content: str) -> bool:
        """Check if architecture meets comprehensive standards."""
        required_sections = [
            'Admin Portal',
            'User Portal', 
            'Admin API',
            'User API',
            'Role-Based Access',
            'Frontend Structure',
            'Authentication & Authorization',
            'Features Overview'  # New requirement
        ]
        
        # Check length
        if len(content) < 25000:
            return False
            
        # Check for required sections
        content_lower = content.lower()
        for section in required_sections:
            if section.lower() not in content_lower:
                return False
                
        return True
    
    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract a section from the content."""
        # Try to find the section
        pattern = rf"## {section_name}.*?(?=\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(0).strip()
            
        # Try with ### 
        pattern = rf"### {section_name}.*?(?=\n### |\n## |\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(0).strip()
            
        return None
    
    def _is_truncated(self, content: str) -> bool:
        """Check if the document appears to be truncated."""
        content_trimmed = content.rstrip()
        
        # Signs of truncation
        truncation_signs = [
            content_trimmed.endswith('.tsx'),
            content_trimmed.endswith('.ts'),
            content_trimmed.endswith('.php'),
            content_trimmed.endswith(','),
            content_trimmed.endswith('-'),
            content_trimmed.endswith(':**'),
            not content_trimmed.endswith('.') and not content_trimmed.endswith('*') and not content_trimmed.endswith('}'),
            'Frontend Components:' in content_trimmed[-100:] and not content_trimmed.endswith('\n')
        ]
        
        return any(truncation_signs)
    
    async def _enhance_with_chunks(self, project_path: Path, existing_content: str, project_name: str, 
                                   core_entities: Optional[str], design_system: Optional[str]) -> Dict[str, Any]:
        """Enhance architecture by generating in chunks to avoid token limits."""
        self.logger.info(f"Using chunked enhancement for {project_name}")
        
        try:
            # Step 1: Generate the base enhancement (up to Features Overview)
            base_content = await self._generate_base_enhancement(project_name, existing_content, core_entities, design_system)
            
            # Step 2: Generate Feature Implementation Roadmap separately
            features_content = await self._generate_features_roadmap(project_name, existing_content)
            
            # Step 3: Combine the results
            if base_content and features_content:
                # Find where to insert the features
                if "## Feature Implementation Roadmap" in base_content:
                    # Replace placeholder with actual features
                    parts = base_content.split("## Feature Implementation Roadmap")
                    complete_content = parts[0] + "## Feature Implementation Roadmap\n\n" + features_content
                else:
                    # Append features
                    complete_content = base_content + "\n\n## Feature Implementation Roadmap\n\n" + features_content
                
                # Save the complete content
                arch_file = project_path / "ARCHITECTURE.md"
                arch_file.write_text(complete_content)
                
                return {
                    'success': True,
                    'status': 'enhanced',
                    'original_size': len(existing_content),
                    'enhanced_size': len(complete_content),
                    'improvement_ratio': len(complete_content) / len(existing_content)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to generate enhancement in chunks'
                }
                
        except Exception as e:
            self.logger.error(f"Error in chunked enhancement: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_base_enhancement(self, project_name: str, existing_content: str, 
                                         core_entities: Optional[str], design_system: Optional[str]) -> Optional[str]:
        """Generate the base enhancement without the full Feature Implementation Roadmap."""
        prompt = f"""
Enhance the ARCHITECTURE.md for {project_name} project.

Based on the existing content, this appears to be a project about:
{existing_content[:1000]}...

Generate a comprehensive architecture document that includes:

1. Overview (with Problem Statement and Target Audience based on the project context)
2. Core Entities (keep if exists: {core_entities if core_entities else 'Analyze the project and create appropriate entities'})
3. Design System (keep if exists: {design_system if design_system else 'Create appropriate design system'})
4. System Architecture with admin/user separation
5. Database Schema with RBAC tables
6. Comprehensive API Design (20+ admin endpoints, 20+ user endpoints)
7. Frontend Structure with admin/user portals
8. Real-time & Events Architecture
9. Authentication & Authorization with RBAC
10. Deployment & Scalability
11. Testing Strategy
12. Security & Hardening
13. Logging & Observability
14. Features Overview (analyze the project type and list 30-40 relevant features)

For the Feature Implementation Roadmap section, just add a placeholder:
## Feature Implementation Roadmap
[Features will be added here]

Make it detailed and comprehensive but DO NOT include the actual feature implementations yet.
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        return response.get('content', '') if isinstance(response, dict) else str(response)
    
    async def _generate_features_roadmap(self, project_name: str, existing_content: str) -> Optional[str]:
        """Generate the complete Feature Implementation Roadmap."""
        prompt = f"""
Analyze this project and generate a Feature Implementation Roadmap.

Project: {project_name}

Existing architecture content:
{existing_content[:2000]}...

Based on:
1. The project name
2. The problem statement and target audience
3. The core entities mentioned
4. The overall architecture

Identify and implement the 5 MOST IMPORTANT features this project should have.

For EACH feature, provide:
- Feature number and title
- Description (2-3 sentences explaining what it does)
- User Story
- Technical Requirements (bullet points)
- Database Changes (specific tables and fields)
- Backend Components (models, controllers, services)
- Frontend Components (specific .tsx files and their purposes)
- API Endpoints (if applicable)

Use this format:

### 1. [Feature Name]
**Description:** [Description]

**User Story:** As a [user type], I want to [action] so that [benefit].

**Technical Requirements:**
- [Requirement 1]
- [Requirement 2]

**Database Changes:**
- [Change 1]

**Backend Components:**
- [Component 1]

**Frontend Components:**
- [Component 1]

**API Endpoints:**
- `METHOD /api/path`: Description

Generate ALL 5 features with complete details based on your analysis of what this specific project needs.
"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        return response.get('content', '') if isinstance(response, dict) else str(response)


async def enhance_architecture_documentation(
    project_path: str = None,
    enhance_all: bool = False,
    base_path: str = None,
    project_names: List[str] = None
) -> Dict[str, Any]:
    """
    Enhance ARCHITECTURE.md files to meet comprehensive standards.
    
    Args:
        project_path: Path to specific project
        enhance_all: Enhance all projects
        base_path: Base directory for projects
        project_names: List of specific projects
        
    Returns:
        Enhancement results
    """
    enhancer = ArchitectureEnhancer()
    
    # Handle single project
    if project_path:
        result = await enhancer.enhance_architecture(project_path)
        return {
            'success': result['success'],
            'enhanced': 1 if result.get('status') == 'enhanced' else 0,
            'skipped': 1 if result.get('status') == 'already_comprehensive' else 0,
            'failed': 0 if result['success'] else 1,
            'details': {
                Path(project_path).name: result
            }
        }
    
    # Handle multiple projects
    if not base_path:
        base_path = Path.cwd() / "local_repos" / "repositories" / "CodeWebMobile-AI"
    else:
        base_path = Path(base_path)
        
    projects = []
    if enhance_all:
        projects = [p for p in base_path.iterdir() if p.is_dir() and (p / "ARCHITECTURE.md").exists()]
    elif project_names:
        projects = [base_path / name for name in project_names if (base_path / name / "ARCHITECTURE.md").exists()]
    else:
        return {
            'success': False,
            'error': 'No projects specified'
        }
    
    results = {
        'success': True,
        'enhanced': 0,
        'skipped': 0,
        'failed': 0,
        'details': {}
    }
    
    for project in projects:
        try:
            result = await enhancer.enhance_architecture(str(project))
            results['details'][project.name] = result
            
            if result['success']:
                if result.get('status') == 'enhanced':
                    results['enhanced'] += 1
                elif result.get('status') == 'already_comprehensive':
                    results['skipped'] += 1
            else:
                results['failed'] += 1
                
        except Exception as e:
            results['failed'] += 1
            results['details'][project.name] = {
                'success': False,
                'error': str(e)
            }
    
    return results


# For testing
if __name__ == "__main__":
    async def test():
        result = await enhance_architecture_documentation(
            project_path=str(Path.cwd() / "local_repos" / "repositories" / "CodeWebMobile-AI" / "micronutrient-optimizer")
        )
        print(f"Result: {result}")
    
    asyncio.run(test())