#!/usr/bin/env python3
"""
Update all repositories with detailed architecture documents matching new project quality.
"""

import os
import asyncio
from github import Github
from scripts.http_ai_client import HTTPAIClient
import json
import re
from typing import Dict, Any


# Repository-specific context based on their names and purposes
REPO_CONTEXTS = {
    'eco-track-ai': {
        'problem': 'Businesses and individuals struggle to track and reduce their carbon footprint. Without visibility into environmental impact, making eco-friendly decisions is difficult, leading to unnecessary waste and emissions.',
        'audience': 'Environmentally conscious businesses, sustainability managers, eco-minded individuals, and organizations with ESG reporting requirements',
        'features': [
            'Carbon footprint calculator for activities and purchases',
            'Real-time emissions tracking across categories',
            'Sustainability recommendations and alternatives',
            'Progress tracking and goal setting',
            'ESG reporting and compliance tools',
            'Community challenges and achievements'
        ],
        'entities': ['User', 'Activity', 'Emission', 'Goal', 'Report', 'Challenge', 'Achievement', 'Recommendation']
    },
    'ai-powered-inventory-sync': {
        'problem': 'Multi-channel retailers lose sales and customer trust due to inventory discrepancies. Manual inventory management leads to overselling, stockouts, and inefficient warehouse operations.',
        'audience': 'E-commerce businesses, multi-location retailers, warehouse managers, and omnichannel merchants managing inventory across platforms',
        'features': [
            'Real-time inventory synchronization across channels',
            'AI-powered demand forecasting',
            'Automatic reorder point calculations',
            'Multi-warehouse inventory optimization',
            'Low stock alerts and automated purchasing',
            'Barcode scanning and RFID integration',
            'Inventory analytics and reporting'
        ],
        'entities': ['Product', 'Warehouse', 'Channel', 'StockLevel', 'Order', 'Forecast', 'Transfer', 'Supplier']
    },
    'community-connect-platform': {
        'problem': 'Local communities lack centralized platforms to organize events, share resources, and build meaningful connections. Neighbors remain strangers, local businesses struggle to reach residents, and community resources go underutilized.',
        'audience': 'Local residents, community organizers, small business owners, neighborhood associations, and municipal governments',
        'features': [
            'Neighborhood event creation and discovery',
            'Local marketplace for goods and services',
            'Community resource sharing (tools, skills)',
            'Neighborhood watch and safety alerts',
            'Local business directory and promotions',
            'Community polls and decision making',
            'Volunteer opportunity matching'
        ],
        'entities': ['User', 'Community', 'Event', 'Listing', 'Business', 'Alert', 'Poll', 'Resource']
    },
    'brand-guardian-ai': {
        'problem': 'Brands face reputation damage from negative reviews, social media crises, and misinformation spreading faster than they can respond. Manual monitoring misses critical mentions, leading to PR disasters.',
        'audience': 'Brand managers, PR agencies, social media managers, and businesses concerned about online reputation',
        'features': [
            'Real-time brand mention monitoring across platforms',
            'Sentiment analysis and threat detection',
            'Automated response suggestions',
            'Crisis alert system with severity scoring',
            'Competitor reputation tracking',
            'Influencer identification and tracking',
            'Reputation trend analytics and reporting'
        ],
        'entities': ['Brand', 'Mention', 'Sentiment', 'Alert', 'Response', 'Platform', 'Influencer', 'Crisis']
    },
    'reputation-ai': {
        'problem': 'Online reviews make or break businesses, but managing reviews across multiple platforms is overwhelming. Negative reviews go unaddressed, positive customers aren\'t encouraged to review, and fake reviews damage credibility.',
        'audience': 'Small business owners, restaurant managers, hotel operators, and service providers who rely on online reviews',
        'features': [
            'Multi-platform review aggregation',
            'AI-powered review response generation',
            'Review sentiment analysis and insights',
            'Fake review detection',
            'Customer feedback request automation',
            'Reputation score tracking',
            'Competitive review analysis'
        ],
        'entities': ['Business', 'Review', 'Platform', 'Response', 'Customer', 'Sentiment', 'Score', 'Competitor']
    },
    'mindleap-ai': {
        'problem': 'Mental health support is expensive, inaccessible, and stigmatized. People struggling with stress, anxiety, or emotional challenges often have nowhere to turn, especially outside business hours.',
        'audience': 'Individuals seeking mental wellness support, people with mild anxiety/stress, those exploring therapy, and users wanting 24/7 emotional support',
        'features': [
            '24/7 AI mental health chatbot',
            'Personalized coping strategies',
            'Mood tracking and patterns',
            'Crisis intervention and helpline connection',
            'Guided meditation and exercises',
            'Progress tracking and insights',
            'Therapist matching and referrals'
        ],
        'entities': ['User', 'Session', 'Mood', 'Exercise', 'Crisis', 'Referral', 'Progress', 'Strategy']
    },
    'vid-gen-ai': {
        'problem': 'Small businesses can\'t afford professional video marketing. Creating engaging marketing videos requires expensive equipment, software, and expertise, leaving them unable to compete with larger companies.',
        'audience': 'Small business owners, solopreneurs, social media managers, and content creators needing quick marketing videos',
        'features': [
            'AI-powered video generation from product info',
            'Customizable video templates by industry',
            'Automatic scene composition and transitions',
            'AI voiceover and background music',
            'Brand color and logo integration',
            'Multiple format exports for different platforms',
            'A/B testing for video variations'
        ],
        'entities': ['User', 'Product', 'Video', 'Template', 'Scene', 'Asset', 'Brand', 'Export']
    },
    'review-sentry': {
        'problem': 'Businesses miss critical feedback buried in reviews across dozens of platforms. By the time they respond to complaints, customers are lost and damage is done to their reputation.',
        'audience': 'Multi-location businesses, franchise owners, hospitality managers, and customer service teams',
        'features': [
            'Real-time review monitoring across 50+ platforms',
            'Urgent issue detection and alerts',
            'Team assignment and response workflows',
            'Response template library',
            'Review analytics by location and time',
            'Customer sentiment trending',
            'Automated thank you responses'
        ],
        'entities': ['Business', 'Location', 'Review', 'Alert', 'Response', 'Team', 'Workflow', 'Template']
    },
    'project-analytics-dashboard': {
        'problem': 'Project managers lack real-time visibility into project health, budget burns, and team productivity. Decisions are made on outdated data, leading to budget overruns and missed deadlines.',
        'audience': 'Project managers, team leads, executives, and PMO departments needing data-driven project insights',
        'features': [
            'Real-time project health metrics',
            'Budget tracking and burn rate analysis',
            'Team productivity and velocity metrics',
            'Risk identification and mitigation tracking',
            'Milestone and deadline monitoring',
            'Resource allocation optimization',
            'Custom KPI dashboards'
        ],
        'entities': ['Project', 'Task', 'Milestone', 'Budget', 'Resource', 'Risk', 'Metric', 'Dashboard']
    },
    'business-analytics-dashboard': {
        'problem': 'Business owners drown in data from multiple sources but lack actionable insights. Critical business metrics are scattered across different tools, making it impossible to see the full picture.',
        'audience': 'Business owners, executives, operations managers, and analysts needing unified business intelligence',
        'features': [
            'Multi-source data integration',
            'Real-time KPI monitoring',
            'Predictive analytics and forecasting',
            'Custom metric creation',
            'Automated insight generation',
            'Mobile executive dashboards',
            'Anomaly detection and alerts'
        ],
        'entities': ['DataSource', 'Metric', 'Dashboard', 'Alert', 'Forecast', 'Insight', 'Report', 'Integration']
    }
}


async def generate_detailed_architecture(repo_name: str, context: Dict[str, Any], ai_client: HTTPAIClient) -> str:
    """Generate a detailed architecture document for a repository."""
    
    prompt = f"""
    You are a distinguished CTO and Principal Engineer creating a comprehensive architecture document.
    
    Generate a DETAILED architecture document for: {repo_name}
    
    Context:
    - Problem: {context['problem']}
    - Target Audience: {context['audience']}
    - Key Features: {json.dumps(context['features'])}
    - Core Entities: {json.dumps(context['entities'])}
    
    Create an architecture document that includes:
    
    1. Overview with problem statement and target audience
    2. Design System with specific typography and color scheme
    3. Detailed database schema with CREATE TABLE statements
    4. Complete API endpoints with examples
    5. Frontend component structure
    6. Feature implementation roadmap with technical details
    
    The document should be:
    - Specific to THIS project (not generic)
    - Include actual SQL schemas
    - Have real API endpoints
    - Contain implementation details
    - Be at least 3000 words
    
    Format as markdown with proper sections and code blocks.
    """
    
    try:
        response = await ai_client.generate_enhanced_response(prompt)
        
        if isinstance(response, dict):
            content = response.get('content', '')
        else:
            content = str(response)
        
        # Ensure we got substantial content
        if len(content) < 1000:
            print(f"⚠️  Generated content too short for {repo_name}, using template")
            return generate_template_architecture(repo_name, context)
        
        return content
        
    except Exception as e:
        print(f"⚠️  AI generation failed for {repo_name}: {e}")
        return generate_template_architecture(repo_name, context)


def generate_template_architecture(repo_name: str, context: Dict[str, Any]) -> str:
    """Generate a detailed architecture from template when AI fails."""
    
    # Format repo name for display
    display_name = repo_name.replace('-', ' ').title()
    
    # Generate color scheme based on project type
    color_schemes = {
        'eco': {'primary': '#10B981', 'secondary': '#059669'},
        'ai': {'primary': '#7C3AED', 'secondary': '#6D28D9'},
        'community': {'primary': '#3B82F6', 'secondary': '#2563EB'},
        'brand': {'primary': '#F59E0B', 'secondary': '#D97706'},
        'review': {'primary': '#EF4444', 'secondary': '#DC2626'},
        'mind': {'primary': '#8B5CF6', 'secondary': '#7C3AED'},
        'vid': {'primary': '#EC4899', 'secondary': '#DB2777'},
        'analytics': {'primary': '#6366F1', 'secondary': '#4F46E5'}
    }
    
    # Determine color scheme
    colors = {'primary': '#2563EB', 'secondary': '#1E40AF'}  # Default
    for key, scheme in color_schemes.items():
        if key in repo_name.lower():
            colors = scheme
            break
    
    # Generate comprehensive architecture
    architecture = f"""# {repo_name} Architecture

## Overview
{context['problem']}

### Problem Statement
{context['problem']}

### Target Audience
{context['audience']}

### Core Entities
"""
    
    for entity in context['entities']:
        architecture += f"- {entity}\n"
    
    architecture += f"""

## Design System

### Typography
- **Font**: Inter
- **Font Stack**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`
- **Google Fonts**: https://fonts.google.com/specimen/Inter
- **Rationale**: Inter provides excellent readability across all device sizes with a modern, professional appearance perfect for {display_name.lower()}.

### Color Palette
- **Primary**: `{colors['primary']}` - Main brand color for primary actions and key UI elements
- **Secondary**: `{colors['secondary']}` - Supporting color for secondary actions
- **Accent**: `#F59E0B` - Highlight color for important notifications
- **Success**: `#10B981` - Positive actions and success states
- **Warning**: `#F59E0B` - Warning messages and caution states
- **Error**: `#EF4444` - Error states and destructive actions
- **Background**: `#F9FAFB` - Main application background
- **Surface**: `#FFFFFF` - Card and component backgrounds
- **Text Primary**: `#111827` - Main text color
- **Text Secondary**: `#6B7280` - Secondary and helper text

## Foundational Architecture

### 1. Core Components & Rationale

The {display_name} architecture is built on a microservices-inspired monolithic structure, providing the benefits of service isolation while maintaining deployment simplicity.

**Backend Services**:
"""
    
    # Add service components based on features
    for i, feature in enumerate(context['features'][:4]):
        service_name = feature.split()[0] + 'Service'
        architecture += f"- `{service_name}`: {feature}\n"
    
    architecture += f"""
**Frontend Components**:
- `AppShell`: Main application wrapper with navigation and authentication state
- `DashboardView`: Primary user interface showing key metrics and actions
- `DataVisualization`: Rich charts and graphs for analytics
- `ActionCenter`: Quick access to primary user actions
- `NotificationHub`: Real-time updates and alerts

### 2. Database Schema Design

```sql
-- Core user management
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role ENUM('user', 'admin', 'manager') DEFAULT 'user',
    subscription_tier VARCHAR(50) DEFAULT 'free',
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_role (role)
);
"""
    
    # Add entity-specific tables
    for entity in context['entities'][:4]:
        table_name = entity.lower() + 's'
        architecture += f"""
-- {entity} management
CREATE TABLE {table_name} (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status ENUM('active', 'inactive', 'pending') DEFAULT 'active',
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_{table_name} (user_id, created_at),
    INDEX idx_status (status)
);
"""
    
    architecture += """```

### 3. API Design & Key Endpoints

**Authentication & Authorization**:
- `POST /api/auth/register` - User registration with email verification
- `POST /api/auth/login` - User authentication returning JWT token
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/logout` - Invalidate refresh token
- `GET /api/auth/me` - Get current user profile

"""
    
    # Add feature-specific endpoints
    for entity in context['entities'][:3]:
        entity_lower = entity.lower()
        architecture += f"""**{entity} Management**:
- `GET /api/{entity_lower}s` - List all {entity_lower}s with pagination
- `POST /api/{entity_lower}s` - Create new {entity_lower}
- `GET /api/{entity_lower}s/{{id}}` - Get specific {entity_lower} details
- `PUT /api/{entity_lower}s/{{id}}` - Update {entity_lower}
- `DELETE /api/{entity_lower}s/{{id}}` - Delete {entity_lower}

"""
    
    architecture += """### 4. Frontend Structure

```
src/
├── components/
│   ├── common/
│   │   ├── Layout/
│   │   ├── Navigation/
│   │   ├── LoadingStates/
│   │   └── ErrorBoundary/
│   ├── features/
"""
    
    # Add feature components
    for entity in context['entities'][:3]:
        entity_lower = entity.lower()
        architecture += f"""│   │   ├── {entity_lower}/
│   │   │   ├── {entity}List.tsx
│   │   │   ├── {entity}Detail.tsx
│   │   │   ├── {entity}Form.tsx
│   │   │   └── {entity}Card.tsx
"""
    
    architecture += """│   └── ui/
│       ├── Button/
│       ├── Card/
│       ├── Modal/
│       └── Form/
├── hooks/
│   ├── useAuth.ts
│   ├── useApi.ts
│   └── useWebSocket.ts
├── services/
│   ├── api.service.ts
│   ├── auth.service.ts
│   └── websocket.service.ts
└── utils/
    ├── validators.ts
    ├── formatters.ts
    └── constants.ts
```

## Feature Implementation Roadmap

"""
    
    # Add detailed feature implementations
    for i, feature in enumerate(context['features'][:3]):
        architecture += f"""### {i+1}. {feature}
**Priority**: {'High' if i == 0 else 'Medium'}
**Complexity**: {'High' if 'AI' in feature else 'Medium'}

**Database Changes**:
- Add required tables and indexes
- Create junction tables for many-to-many relationships
- Add audit logging tables

**Backend Implementation**:
- Service layer for business logic
- Repository pattern for data access
- Event-driven updates via websockets
- Background job processing

**Frontend Components**:
- Responsive UI components
- Real-time data updates
- Progressive enhancement
- Accessibility compliance

**API Endpoints**:
- RESTful resource endpoints
- Bulk operations support
- Filtering and sorting
- Rate limiting

"""
    
    architecture += """## Technical Specifications

### Performance Requirements
- Page load time < 2 seconds
- API response time < 200ms for queries
- Support 10,000 concurrent users
- 99.9% uptime SLA

### Security Measures
- JWT authentication with refresh tokens
- Rate limiting per user/IP
- Input validation and sanitization
- SQL injection prevention
- XSS protection headers
- CORS configuration

### Scalability Approach
- Horizontal scaling ready
- Database read replicas
- Redis caching layer
- CDN for static assets
- Queue-based job processing

### Monitoring & Observability
- Application performance monitoring
- Error tracking and alerting
- User behavior analytics
- System health dashboards
- Audit logging

## Development Guidelines

### Code Standards
- TypeScript strict mode
- ESLint + Prettier configuration
- Git commit conventions
- PR review requirements
- Test coverage > 80%

### Testing Strategy
- Unit tests for services
- Integration tests for APIs
- E2E tests for critical paths
- Performance testing
- Security scanning

### Deployment Pipeline
- Git-based workflows
- Automated CI/CD
- Blue-green deployments
- Rollback capabilities
- Environment promotion

## Success Metrics
- User engagement rate > 70%
- Feature adoption > 50%
- System reliability > 99.9%
- User satisfaction > 4.5/5
- Performance benchmarks met"""
    
    return architecture


async def update_repository_architecture(repo_name: str, ai_client: HTTPAIClient):
    """Update a single repository's architecture."""
    try:
        print(f"\n🏗️  Updating architecture for {repo_name}...")
        
        # Get context
        context = REPO_CONTEXTS.get(repo_name, {
            'problem': f'Solution for {repo_name.replace("-", " ")}',
            'audience': 'Target users',
            'features': ['Core functionality'],
            'entities': ['User', 'Data']
        })
        
        # Generate architecture
        architecture_content = await generate_detailed_architecture(repo_name, context, ai_client)
        
        if not architecture_content:
            print(f"  ❌ Failed to generate content")
            return False
        
        # Update in GitHub
        g = Github(os.environ['GITHUB_TOKEN'])
        repo = g.get_repo(f'CodeWebMobile-AI/{repo_name}')
        
        try:
            arch_file = repo.get_contents('ARCHITECTURE.md')
            repo.update_file(
                'ARCHITECTURE.md',
                f'Update architecture with detailed project-specific content',
                architecture_content,
                arch_file.sha,
                branch='main'
            )
            print(f"  ✅ Successfully updated ARCHITECTURE.md")
            return True
        except Exception as e:
            print(f"  ❌ Failed to update file: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def main():
    """Update all repositories with generic architectures."""
    
    repos_to_update = list(REPO_CONTEXTS.keys())
    
    print(f"🚀 Updating {len(repos_to_update)} repositories with detailed architectures...\n")
    
    # Initialize AI client
    ai_client = HTTPAIClient()
    
    success_count = 0
    
    for repo_name in repos_to_update:
        if await update_repository_architecture(repo_name, ai_client):
            success_count += 1
        
        # Rate limiting
        await asyncio.sleep(3)
    
    print(f"\n📊 Summary: Successfully updated {success_count}/{len(repos_to_update)} repositories")
    
    if success_count < len(repos_to_update):
        print("\n⚠️  Some updates failed. You may need to run again or update manually.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.env.local')
    
    asyncio.run(main())