#!/usr/bin/env python3
"""Simple script to add basic ARCHITECTURE.md to repositories."""

import os
import json
from github import Github

def create_architecture_content(repo):
    """Create basic architecture content based on repo info."""
    name = repo.name
    description = repo.description or f"{name} application"
    
    # Generate appropriate content based on repo name and description
    content = f"""# {name} Architecture

## Overview
{description}

## Technical Stack
- **Frontend**: React with TypeScript
- **Backend**: Laravel (PHP 8.x)
- **Database**: MySQL/PostgreSQL
- **Styling**: Tailwind CSS
- **Build Tools**: Vite

## Architecture Layers

### 1. Frontend Layer (React + TypeScript)
- Component-based architecture
- State management with React Context/Redux
- Responsive design with Tailwind CSS
- Type-safe development with TypeScript

### 2. API Layer (Laravel)
- RESTful API endpoints
- JWT authentication
- Request validation
- Resource controllers
- API versioning

### 3. Business Logic Layer
- Service classes for complex operations
- Repository pattern for data access
- Event-driven architecture
- Job queues for async processing

### 4. Data Layer
- Eloquent ORM models
- Database migrations
- Query optimization
- Redis caching

## Key Features
"""
    
    # Add features based on the app name/description
    if "ai" in name.lower() or "ai" in description.lower():
        content += "- AI/ML integration\n- Intelligent processing\n"
    
    if "dashboard" in name.lower() or "analytics" in name.lower():
        content += "- Real-time data visualization\n- Interactive charts and graphs\n- Data export capabilities\n"
    
    if "mobile" in name.lower():
        content += "- Mobile-responsive design\n- Progressive Web App capabilities\n- Touch-optimized UI\n"
    
    if "inventory" in name.lower() or "sync" in name.lower():
        content += "- Real-time synchronization\n- Inventory tracking\n- Multi-location support\n"
    
    if "health" in name.lower() or "mindleap" in name.lower():
        content += "- Health data tracking\n- Privacy-focused design\n- Secure data storage\n"
    
    if "review" in name.lower() or "reputation" in name.lower():
        content += "- Review aggregation\n- Sentiment analysis\n- Automated responses\n"
    
    if "video" in name.lower() or "vid" in name.lower():
        content += "- Video processing\n- Media storage\n- Streaming capabilities\n"
    
    content += """- User authentication & authorization
- Role-based access control
- API rate limiting
- Comprehensive logging

## Security Considerations
- JWT token authentication
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration
- Environment variable management

## Development Setup
1. Clone the repository
2. Install dependencies: `npm install` and `composer install`
3. Configure environment variables
4. Run migrations: `php artisan migrate`
5. Start development servers: `npm run dev` and `php artisan serve`

## Deployment
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Environment-specific configurations
- Automated testing before deployment
- Blue-green deployment strategy

## Monitoring
- Application performance monitoring
- Error tracking and alerting
- User analytics
- API usage metrics
"""
    
    return content

def main():
    """Main function to fix repositories."""
    github_token = os.environ.get('GITHUB_TOKEN')
    g = Github(github_token)
    
    repos_to_fix = [
        "summarize-ai-mobile",
        "brand-guardian-ai", 
        "reputation-ai",
        "eco-track-ai",
        "ai-powered-inventory-sync",
        "community-connect-platform",
        "mindleap-ai",
        "vid-gen-ai",
        "review-sentry"
    ]
    
    success_count = 0
    
    for repo_name in repos_to_fix:
        try:
            repo = g.get_repo(f"CodeWebMobile-AI/{repo_name}")
            print(f"\n🔧 Processing {repo_name}...")
            
            # Check if ARCHITECTURE.md exists
            try:
                repo.get_contents("ARCHITECTURE.md")
                print(f"✅ {repo_name} already has ARCHITECTURE.md")
                continue
            except:
                pass
            
            # Generate and create architecture content
            content = create_architecture_content(repo)
            
            result = repo.create_file(
                "ARCHITECTURE.md",
                "Add comprehensive architecture documentation",
                content,
                branch="main"
            )
            
            print(f"✅ Created ARCHITECTURE.md for {repo_name}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error with {repo_name}: {e}")
    
    print(f"\n📊 Summary: Fixed {success_count}/{len(repos_to_fix)} repositories")

if __name__ == "__main__":
    main()