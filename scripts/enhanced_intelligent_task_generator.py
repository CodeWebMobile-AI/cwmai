"""
Enhanced Intelligent Task Generator with Monetization Strategy Creation

This module extends the intelligent task generator to create comprehensive
monetization strategies and market analysis for new projects.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

from scripts.intelligent_task_generator import IntelligentTaskGenerator
from scripts.business_market_research_engine import BusinessMarketResearchEngine
from scripts.enhanced_project_opportunity import (
    EnhancedProjectOpportunity,
    ProjectOpportunityFactory
)
from scripts.existing_project_monetization_analyzer import ExistingProjectMonetizationAnalyzer


class EnhancedIntelligentTaskGenerator(IntelligentTaskGenerator):
    """Enhanced task generator that creates monetization strategies."""
    
    def __init__(self, ai_brain, charter_system, learning_system):
        """Initialize enhanced task generator."""
        super().__init__(ai_brain, charter_system, learning_system)
        
        # Add business research components
        self.business_research = BusinessMarketResearchEngine(ai_brain)
        self.monetization_analyzer = ExistingProjectMonetizationAnalyzer(ai_brain)
        self.template_path = Path("scripts/templates/monetization_strategy_template.md")
        
    async def _generate_new_project_task(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a NEW_PROJECT task with comprehensive monetization strategy.
        
        This overrides the parent method to add monetization capabilities.
        """
        self.logger.info("Generating enhanced NEW_PROJECT task with monetization...")
        
        try:
            # Get existing portfolio
            existing_portfolio = context.get('active_projects', {})
            
            # Discover business opportunities
            opportunities = await self.business_research.discover_business_opportunities(
                existing_portfolio,
                max_opportunities=3
            )
            
            if not opportunities:
                # Fallback to AI-generated ideas
                self.logger.info("No business opportunities found, using AI generation")
                return await self._generate_ai_project_with_monetization(context)
            
            # Select best opportunity
            best_opportunity = opportunities[0]  # Already sorted by score
            
            # Convert to enhanced project opportunity
            enhanced_opp = ProjectOpportunityFactory.from_business_opportunity(
                best_opportunity,
                architecture_type=self._select_architecture_type(best_opportunity)
            )
            
            # Generate complete project files content
            project_files = await self._generate_project_files(enhanced_opp)
            
            # Create task with all details
            task = {
                'id': f"new_project_{enhanced_opp.id}",
                'type': 'NEW_PROJECT',
                'title': f"Launch {enhanced_opp.title} - {enhanced_opp.tagline}",
                'description': self._create_detailed_description(enhanced_opp),
                'priority': enhanced_opp.get_priority(),
                'target_project': None,  # New project
                'details': {
                    'project_name': enhanced_opp.title,
                    'project_type': enhanced_opp.monetization_model.model_type,
                    'tech_stack': enhanced_opp.tech_stack,
                    'architecture_type': enhanced_opp.architecture_type,
                    'monetization_summary': enhanced_opp.get_monetization_summary(),
                    'revenue_projection_year_1': enhanced_opp.get_first_year_revenue_projection(),
                    'implementation_plan': project_files.get('implementation_plan', {})
                },
                'metadata': enhanced_opp.to_task_metadata(),
                'files_to_create': project_files,
                'estimated_hours': enhanced_opp.time_to_market_days * 8,
                'success_criteria': [
                    f"Project repository created with name '{enhanced_opp.title.lower().replace(' ', '-')}'",
                    "Complete architecture.md with technical design",
                    "Monetization strategy documented with revenue projections",
                    "Market analysis completed with competitor research",
                    "Initial codebase with authentication and billing setup",
                    f"Achieve ${enhanced_opp.monetization_model.target_arpu} ARPU within 3 months"
                ]
            }
            
            # Record task
            self.task_history.append({
                'task': task,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'opportunity_score': enhanced_opp.total_opportunity_score,
                'source': 'business_research'
            })
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced NEW_PROJECT task: {e}")
            # Fallback to parent implementation
            return await super()._generate_new_project_task(context)
    
    async def _generate_ai_project_with_monetization(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate AI-based project with monetization strategy."""
        # Use parent method to get basic idea
        basic_task = await super()._generate_new_project_task(context)
        
        if not basic_task:
            return None
        
        # Enhance with monetization
        project_name = basic_task.get('details', {}).get('project_name', 'NewProject')
        
        # Generate monetization model
        monetization_prompt = f"""
        Create a comprehensive monetization strategy for {project_name}:
        {basic_task.get('description', '')}
        
        Design:
        1. SaaS pricing tiers (Free, Pro, Enterprise)
        2. Target ARPU and unit economics
        3. Revenue streams and 24/7 automation
        4. Growth strategy and projections
        
        Return as structured JSON matching our monetization model format.
        """
        
        response = await self.ai_brain.execute_capability(
            'problem_analysis',
            {'prompt': monetization_prompt}
        )
        
        if response and response.get('status') == 'success':
            # Parse and add monetization to task
            monetization_data = self._parse_monetization_response(response.get('result', ''))
            
            # Add monetization files
            if 'files_to_create' not in basic_task:
                basic_task['files_to_create'] = {}
            
            basic_task['files_to_create']['monetization_strategy.md'] = \
                await self._generate_monetization_document(project_name, monetization_data)
            
            basic_task['files_to_create']['market_analysis.md'] = \
                await self._generate_market_analysis_document(project_name, basic_task)
        
        return basic_task
    
    async def _generate_project_files(self, opportunity: EnhancedProjectOpportunity) -> Dict[str, Any]:
        """Generate all project files including monetization documents."""
        files = {}
        
        # 1. Generate monetization strategy document
        files['monetization_strategy.md'] = await self._generate_monetization_document(
            opportunity.title,
            opportunity.monetization_model,
            opportunity
        )
        
        # 2. Generate market analysis document
        files['market_analysis.md'] = await self._generate_market_analysis_document(
            opportunity.title,
            opportunity
        )
        
        # 3. Generate technical architecture
        files['architecture.md'] = await self._generate_architecture_with_monetization(
            opportunity
        )
        
        # 4. Generate README with business context
        files['README.md'] = await self._generate_readme_with_business_context(
            opportunity
        )
        
        # 5. Implementation plan
        files['implementation_plan'] = {
            'phase_1': {
                'duration': '2 weeks',
                'goals': ['MVP with core value prop', 'Basic billing integration'],
                'deliverables': opportunity.initial_features[:3]
            },
            'phase_2': {
                'duration': '2 weeks', 
                'goals': ['Full payment flow', 'User onboarding optimization'],
                'deliverables': opportunity.initial_features[3:]
            },
            'phase_3': {
                'duration': '2 weeks',
                'goals': ['Launch and iterate', 'Growth features'],
                'deliverables': ['Analytics', 'Referral system', 'API']
            }
        }
        
        return files
    
    async def _generate_monetization_document(
        self, 
        project_name: str,
        monetization_model: Any,
        opportunity: Optional[EnhancedProjectOpportunity] = None
    ) -> str:
        """Generate complete monetization strategy document."""
        
        if not self.template_path.exists():
            return f"# {project_name} Monetization Strategy\n\nTemplate not found."
        
        template = self.template_path.read_text()
        
        # If we have a full opportunity object, use it
        if opportunity:
            replacements = {
                '{PROJECT_NAME}': project_name,
                '{TARGET_MRR}': f"{opportunity.revenue_projections.get(6, 10000):,.0f}",
                '{TIMEFRAME}': '6 months',
                '{REVENUE_MODEL}': opportunity.monetization_model.model_type,
                '{TARGET_AUDIENCE}': opportunity.target_market.get('primary', 'businesses'),
                '{PROBLEM_STATEMENT}': opportunity.problem_statement,
                
                # Value proposition
                '{CURRENT_STATE_PROBLEM}': opportunity.problem_statement,
                '{SOLUTION_DESCRIPTION}': opportunity.solution_approach,
                '{QUANTIFIED_BENEFIT}': f"${opportunity.monetization_model.target_arpu * 10}/month",
                '{VALUE_PER_MONTH}': f"{opportunity.monetization_model.target_arpu * 10}",
                '{PRICE_POINT}': str(opportunity.monetization_model.target_arpu),
                '{RATIO}': '10',
                
                # Pricing tiers
                '{FREE_LIMITS}': '10 operations/day',
                '{FREE_FEATURES}': 'Core features with limits',
                '{FREE_PURPOSE}': 'Prove value and convert to paid',
                '{FREE_TO_PAID_TARGET}': '2.5',
                
                # Financial projections
                '{ARPU}': str(opportunity.monetization_model.target_arpu),
                '{CAC_FINAL}': str(opportunity.monetization_model.target_cac),
                '{LTV}': str(opportunity.monetization_model.target_ltv),
                '{LTV_CAC_RATIO}': f"{opportunity.monetization_model.ltv_cac_ratio:.1f}",
                '{PAYBACK_MONTHS}': str(opportunity.monetization_model.payback_months),
                '{GROSS_MARGIN}': f"{opportunity.monetization_model.gross_margin_target * 100:.0f}",
                
                # Growth metrics
                '{MRR_GROWTH_TARGET}': '20',
                '{CHURN_TARGET}': '5',
                '{MARGIN_TARGET}': '80'
            }
            
            # Add monthly projections
            for month, mrr in opportunity.revenue_projections.items():
                replacements[f'{{M{month}_MRR}}'] = f"{mrr:,.0f}"
                # Estimate user counts
                free_users = int(mrr / opportunity.monetization_model.target_arpu * 20)
                paid_users = int(mrr / opportunity.monetization_model.target_arpu)
                replacements[f'{{M{month}_FREE}}'] = str(free_users)
                replacements[f'{{M{month}_PAID}}'] = str(paid_users)
                
                # Calculate growth rate
                if month > 1 and month - 1 in opportunity.revenue_projections:
                    prev_mrr = opportunity.revenue_projections[month - 1]
                    growth = ((mrr - prev_mrr) / prev_mrr * 100) if prev_mrr > 0 else 0
                    replacements[f'{{M{month}_GROWTH}}'] = f"{growth:.0f}"
        else:
            # Basic replacements for fallback
            replacements = {
                '{PROJECT_NAME}': project_name,
                '{TARGET_MRR}': '10000',
                '{TIMEFRAME}': '6 months'
            }
        
        # Replace all placeholders
        content = template
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, str(value))
        
        # Replace remaining placeholders with sensible defaults
        import re
        remaining = re.findall(r'\{[^}]+\}', content)
        defaults = {
            'URGENCY_RATING': '8',
            'SPECIFIC_PAIN_POINTS': 'Time wasted on manual processes',
            'TIER_1_NAME': 'Professional',
            'TIER_1_PRICE': '29',
            'TIER_2_NAME': 'Business', 
            'TIER_2_PRICE': '99',
            'ENTERPRISE_BASE': '499',
            'MONTHLY_MIX': '70',
            'ANNUAL_MIX': '30',
            'ANNUAL_DISCOUNT': '20',
            'LTV_MONTHS': '24',
            'BREAKEVEN_MONTHS': '8'
        }
        
        for placeholder in remaining:
            key = placeholder.strip('{}')
            if key in defaults:
                content = content.replace(placeholder, defaults[key])
            else:
                content = content.replace(placeholder, '[TO BE DETERMINED]')
        
        return content
    
    async def _generate_market_analysis_document(
        self,
        project_name: str,
        opportunity: Any
    ) -> str:
        """Generate market analysis document."""
        
        if hasattr(opportunity, 'market_validation'):
            market_data = opportunity.market_validation
            content = f"""# Market Analysis for {project_name}

Generated: {datetime.now(timezone.utc).isoformat()}

## Executive Summary

{project_name} targets a ${market_data.market_size['tam'] / 1000000:.0f}M total addressable market with a clear path to capturing ${market_data.market_size['som'] / 1000000:.1f}M in the first 3 years.

## Market Size

- **Total Addressable Market (TAM)**: ${market_data.market_size['tam']:,.0f}
- **Serviceable Addressable Market (SAM)**: ${market_data.market_size['sam']:,.0f}
- **Serviceable Obtainable Market (SOM)**: ${market_data.market_size['som']:,.0f}

## Target Market Segments

### Primary Segment
- **Description**: {opportunity.target_market.get('primary', 'Small to medium businesses')}
- **Size**: {opportunity.target_market.get('size', '1M+ potential customers')}
- **Characteristics**: Cost-conscious, seeking efficiency, open to new solutions
- **Willingness to Pay**: ${market_data.willingness_to_pay['average']}/month average

## Competitive Landscape

### Direct Competitors
1. **Legacy Solutions**
   - Expensive and complex
   - Poor user experience
   - Limited automation

2. **Emerging Players**  
   - Feature-limited
   - Focusing on different segments
   - Opportunity for differentiation

### Competitive Advantages
"""
            if hasattr(opportunity, 'competitor_advantages'):
                for advantage in opportunity.competitor_advantages:
                    content += f"- {advantage}\n"
            else:
                content += """- AI-powered automation
- Superior user experience
- Transparent pricing
- Quick time-to-value
"""

            content += f"""
## Market Trends

1. **Digital Transformation Acceleration**
   - Businesses seeking cloud-based solutions
   - Demand for automation growing 30% YoY

2. **Shift to Subscription Models**
   - Predictable costs preferred
   - Lower barrier to entry

3. **AI Adoption**
   - Businesses want AI benefits without complexity
   - Opportunity to be AI-first in the space

## Customer Acquisition Strategy

### Channels
"""
            for channel in market_data.acquisition_channels:
                content += f"- **{channel['channel']}**: ${channel['cac']} CAC\n"
            
            content += f"""
## Risk Analysis

### Market Risks
- New entrants with significant funding
- Platform players adding competing features
- Economic downturn affecting budgets

### Mitigation Strategies  
- Build strong customer relationships
- Focus on unique value proposition
- Maintain pricing flexibility

## 3-Year Growth Projection

- Year 1: Capture 0.1% of SOM (${market_data.market_size['som'] * 0.001:,.0f})
- Year 2: Expand to 0.5% of SOM (${market_data.market_size['som'] * 0.005:,.0f})
- Year 3: Reach 1% of SOM (${market_data.market_size['som'] * 0.01:,.0f})

## Conclusion

The market opportunity for {project_name} is validated by strong demand, clear pain points, and favorable market dynamics. With proper execution, the path to ${opportunity.revenue_projections.get(12, 100000) * 3:,.0f} ARR by Year 3 is achievable.
"""
        else:
            # Simpler version for basic opportunities
            content = f"""# Market Analysis for {project_name}

Generated: {datetime.now(timezone.utc).isoformat()}

## Overview

Market analysis indicates strong potential for {project_name} based on identified pain points and willingness to pay.

## Target Market
- Primary audience: {getattr(opportunity, 'target_market', 'Businesses seeking efficiency')}
- Market size: Large and growing
- Key pain points: Time waste, manual processes, lack of insights

## Competitive Landscape
- Existing solutions are complex and expensive
- Opportunity for simpler, more focused solution
- AI-powered features provide differentiation

## Growth Strategy
- Start with early adopters
- Expand through word-of-mouth
- Scale with content marketing and partnerships
"""
        
        return content
    
    async def _generate_architecture_with_monetization(
        self,
        opportunity: EnhancedProjectOpportunity
    ) -> str:
        """Generate architecture.md with monetization components."""
        
        content = f"""# {opportunity.title} Technical Architecture

## Overview

{opportunity.title} is built with a focus on scalability, maintainability, and revenue generation from day one.

## Technology Stack

### Backend
"""
        
        if 'laravel' in opportunity.tech_stack:
            content += """- **Framework**: Laravel 10.x
- **Language**: PHP 8.2+
- **Database**: MySQL 8.0 / PostgreSQL 14
- **Cache**: Redis 7.0
- **Queue**: Redis Queue
"""
        elif 'django' in opportunity.tech_stack:
            content += """- **Framework**: Django 4.2
- **Language**: Python 3.11+
- **Database**: PostgreSQL 14
- **Cache**: Redis 7.0
- **Queue**: Celery with Redis
"""
        else:
            content += """- **Framework**: Next.js 14
- **Language**: TypeScript
- **Database**: PostgreSQL 14
- **ORM**: Prisma
- **Cache**: Redis 7.0
"""
        
        content += f"""
### Frontend
- **Framework**: React 18 with TypeScript
- **State Management**: Zustand / Redux Toolkit
- **UI Library**: Tailwind CSS + Headless UI
- **Build Tool**: Vite

### Infrastructure
- **Hosting**: AWS / DigitalOcean
- **CDN**: CloudFlare
- **Monitoring**: Sentry, LogRocket
- **Analytics**: Mixpanel, Google Analytics

## Core Architecture Components

### 1. Authentication & Authorization
- JWT-based authentication
- Role-based access control (Free, Pro, Enterprise)
- OAuth integration (Google, GitHub)
- Session management with Redis

### 2. Billing & Subscription Management
```
├── app/Services/Billing/
│   ├── StripeService.php         # Stripe integration
│   ├── SubscriptionManager.php   # Subscription lifecycle
│   ├── UsageTracker.php         # Usage-based billing
│   └── InvoiceGenerator.php     # Invoice generation
```

### 3. Core Business Logic
- **{opportunity.title} Engine**: Main value delivery
- **Analytics Service**: Track user behavior and value delivered
- **Notification System**: Email, in-app, webhooks
- **API Layer**: RESTful + GraphQL for integrations

### 4. Revenue Optimization Components

#### Usage Tracking
- Track feature usage per user/account
- Enforce plan limits in real-time
- Trigger upgrade prompts at limits

#### Automated Billing
- Stripe/Paddle integration
- Dunning management for failed payments
- Automatic invoice generation
- Revenue recognition compliance

#### Growth Features
- Referral system with rewards
- In-app upgrade prompts
- Usage-based upsells
- Annual plan promotions

## Database Schema (Key Tables)

```sql
-- Users and Authentication
users (id, email, password_hash, created_at, trial_ends_at)
teams (id, name, owner_id, plan_id, created_at)
team_members (team_id, user_id, role, joined_at)

-- Subscriptions and Billing
plans (id, name, price_monthly, price_annual, features_json)
subscriptions (id, team_id, plan_id, status, current_period_end)
usage_records (id, team_id, feature, count, recorded_at)
invoices (id, subscription_id, amount, status, due_date)

-- Core Features
{opportunity.title.lower()}_entities (id, team_id, data, created_at)
{opportunity.title.lower()}_activities (id, entity_id, action, metadata)
```

## Security Considerations

1. **Data Protection**
   - Encryption at rest and in transit
   - GDPR compliance tools
   - Regular security audits

2. **API Security**
   - Rate limiting by plan tier
   - API key management
   - Webhook signature validation

3. **Payment Security**
   - PCI compliance via Stripe
   - No credit card storage
   - Secure webhook endpoints

## Scalability Plan

### Phase 1 (0-1K customers)
- Single server deployment
- Basic monitoring

### Phase 2 (1K-10K customers)  
- Load balancer + 2-3 app servers
- Read replicas for database
- Enhanced caching

### Phase 3 (10K+ customers)
- Microservices architecture
- Kubernetes deployment
- Global CDN distribution

## Monitoring & Analytics

### Technical Monitoring
- Uptime monitoring (99.9% SLA)
- Performance metrics
- Error tracking with Sentry

### Business Metrics
- MRR tracking
- Churn analysis
- Feature usage analytics
- Conversion funnel optimization

## Development Workflow

1. **Local Development**
   ```bash
   docker-compose up -d
   npm run dev
   ```

2. **Testing**
   - Unit tests for business logic
   - Integration tests for billing
   - E2E tests for critical paths

3. **Deployment**
   - CI/CD with GitHub Actions
   - Blue-green deployments
   - Automatic rollback on errors
"""
        
        return content
    
    async def _generate_readme_with_business_context(
        self,
        opportunity: EnhancedProjectOpportunity
    ) -> str:
        """Generate README with business context."""
        
        content = f"""# {opportunity.title}

> {opportunity.tagline}

## 🚀 Overview

{opportunity.description}

## 💰 Business Model

- **Revenue Model**: {opportunity.monetization_model.model_type.upper()}
- **Target ARPU**: ${opportunity.monetization_model.target_arpu}
- **Target Market**: {opportunity.target_market.get('primary', 'Businesses')}
- **Year 1 Revenue Target**: ${opportunity.get_first_year_revenue_projection():,.0f}

## 🎯 Problem We Solve

{opportunity.problem_statement}

## 💡 Our Solution

{opportunity.solution_approach}

## 🛠 Tech Stack

- **Backend**: {', '.join([t for t in opportunity.tech_stack if t in ['laravel', 'django', 'express', 'fastapi']])}
- **Frontend**: {', '.join([t for t in opportunity.tech_stack if t in ['react', 'vue', 'angular', 'nextjs']])}
- **Database**: {', '.join([t for t in opportunity.tech_stack if t in ['mysql', 'postgresql', 'mongodb', 'redis']])}

## 📦 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
"""
        
        if 'laravel' in opportunity.tech_stack:
            content += "- PHP 8.2+\n- Composer\n"
        elif 'django' in opportunity.tech_stack:
            content += "- Python 3.11+\n- pip\n"
            
        content += f"""
### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/CodeWebMobile-AI/{opportunity.title.lower().replace(' ', '-')}.git
   cd {opportunity.title.lower().replace(' ', '-')}
   ```

2. Copy environment variables
   ```bash
   cp .env.example .env
   ```

3. Start with Docker
   ```bash
   docker-compose up -d
   ```

4. Install dependencies
   ```bash
   npm install
"""
        
        if 'laravel' in opportunity.tech_stack:
            content += "   composer install\n"
        elif 'django' in opportunity.tech_stack:
            content += "   pip install -r requirements.txt\n"
            
        content += """   ```

5. Run migrations
   ```bash
"""
        
        if 'laravel' in opportunity.tech_stack:
            content += "   php artisan migrate\n"
        elif 'django' in opportunity.tech_stack:
            content += "   python manage.py migrate\n"
        else:
            content += "   npx prisma migrate dev\n"
            
        content += """   ```

6. Start development server
   ```bash
   npm run dev
   ```

## 📊 Key Features

"""
        
        for feature in opportunity.initial_features[:5]:
            content += f"- ✅ {feature}\n"
            
        content += f"""
## 💳 Pricing Tiers

"""
        
        for tier in opportunity.monetization_model.pricing_tiers[:3]:
            content += f"### {tier.get('name', 'Tier')}\n"
            if tier.get('price', 0) == 0:
                content += "- **Price**: Free\n"
            else:
                content += f"- **Price**: ${tier.get('price', 0)}/month\n"
            content += f"- **Features**: {', '.join(tier.get('features', ['Core features'])[:3])}\n\n"
            
        content += """
## 🚦 Project Status

- [x] Market research completed
- [x] Monetization strategy defined
- [ ] MVP development
- [ ] Beta testing
- [ ] Public launch

## 📈 Success Metrics

- **Target Launch**: {opportunity.time_to_market_days} days
- **Time to Revenue**: {opportunity.time_to_revenue_days} days  
- **6-Month MRR Target**: ${opportunity.revenue_projections.get(6, 10000):,.0f}
- **LTV:CAC Ratio Target**: {opportunity.monetization_model.ltv_cac_ratio:.1f}:1

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

This project is proprietary and confidential.

---

Built with ❤️ by the CodeWebMobile-AI team
"""
        
        return content
    
    def _select_architecture_type(self, opportunity: Any) -> str:
        """Select appropriate architecture type based on project needs."""
        
        # Simple heuristic based on project type
        if hasattr(opportunity, 'monetization_model'):
            model_type = opportunity.monetization_model.model_type
            
            if model_type == 'marketplace':
                return 'laravel-react'  # Good for complex business logic
            elif model_type == 'saas':
                return 'laravel-react'  # Reliable for SaaS
            elif 'real-time' in str(opportunity.solution_description).lower():
                return 'nextjs'  # Better for real-time features
            else:
                return 'laravel-react'  # Default
        
        return 'laravel-react'
    
    def _create_detailed_description(self, opportunity: EnhancedProjectOpportunity) -> str:
        """Create detailed task description."""
        
        return f"""## 🚀 Project Overview

**{opportunity.title}** - {opportunity.tagline}

### Problem
{opportunity.problem_statement}

### Solution  
{opportunity.solution_approach}

### Unique Value Proposition
{opportunity.unique_value_proposition}

### Target Market
- **Primary**: {opportunity.target_market.get('primary', 'Businesses')}
- **Market Size**: ${opportunity.market_validation.market_size['som'] / 1000000:.1f}M addressable market

### Monetization Strategy
{opportunity.get_monetization_summary()}

### Revenue Projections
- Month 1: ${opportunity.revenue_projections.get(1, 0):,.0f} MRR
- Month 3: ${opportunity.revenue_projections.get(3, 0):,.0f} MRR  
- Month 6: ${opportunity.revenue_projections.get(6, 0):,.0f} MRR
- Month 12: ${opportunity.revenue_projections.get(12, 0):,.0f} MRR

### Implementation Details
- **Tech Stack**: {', '.join(opportunity.tech_stack)}
- **Architecture**: {opportunity.architecture_type}
- **Complexity**: {opportunity.estimated_complexity}
- **Time to Market**: {opportunity.time_to_market_days} days
- **Time to Revenue**: {opportunity.time_to_revenue_days} days

### Success Criteria
1. Launch MVP within {opportunity.time_to_market_days} days
2. Achieve first paying customer within {opportunity.time_to_revenue_days} days
3. Reach ${opportunity.monetization_model.target_arpu} ARPU
4. Maintain {opportunity.monetization_model.ltv_cac_ratio:.1f}:1 LTV:CAC ratio
5. Generate ${opportunity.revenue_projections.get(6, 10000):,.0f} MRR by month 6

### Key Risks & Mitigation
- **Competition**: Differentiate through {opportunity.unique_value_proposition}
- **Technical**: Use proven {opportunity.architecture_type} architecture
- **Market**: Validated through research scoring {opportunity.market_validation.validation_score:.2f}
"""
    
    def _parse_monetization_response(self, response_text: str) -> Dict[str, Any]:
        """Parse monetization data from AI response."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.error(f"Error parsing monetization response: {e}")
        
        # Return default structure
        return {
            'model_type': 'saas',
            'pricing_tiers': [
                {'name': 'Free', 'price': 0},
                {'name': 'Pro', 'price': 29},
                {'name': 'Enterprise', 'price': 'custom'}
            ],
            'target_arpu': 50,
            'target_cac': 100,
            'target_ltv': 1200
        }