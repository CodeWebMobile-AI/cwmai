# System Architecture - [Project Name]

## Executive Summary

**Project**: [Name] - [One-line value proposition]  
**Problem Domain**: [Specific problem being solved]  
**Solution Type**: AI-powered [category] platform  
**Architecture Pattern**: Inertia.js Monolith with Event-Driven AI Services  
**Deployment Target**: Local Development → Cloud Production

## Table of Contents
1. [System Context](#system-context)
2. [Building Blocks](#building-blocks)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Data Architecture](#data-architecture)
5. [AI Integration](#ai-integration)
6. [Security & Compliance](#security-compliance)
7. [Performance & Scaling](#performance-scaling)
8. [Development Guidelines](#development-guidelines)

## System Context

### Problem Analysis
```yaml
Problem: [Detailed problem description]
Current Solutions: [Existing solutions and their limitations]
Our Approach: [How we solve it differently]
Key Innovation: [What makes our solution unique]
```

### User Journey Map
```
1. Discovery → User finds platform through [channels]
2. Onboarding → User signs up and completes [setup steps]
3. First Value → User experiences [key feature] within [timeframe]
4. Engagement → User regularly uses [core features]
5. Retention → User stays because of [retention drivers]
```

## Building Blocks

### Laravel React Starter Kit Foundation
```
What We Inherit:
├── Authentication System (complete)
│   ├── Login/Register/Password Reset
│   ├── Email Verification
│   └── Profile Management
├── UI Components (shadcn/ui)
│   ├── All form elements
│   ├── Layout components
│   └── Theming system
├── Inertia.js Architecture
│   ├── Server-side routing
│   ├── Page components
│   └── Shared data system
└── Development Setup
    ├── TypeScript configuration
    ├── Tailwind CSS
    └── Vite bundling
```

### Domain-Specific Extensions

#### 1. Models & Relationships
```php
// app/Models/[DomainModel].php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class [DomainModel] extends Model
{
    protected $fillable = [
        // List all fillable fields with types in comments
        'field_name', // string: Description of field
    ];

    protected $casts = [
        'field_name' => 'array', // Cast types for special fields
    ];

    // Relationships
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    // Business Logic Methods
    public function calculateValue(): float
    {
        // Implementation details
    }
}
```

#### 2. Page Component Structure
```typescript
// resources/js/pages/[Portal]/[Feature]/[Action].tsx
import { Head } from '@inertiajs/react';
import { [Feature]Layout } from '@/layouts/[Portal]/[Feature]Layout';
import { [Component] } from '@/components/[domain]/[Component]';

interface Props {
    data: {
        // Type all props from controller
    };
}

export default function [PageName]({ data }: Props) {
    // State management
    const [state, setState] = useState();
    
    // Event handlers
    const handle[Action] = async () => {
        // Implementation
    };
    
    return (
        <[Feature]Layout>
            <Head title="[Page Title]" />
            {/* Component structure */}
        </[Feature]Layout>
    );
}
```

#### 3. Controller Implementation
```php
// app/Http/Controllers/[Portal]/[Feature]Controller.php
namespace App\Http\Controllers\[Portal];

use App\Http\Controllers\Controller;
use App\Models\[Model];
use Illuminate\Http\Request;
use Inertia\Inertia;

class [Feature]Controller extends Controller
{
    public function index(Request $request)
    {
        // Data preparation with eager loading
        $data = [Model]::with(['relation1', 'relation2'])
            ->where('user_id', $request->user()->id)
            ->paginate(15);
            
        return Inertia::render('[Portal]/[Feature]/Index', [
            'data' => $data,
            'filters' => $request->only(['search', 'status']),
            'stats' => $this->calculateStats($request->user()),
        ]);
    }
    
    public function store([Feature]Request $request)
    {
        // Validation through form request
        // Business logic execution
        // Return redirect with flash message
    }
}
```

### Portal Separation Strategy

#### File Organization
```
app/Http/Controllers/
├── Admin/
│   ├── DashboardController.php      # Admin overview
│   ├── UsersController.php          # User management
│   ├── AnalyticsController.php      # Platform analytics
│   └── SettingsController.php       # System settings
└── User/
    ├── DashboardController.php      # User dashboard
    ├── [Feature]Controller.php      # Core features
    └── PreferencesController.php    # User settings

resources/js/pages/
├── Admin/
│   ├── Dashboard.tsx               # Admin landing
│   ├── Users/
│   │   ├── Index.tsx              # User list
│   │   └── Show.tsx               # User details
│   └── Analytics/
│       └── Overview.tsx           # Analytics dashboard
└── User/
    ├── Dashboard.tsx              # User landing
    ├── [Feature]/
    │   ├── Index.tsx             # Feature list
    │   ├── Create.tsx            # Create new
    │   └── Edit.tsx              # Edit existing
    └── Settings/
        └── Profile.tsx           # User preferences
```

#### Middleware Configuration
```php
// app/Http/Middleware/EnsureUserIsAdmin.php
namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;

class EnsureUserIsAdmin
{
    public function handle(Request $request, Closure $next)
    {
        if (!$request->user()?->is_admin) {
            return redirect()->route('dashboard')
                ->with('error', 'Unauthorized access.');
        }
        
        return $next($request);
    }
}

// app/Http/Kernel.php - Register middleware
protected $middlewareAliases = [
    // ... existing middleware
    'admin' => \App\Http\Middleware\EnsureUserIsAdmin::class,
];
```

#### Route Organization
```php
// routes/web.php
use App\Http\Controllers\Admin;
use App\Http\Controllers\User;

// Public routes
Route::get('/', function () {
    return Inertia::render('Welcome');
});

// User authenticated routes
Route::middleware(['auth', 'verified'])->group(function () {
    Route::get('/dashboard', [User\DashboardController::class, 'index'])
        ->name('dashboard');
    
    Route::prefix('app')->name('app.')->group(function () {
        Route::resource('[feature]', User\[Feature]Controller::class);
        // Add more user routes
    });
});

// Admin authenticated routes
Route::middleware(['auth', 'verified', 'admin'])->prefix('admin')->name('admin.')->group(function () {
    Route::get('/dashboard', [Admin\DashboardController::class, 'index'])
        ->name('dashboard');
    
    Route::resource('users', Admin\UsersController::class);
    Route::get('analytics', [Admin\AnalyticsController::class, 'index'])
        ->name('analytics');
});
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
```yaml
Tasks:
  - Set up Laravel React starter kit
  - Configure local development environment
  - Implement domain models and migrations
  - Create base layouts for admin/user portals
  - Set up CI/CD pipeline

Deliverables:
  - Working authentication system
  - Database schema implemented
  - Basic portal separation
  - Development environment ready
```

### Phase 2: Core Features (Week 2-3)
```yaml
Tasks:
  - Implement primary user workflows
  - Build admin management interfaces
  - Integrate AI services
  - Create real-time features
  - Implement data validation

Deliverables:
  - 3-5 core features working
  - Admin can manage users/content
  - AI integration operational
  - Real-time updates working
```

### Phase 3: Enhancement (Week 4)
```yaml
Tasks:
  - Add analytics and reporting
  - Implement advanced features
  - Optimize performance
  - Add comprehensive testing
  - Polish UI/UX

Deliverables:
  - Analytics dashboard
  - Advanced features complete
  - Performance optimized
  - Test coverage > 80%
```

## Data Architecture

### Database Schema
```sql
-- Core Tables
CREATE TABLE users (
    id BIGINT UNSIGNED PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    subscription_tier ENUM('free', 'basic', 'pro', 'enterprise') DEFAULT 'free',
    trial_ends_at TIMESTAMP NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    INDEX idx_subscription_tier (subscription_tier),
    INDEX idx_trial_ends (trial_ends_at)
);

-- Domain-Specific Tables
CREATE TABLE [domain_entities] (
    id BIGINT UNSIGNED PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    [domain_specific] VARCHAR(255) NOT NULL,
    status ENUM('active', 'pending', 'completed') DEFAULT 'pending',
    metadata JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_status (user_id, status),
    INDEX idx_created (created_at)
);

-- Analytics Tables
CREATE TABLE user_activities (
    id BIGINT UNSIGNED PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    action VARCHAR(100) NOT NULL,
    metadata JSON,
    created_at TIMESTAMP,
    INDEX idx_user_action (user_id, action),
    INDEX idx_created (created_at)
);
```

### Caching Strategy
```php
// config/cache.php additions
'cache_keys' => [
    'user_stats' => 'user:{id}:stats',              // TTL: 5 minutes
    'dashboard_data' => 'user:{id}:dashboard',      // TTL: 1 minute
    'analytics' => 'analytics:{type}:{period}',     // TTL: 1 hour
    'ai_results' => 'ai:{model}:{hash}',           // TTL: 24 hours
];

// Usage in services
public function getUserStats(User $user): array
{
    return Cache::remember(
        str_replace('{id}', $user->id, config('cache.cache_keys.user_stats')),
        300, // 5 minutes
        fn() => $this->calculateUserStats($user)
    );
}
```

### Queue Jobs
```php
// app/Jobs/Process[Feature].php
namespace App\Jobs;

use App\Models\[Model];
use App\Services\AIService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

class Process[Feature] implements ShouldQueue
{
    use InteractsWithQueue, Queueable, SerializesModels;
    
    public function __construct(
        public [Model] $model
    ) {}
    
    public function handle(AIService $aiService): void
    {
        // Process with AI
        $result = $aiService->analyze($this->model);
        
        // Update model
        $this->model->update([
            'ai_result' => $result,
            'processed_at' => now(),
        ]);
        
        // Broadcast update
        broadcast(new [Feature]Processed($this->model));
    }
}
```

## AI Integration

### Service Architecture
```php
// app/Services/AI/AIServiceInterface.php
namespace App\Services\AI;

interface AIServiceInterface
{
    public function analyze(array $data): AIResult;
    public function generateContent(string $prompt, array $context = []): string;
    public function extractInsights(string $content): array;
}

// app/Services/AI/OpenAIService.php
namespace App\Services\AI;

use OpenAI\Client;

class OpenAIService implements AIServiceInterface
{
    private Client $client;
    private MetricsCollector $metrics;
    
    public function __construct()
    {
        $this->client = OpenAI::client(config('services.openai.key'));
        $this->metrics = app(MetricsCollector::class);
    }
    
    public function analyze(array $data): AIResult
    {
        $start = microtime(true);
        
        try {
            $response = $this->client->chat()->create([
                'model' => 'gpt-4',
                'messages' => [
                    ['role' => 'system', 'content' => $this->getSystemPrompt()],
                    ['role' => 'user', 'content' => json_encode($data)],
                ],
                'temperature' => 0.7,
                'max_tokens' => 1000,
            ]);
            
            $result = new AIResult(
                content: $response->choices[0]->message->content,
                confidence: $this->calculateConfidence($response),
                metadata: ['model' => 'gpt-4', 'tokens' => $response->usage->totalTokens]
            );
            
            $this->metrics->record('ai_analysis', microtime(true) - $start);
            
            return $result;
            
        } catch (\Exception $e) {
            $this->metrics->recordError('ai_analysis', $e);
            throw new AIServiceException('Analysis failed', 0, $e);
        }
    }
}
```

### AI Prompt Templates
```php
// app/Services/AI/Prompts/[Feature]Prompts.php
namespace App\Services\AI\Prompts;

class [Feature]Prompts
{
    public static function analysisPrompt(): string
    {
        return <<<PROMPT
        You are an AI assistant specialized in [domain].
        
        Your task is to analyze the provided data and:
        1. Identify key patterns and insights
        2. Provide actionable recommendations
        3. Highlight potential issues or opportunities
        
        Format your response as JSON with the following structure:
        {
            "summary": "Brief overview",
            "insights": ["insight1", "insight2"],
            "recommendations": ["rec1", "rec2"],
            "risk_level": "low|medium|high"
        }
        PROMPT;
    }
}
```

## Security & Compliance

### Authentication & Authorization
```php
// app/Policies/[Model]Policy.php
namespace App\Policies;

use App\Models\User;
use App\Models\[Model];

class [Model]Policy
{
    public function viewAny(User $user): bool
    {
        return true; // All authenticated users can view list
    }
    
    public function view(User $user, [Model] $model): bool
    {
        return $user->id === $model->user_id || $user->is_admin;
    }
    
    public function create(User $user): bool
    {
        // Check subscription limits
        return $user->canCreate[Model]();
    }
    
    public function update(User $user, [Model] $model): bool
    {
        return $user->id === $model->user_id;
    }
    
    public function delete(User $user, [Model] $model): bool
    {
        return $user->id === $model->user_id || $user->is_admin;
    }
}
```

### Data Protection
```yaml
Encryption:
  - At Rest: MySQL encryption enabled
  - In Transit: HTTPS enforced
  - Sensitive Fields: Laravel encryption for PII

Access Control:
  - Row-level security via policies
  - API rate limiting per user tier
  - Session timeout after 2 hours

Compliance:
  - GDPR: Data export/deletion features
  - Audit logging for all actions
  - Privacy policy acceptance tracking
```

## Performance & Scaling

### Performance Targets
```yaml
Response Times:
  - Page Load: < 2 seconds
  - API Response: < 200ms (p95)
  - AI Processing: < 5 seconds
  
Capacity:
  - Concurrent Users: 1,000 (initial)
  - Requests/Second: 100
  - Data Storage: 100GB (Year 1)
  
Availability:
  - Uptime Target: 99.9%
  - Maintenance Window: Sunday 2-4 AM UTC
```

### Optimization Strategies
```php
// Eager Loading
$data = Model::with(['relation1', 'relation2'])
    ->select(['id', 'name', 'status']) // Only needed fields
    ->where('active', true)
    ->paginate(20);

// Database Indexing
Schema::table('[table]', function (Blueprint $table) {
    $table->index(['user_id', 'status']);
    $table->index('created_at');
});

// Query Optimization
DB::table('[table]')
    ->select(DB::raw('DATE(created_at) as date'), DB::raw('COUNT(*) as count'))
    ->where('created_at', '>=', now()->subDays(30))
    ->groupBy('date')
    ->get();
```

## Development Guidelines

### Code Standards
```yaml
PHP:
  - PSR-12 coding standard
  - Type declarations required
  - Strict types enabled
  - PHPDoc for public methods

TypeScript:
  - ESLint configuration
  - Strict mode enabled
  - Interfaces for all props
  - Consistent naming conventions

Testing:
  - Unit tests for services
  - Feature tests for controllers
  - Component tests for React
  - E2E tests for critical paths
```

### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/[feature-name]
git add .
git commit -m "feat: implement [feature]"
git push origin feature/[feature-name]

# Commit message format
feat: add new feature
fix: resolve bug
docs: update documentation
refactor: improve code structure
test: add tests
chore: update dependencies
```

### Local Development Commands
```bash
# Start development
composer install
npm install
php artisan migrate:fresh --seed
npm run dev
php artisan serve

# Run tests
php artisan test
npm run test

# Code quality
./vendor/bin/phpcs
npm run lint

# Build for production
npm run build
php artisan optimize
```

## Monitoring & Observability

### Health Checks
```php
// routes/api.php
Route::get('/health', function () {
    return response()->json([
        'status' => 'healthy',
        'timestamp' => now(),
        'checks' => [
            'database' => DB::connection()->getPdo() ? 'connected' : 'disconnected',
            'cache' => Cache::has('health_check') ? 'working' : 'not working',
            'queue' => Queue::size() < 1000 ? 'healthy' : 'backlogged',
        ],
    ]);
});
```

### Logging Strategy
```php
// Strategic logging points
Log::channel('user_activity')->info('User action', [
    'user_id' => $user->id,
    'action' => 'feature_used',
    'feature' => '[feature_name]',
    'metadata' => $additionalData,
]);

Log::channel('ai_operations')->info('AI processing', [
    'operation' => 'analysis',
    'duration' => $duration,
    'tokens_used' => $tokens,
    'cost' => $estimatedCost,
]);
```

## Future Roadmap

### Next Features
1. **Mobile App**: React Native companion app
2. **API Public**: RESTful API for integrations
3. **Webhooks**: Event notifications
4. **Advanced AI**: Custom model training
5. **Multi-tenancy**: White-label solution

### Technical Debt
- Refactor [legacy component] by Q2
- Migrate to [new service] by Q3
- Improve test coverage to 90%
- Implement GraphQL gateway