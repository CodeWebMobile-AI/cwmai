# System Architecture - ReviewInsight AI

## Executive Summary

**Project**: ReviewInsight AI - Transform customer feedback into business growth  
**Problem Domain**: Small businesses struggle to manage and respond to online reviews across multiple platforms  
**Solution Type**: AI-powered review management and reputation enhancement platform  
**Architecture Pattern**: Inertia.js Monolith with Event-Driven AI Services  
**Deployment Target**: Local Development → AWS/DigitalOcean Production

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
Problem: Small businesses receive reviews on 50+ platforms but lack time and expertise to manage them effectively
Current Solutions: Manual monitoring, expensive agencies, or ignoring reviews entirely
Our Approach: AI-powered centralized dashboard with automated response generation and sentiment analysis
Key Innovation: GPT-4 trained on industry-specific response patterns with human-in-the-loop approval
```

### User Journey Map
```
1. Discovery → Business owner searches "review management software" or gets referral
2. Onboarding → Connect Google, Yelp, Facebook, TripAdvisor accounts (OAuth)
3. First Value → See all reviews in one dashboard within 2 minutes
4. Engagement → Daily AI-generated response suggestions, weekly reputation reports
5. Retention → Improved ratings, time saved, customer insights drive continued usage
```

## Building Blocks

### Laravel React Starter Kit Foundation
```
What We Inherit:
├── Authentication System (complete)
│   ├── Login/Register/Password Reset ✓
│   ├── Email Verification ✓
│   └── Profile Management ✓
├── UI Components (shadcn/ui)
│   ├── All form elements ✓
│   ├── Layout components ✓
│   └── Theming system ✓
├── Inertia.js Architecture
│   ├── Server-side routing ✓
│   ├── Page components ✓
│   └── Shared data system ✓
└── Development Setup
    ├── TypeScript configuration ✓
    ├── Tailwind CSS ✓
    └── Vite bundling ✓
```

### Domain-Specific Extensions

#### 1. Models & Relationships
```php
// app/Models/Business.php
namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Business extends Model
{
    protected $fillable = [
        'user_id',           // bigint: Owner of the business
        'name',              // string: Business name
        'industry',          // string: Restaurant, Hotel, Retail, etc.
        'description',       // text: Business description
        'website',           // string: Business website
        'phone',             // string: Contact phone
        'address',           // json: Full address object
        'timezone',          // string: Business timezone
        'settings',          // json: Notification preferences, AI settings
    ];

    protected $casts = [
        'address' => 'array',
        'settings' => 'array',
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function platforms(): HasMany
    {
        return $this->hasMany(Platform::class);
    }

    public function reviews(): HasMany
    {
        return $this->hasMany(Review::class);
    }

    public function responses(): HasMany
    {
        return $this->hasMany(Response::class);
    }

    public function calculateReputationScore(): float
    {
        $avgRating = $this->reviews()->avg('rating');
        $responseRate = $this->responses()->count() / max($this->reviews()->count(), 1);
        $recentTrend = $this->calculateRecentTrend();
        
        return ($avgRating * 0.5) + ($responseRate * 0.3) + ($recentTrend * 0.2);
    }
}
```

#### 2. Page Component Structure
```typescript
// resources/js/pages/User/Reviews/Index.tsx
import { Head } from '@inertiajs/react';
import { ReviewsLayout } from '@/layouts/User/ReviewsLayout';
import { ReviewCard } from '@/components/reviews/ReviewCard';
import { ReviewFilters } from '@/components/reviews/ReviewFilters';
import { AIResponseModal } from '@/components/reviews/AIResponseModal';
import { useState } from 'react';

interface Props {
    reviews: {
        data: Review[];
        meta: PaginationMeta;
    };
    filters: ReviewFilters;
    stats: {
        total: number;
        unresponded: number;
        averageRating: number;
        trend: 'up' | 'down' | 'stable';
    };
}

export default function ReviewsIndex({ reviews, filters, stats }: Props) {
    const [selectedReview, setSelectedReview] = useState<Review | null>(null);
    const [showAIModal, setShowAIModal] = useState(false);
    
    const handleGenerateResponse = async (review: Review) => {
        setSelectedReview(review);
        setShowAIModal(true);
    };
    
    return (
        <ReviewsLayout>
            <Head title="Reviews Dashboard" />
            
            <div className="space-y-6">
                {/* Stats Overview */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <StatCard title="Total Reviews" value={stats.total} />
                    <StatCard title="Need Response" value={stats.unresponded} alert />
                    <StatCard title="Average Rating" value={stats.averageRating.toFixed(1)} />
                    <StatCard title="Trend" value={stats.trend} trend />
                </div>
                
                {/* Filters */}
                <ReviewFilters defaultFilters={filters} />
                
                {/* Reviews List */}
                <div className="space-y-4">
                    {reviews.data.map((review) => (
                        <ReviewCard
                            key={review.id}
                            review={review}
                            onGenerateResponse={() => handleGenerateResponse(review)}
                        />
                    ))}
                </div>
                
                {/* Pagination */}
                <Pagination meta={reviews.meta} />
            </div>
            
            {/* AI Response Modal */}
            <AIResponseModal
                open={showAIModal}
                onClose={() => setShowAIModal(false)}
                review={selectedReview}
            />
        </ReviewsLayout>
    );
}
```

#### 3. Controller Implementation
```php
// app/Http/Controllers/User/ReviewController.php
namespace App\Http\Controllers\User;

use App\Http\Controllers\Controller;
use App\Models\Review;
use App\Services\AI\ReviewResponseGenerator;
use Illuminate\Http\Request;
use Inertia\Inertia;

class ReviewController extends Controller
{
    public function index(Request $request)
    {
        $business = $request->user()->businesses()->firstOrFail();
        
        $reviews = Review::with(['platform', 'response'])
            ->where('business_id', $business->id)
            ->when($request->platform, fn($q) => $q->where('platform_id', $request->platform))
            ->when($request->status === 'unresponded', fn($q) => $q->whereDoesntHave('response'))
            ->when($request->rating, fn($q) => $q->where('rating', $request->rating))
            ->when($request->search, fn($q) => $q->where('content', 'like', "%{$request->search}%"))
            ->orderBy('created_at', 'desc')
            ->paginate(20);
            
        return Inertia::render('User/Reviews/Index', [
            'reviews' => $reviews,
            'filters' => $request->only(['platform', 'status', 'rating', 'search']),
            'stats' => [
                'total' => $business->reviews()->count(),
                'unresponded' => $business->reviews()->whereDoesntHave('response')->count(),
                'averageRating' => $business->reviews()->avg('rating') ?? 0,
                'trend' => $business->calculateRatingTrend(),
            ],
        ]);
    }
    
    public function generateResponse(Review $review, ReviewResponseGenerator $generator)
    {
        $this->authorize('respond', $review);
        
        $suggestions = $generator->generateSuggestions($review);
        
        return response()->json([
            'suggestions' => $suggestions,
            'review' => $review->load('platform'),
        ]);
    }
}
```

### Portal Separation Strategy

#### File Organization
```
app/Http/Controllers/
├── Admin/
│   ├── DashboardController.php      # Platform metrics, user activity
│   ├── UsersController.php          # User/business management
│   ├── AnalyticsController.php      # Platform-wide analytics
│   ├── AIMonitorController.php     # AI usage and costs
│   └── SettingsController.php       # Platform settings
└── User/
    ├── DashboardController.php      # Business overview
    ├── ReviewController.php         # Review management
    ├── ResponseController.php       # Response handling
    ├── PlatformController.php       # Platform connections
    ├── AnalyticsController.php      # Business analytics
    └── SettingsController.php       # Business settings

resources/js/pages/
├── Admin/
│   ├── Dashboard.tsx               # Admin metrics overview
│   ├── Users/
│   │   ├── Index.tsx              # All users list
│   │   └── Show.tsx               # User details & businesses
│   ├── Analytics/
│   │   ├── Overview.tsx           # Platform analytics
│   │   └── AIUsage.tsx            # AI cost tracking
│   └── Settings/
│       └── Platform.tsx           # Platform configuration
└── User/
    ├── Dashboard.tsx              # Business dashboard
    ├── Reviews/
    │   ├── Index.tsx             # All reviews
    │   ├── Respond.tsx           # Response interface
    │   └── Analytics.tsx         # Review analytics
    ├── Platforms/
    │   ├── Index.tsx             # Connected platforms
    │   └── Connect.tsx           # OAuth connections
    └── Settings/
        ├── Business.tsx          # Business profile
        └── Notifications.tsx     # Alert preferences
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
```yaml
Tasks:
  - Set up Laravel React starter kit ✓
  - Configure local development environment
  - Implement business and review models
  - Create OAuth integrations for Google, Yelp
  - Build basic review dashboard
  - Set up review import jobs

Deliverables:
  - User can connect Google My Business
  - Reviews import and display
  - Basic filtering and search
  - Manual response capability
```

### Phase 2: AI Integration (Week 2)
```yaml
Tasks:
  - Integrate OpenAI GPT-4 API
  - Build response generation service
  - Create response templates system
  - Implement sentiment analysis
  - Add response approval workflow
  - Build AI training on past responses

Deliverables:
  - AI generates 3 response options per review
  - Sentiment analysis on all reviews
  - Response history tracking
  - Custom AI fine-tuning per industry
```

### Phase 3: Multi-Platform & Analytics (Week 3)
```yaml
Tasks:
  - Add Facebook, TripAdvisor, Yelp APIs
  - Build unified review queue
  - Create analytics dashboard
  - Implement reputation monitoring
  - Add competitor tracking
  - Build reporting system

Deliverables:
  - 5+ platforms supported
  - Comprehensive analytics
  - Weekly email reports
  - Reputation score tracking
```

## Data Architecture

### Database Schema
```sql
-- Core Business Tables
CREATE TABLE businesses (
    id BIGINT UNSIGNED PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100) NOT NULL,
    description TEXT,
    website VARCHAR(255),
    phone VARCHAR(50),
    address JSON,
    timezone VARCHAR(50) DEFAULT 'UTC',
    settings JSON,
    reputation_score DECIMAL(3,2) DEFAULT 0.00,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_industry (industry)
);

-- Platform Connections
CREATE TABLE platforms (
    id BIGINT UNSIGNED PRIMARY KEY,
    business_id BIGINT UNSIGNED NOT NULL,
    platform_type ENUM('google', 'yelp', 'facebook', 'tripadvisor') NOT NULL,
    platform_business_id VARCHAR(255) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP,
    sync_enabled BOOLEAN DEFAULT TRUE,
    last_synced_at TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE,
    UNIQUE KEY unique_business_platform (business_id, platform_type),
    INDEX idx_sync (sync_enabled, last_synced_at)
);

-- Reviews
CREATE TABLE reviews (
    id BIGINT UNSIGNED PRIMARY KEY,
    business_id BIGINT UNSIGNED NOT NULL,
    platform_id BIGINT UNSIGNED NOT NULL,
    platform_review_id VARCHAR(255) NOT NULL,
    author_name VARCHAR(255),
    author_avatar VARCHAR(500),
    rating TINYINT NOT NULL,
    content TEXT,
    sentiment ENUM('positive', 'negative', 'neutral', 'mixed'),
    sentiment_score DECIMAL(3,2),
    language VARCHAR(10) DEFAULT 'en',
    published_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE,
    FOREIGN KEY (platform_id) REFERENCES platforms(id) ON DELETE CASCADE,
    UNIQUE KEY unique_platform_review (platform_id, platform_review_id),
    INDEX idx_business_rating (business_id, rating),
    INDEX idx_published (published_at),
    INDEX idx_sentiment (sentiment)
);

-- AI-Generated Responses
CREATE TABLE responses (
    id BIGINT UNSIGNED PRIMARY KEY,
    review_id BIGINT UNSIGNED NOT NULL,
    business_id BIGINT UNSIGNED NOT NULL,
    content TEXT NOT NULL,
    ai_generated BOOLEAN DEFAULT TRUE,
    ai_model VARCHAR(50),
    ai_confidence DECIMAL(3,2),
    status ENUM('draft', 'approved', 'published', 'rejected') DEFAULT 'draft',
    published_at TIMESTAMP NULL,
    approved_by BIGINT UNSIGNED NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews(id) ON DELETE CASCADE,
    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE,
    FOREIGN KEY (approved_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_status (status),
    INDEX idx_business_status (business_id, status)
);

-- Analytics Events
CREATE TABLE review_analytics (
    id BIGINT UNSIGNED PRIMARY KEY,
    business_id BIGINT UNSIGNED NOT NULL,
    date DATE NOT NULL,
    total_reviews INT DEFAULT 0,
    new_reviews INT DEFAULT 0,
    average_rating DECIMAL(2,1),
    response_rate DECIMAL(3,2),
    sentiment_breakdown JSON,
    platform_breakdown JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (business_id) REFERENCES businesses(id) ON DELETE CASCADE,
    UNIQUE KEY unique_business_date (business_id, date),
    INDEX idx_date (date)
);
```

### Caching Strategy
```php
// config/cache.php additions
'cache_keys' => [
    'business_stats' => 'business:{id}:stats',              // TTL: 5 minutes
    'review_sentiment' => 'review:{id}:sentiment',         // TTL: 1 hour
    'platform_sync' => 'platform:{id}:sync_status',       // TTL: 1 minute
    'ai_suggestions' => 'review:{id}:ai_suggestions',     // TTL: 24 hours
    'reputation_score' => 'business:{id}:reputation',     // TTL: 15 minutes
];
```

## AI Integration

### Review Response Generator
```php
// app/Services/AI/ReviewResponseGenerator.php
namespace App\Services\AI;

use App\Models\Review;
use App\Models\Business;
use OpenAI\Client;

class ReviewResponseGenerator
{
    private Client $openai;
    private ResponseTemplateService $templates;
    
    public function __construct()
    {
        $this->openai = OpenAI::client(config('services.openai.key'));
        $this->templates = app(ResponseTemplateService::class);
    }
    
    public function generateSuggestions(Review $review): array
    {
        $business = $review->business;
        $context = $this->buildContext($review, $business);
        
        // Generate 3 different response styles
        $styles = ['professional', 'friendly', 'empathetic'];
        $suggestions = [];
        
        foreach ($styles as $style) {
            $prompt = $this->buildPrompt($review, $business, $style, $context);
            
            $response = $this->openai->chat()->create([
                'model' => 'gpt-4',
                'messages' => [
                    ['role' => 'system', 'content' => $this->getSystemPrompt($business)],
                    ['role' => 'user', 'content' => $prompt],
                ],
                'temperature' => 0.7,
                'max_tokens' => 300,
            ]);
            
            $suggestions[] = [
                'style' => $style,
                'content' => $response->choices[0]->message->content,
                'confidence' => $this->calculateConfidence($response),
            ];
        }
        
        return $suggestions;
    }
    
    private function buildContext(Review $review, Business $business): array
    {
        return [
            'previous_responses' => $this->getSuccessfulResponses($business, $review->rating),
            'business_info' => [
                'name' => $business->name,
                'industry' => $business->industry,
                'typical_response_length' => $business->settings['response_length'] ?? 'medium',
            ],
            'review_patterns' => $this->identifyPatterns($review),
        ];
    }
    
    private function getSystemPrompt(Business $business): string
    {
        return "You are a professional review response writer for {$business->name}, 
                a {$business->industry} business. Generate authentic, helpful responses 
                that address customer concerns while maintaining the brand voice. 
                Always thank the customer, address specific points they raised, 
                and invite them back when appropriate.";
    }
}
```

### Sentiment Analysis Service
```php
// app/Services/AI/SentimentAnalyzer.php
namespace App\Services\AI;

use App\Models\Review;

class SentimentAnalyzer
{
    public function analyze(Review $review): array
    {
        $response = $this->openai->chat()->create([
            'model' => 'gpt-3.5-turbo',
            'messages' => [
                [
                    'role' => 'system', 
                    'content' => 'Analyze the sentiment of this review. Return JSON with: 
                                  sentiment (positive/negative/neutral/mixed), 
                                  score (-1 to 1), 
                                  key_points (array of main topics), 
                                  urgency (low/medium/high)'
                ],
                ['role' => 'user', 'content' => $review->content],
            ],
            'temperature' => 0.3,
            'response_format' => ['type' => 'json_object'],
        ]);
        
        $analysis = json_decode($response->choices[0]->message->content, true);
        
        // Update review with sentiment
        $review->update([
            'sentiment' => $analysis['sentiment'],
            'sentiment_score' => $analysis['score'],
        ]);
        
        return $analysis;
    }
}
```

## Real-Time Features

### WebSocket Integration
```javascript
// resources/js/hooks/useReviewUpdates.ts
import { useEffect } from 'react';
import Echo from 'laravel-echo';

export function useReviewUpdates(businessId: number, onNewReview: (review: Review) => void) {
    useEffect(() => {
        const channel = Echo.private(`business.${businessId}`)
            .listen('NewReviewReceived', (e: { review: Review }) => {
                onNewReview(e.review);
                
                // Show notification
                new Notification('New Review!', {
                    body: `${e.review.author_name} left a ${e.review.rating}-star review`,
                    icon: '/icon-192x192.png',
                });
            })
            .listen('ResponsePublished', (e: { response: Response }) => {
                // Update UI to show response was published
            });
            
        return () => {
            Echo.leave(`business.${businessId}`);
        };
    }, [businessId]);
}
```

## Performance & Scaling

### Queue Jobs for Platform Sync
```php
// app/Jobs/SyncPlatformReviews.php
namespace App\Jobs;

use App\Models\Platform;
use App\Services\Platforms\PlatformFactory;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Queue\InteractsWithQueue;

class SyncPlatformReviews implements ShouldQueue
{
    use InteractsWithQueue, Queueable;
    
    public $timeout = 300; // 5 minutes
    public $tries = 3;
    
    public function __construct(
        public Platform $platform
    ) {}
    
    public function handle(): void
    {
        $service = PlatformFactory::create($this->platform);
        $reviews = $service->fetchReviews($this->platform->last_synced_at);
        
        foreach ($reviews as $reviewData) {
            ProcessReview::dispatch($this->platform, $reviewData);
        }
        
        $this->platform->update(['last_synced_at' => now()]);
    }
    
    public function failed(\Throwable $exception): void
    {
        // Notify business owner of sync failure
        $this->platform->business->user->notify(
            new PlatformSyncFailed($this->platform, $exception->getMessage())
        );
    }
}
```

### Scheduled Tasks
```php
// app/Console/Kernel.php
protected function schedule(Schedule $schedule): void
{
    // Sync all platforms every 30 minutes
    $schedule->job(new SyncAllPlatforms)
        ->everyThirtyMinutes()
        ->withoutOverlapping();
    
    // Generate daily analytics
    $schedule->job(new GenerateDailyAnalytics)
        ->dailyAt('02:00');
    
    // Send weekly reports
    $schedule->job(new SendWeeklyReports)
        ->weeklyOn(1, '09:00'); // Mondays at 9 AM
    
    // Clean up old data
    $schedule->job(new CleanupOldAnalytics)
        ->monthly();
}
```

This example shows how the three documents work together to provide comprehensive guidance for AI-driven development, with actual code examples and clear implementation paths.