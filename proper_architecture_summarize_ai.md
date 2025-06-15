# summarize-ai-mobile Architecture

## Overview
AI-powered mobile application that enables busy professionals, students, and researchers to instantly summarize lengthy documents, articles, and PDFs with a single tap, saving hours of reading time.

### Problem Statement
Information overload is real. Professionals and students struggle to keep up with the vast amount of reading material - research papers, articles, reports, and documents. Reading everything thoroughly is time-consuming and often impractical, leading to missed insights and delayed decisions.

### Target Audience
- **Busy Professionals**: Executives, consultants, and analysts who need to quickly digest reports, market research, and industry publications
- **Students & Academics**: University students and researchers dealing with numerous academic papers and study materials
- **Content Creators**: Journalists, bloggers, and writers who need to research multiple sources quickly
- **Legal & Medical Professionals**: Lawyers and doctors who must review lengthy documents and case studies

### Core Entities
- User
- Document
- Summary
- SummaryTemplate
- DocumentSource
- ReadingList
- SummaryHistory
- Subscription
- AIModel

## Design System

### Typography
- **Font**: Inter
- **Font Stack**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`
- **Google Fonts**: https://fonts.google.com/specimen/Inter
- **Rationale**: Inter is highly legible on mobile screens with excellent readability for both UI elements and summary text. Its tall x-height and open apertures make it perfect for reading on small screens.

### Color Palette
- **Primary**: `#2563EB` - Deep blue for primary actions and branding, conveys trust and intelligence
- **Secondary**: `#7C3AED` - Purple for AI-related features and premium elements
- **Accent**: `#10B981` - Green for successful summaries and positive actions
- **Success**: `#059669` - Darker green for completion states
- **Warning**: `#F59E0B` - Amber for warnings about document size or processing time
- **Error**: `#EF4444` - Red for errors and failed uploads
- **Background**: `#F9FAFB` - Light gray for main app background
- **Surface**: `#FFFFFF` - Pure white for cards and document viewers
- **Text Primary**: `#111827` - Near black for maximum readability
- **Text Secondary**: `#6B7280` - Medium gray for secondary information

## Foundational Architecture

### 1. Core Components & Rationale

**Mobile App Components**:
- `DocumentUploader`: Handles file uploads, URL inputs, and camera captures with format validation
- `SummaryEngine`: Core AI integration component that manages summarization requests and responses
- `DocumentViewer`: Native document rendering with highlighting and annotation support
- `SummaryDisplay`: Adaptive summary presentation with adjustable detail levels
- `ReadingListManager`: Offline-capable document and summary storage
- `SubscriptionGate`: Premium feature access control and usage tracking

**Backend Services**:
- `DocumentProcessingService`: Extracts text from PDFs, images (OCR), and web pages
- `AIOrchestrationService`: Manages multiple AI models and selects the best one for each document type
- `SummaryGenerationService`: Handles the actual summarization with fallback strategies
- `UsageTrackingService`: Monitors API usage, credits, and subscription limits
- `CacheService`: Intelligent caching of summaries to reduce API costs

### 2. Database Schema Design

```sql
-- Users table with subscription management
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    subscription_tier ENUM('free', 'pro', 'enterprise') DEFAULT 'free',
    monthly_summary_count INT DEFAULT 0,
    total_summaries_generated INT DEFAULT 0,
    preferred_summary_length ENUM('brief', 'standard', 'detailed') DEFAULT 'standard',
    preferred_language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Documents table for uploaded/linked content
CREATE TABLE documents (
    id BIGINT PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    title VARCHAR(500),
    source_type ENUM('upload', 'url', 'camera', 'text') NOT NULL,
    source_url TEXT,
    file_path VARCHAR(500),
    file_size_bytes BIGINT,
    page_count INT,
    word_count INT,
    language VARCHAR(10),
    document_type VARCHAR(50), -- pdf, article, research_paper, etc.
    extracted_text LONGTEXT,
    metadata JSON, -- author, publication date, etc.
    created_at TIMESTAMP,
    INDEX idx_user_documents (user_id, created_at)
);

-- Summaries with versioning and quality tracking
CREATE TABLE summaries (
    id BIGINT PRIMARY KEY,
    document_id BIGINT REFERENCES documents(id),
    user_id BIGINT REFERENCES users(id),
    ai_model VARCHAR(50), -- gpt-4, claude-3, gemini-pro, etc.
    summary_type ENUM('key_points', 'abstract', 'comprehensive', 'bullets') NOT NULL,
    length_preference ENUM('brief', 'standard', 'detailed') NOT NULL,
    summary_text LONGTEXT NOT NULL,
    key_insights JSON, -- structured extraction of main points
    processing_time_ms INT,
    token_count INT,
    quality_score DECIMAL(3,2), -- AI-generated quality metric
    user_rating INT, -- 1-5 star rating
    created_at TIMESTAMP,
    INDEX idx_user_summaries (user_id, created_at),
    INDEX idx_document_summaries (document_id)
);

-- Reading lists for organization
CREATE TABLE reading_lists (
    id BIGINT PRIMARY KEY,
    user_id BIGINT REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    INDEX idx_user_lists (user_id)
);

-- Summary templates for consistent formatting
CREATE TABLE summary_templates (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    prompt_template TEXT NOT NULL,
    output_format JSON,
    is_default BOOLEAN DEFAULT FALSE,
    category VARCHAR(50), -- academic, business, legal, medical
    created_at TIMESTAMP
);
```

### 3. API Design & Key Endpoints

**Document Management**:
- `POST /api/documents/upload` - Upload PDF, DOCX, TXT files (max 10MB free, 50MB pro)
- `POST /api/documents/url` - Submit URL for web article summarization
- `POST /api/documents/text` - Direct text input for summarization
- `GET /api/documents` - List user's documents with pagination
- `GET /api/documents/{id}` - Retrieve document details and extracted text
- `DELETE /api/documents/{id}` - Remove document and associated summaries

**Summarization**:
- `POST /api/summaries/generate` - Create new summary with options
  ```json
  {
    "document_id": 123,
    "summary_type": "key_points",
    "length": "standard",
    "language": "en",
    "focus_areas": ["conclusions", "methodology"]
  }
  ```
- `GET /api/summaries/{id}` - Retrieve specific summary
- `POST /api/summaries/{id}/regenerate` - Re-generate with different parameters
- `POST /api/summaries/{id}/rate` - Rate summary quality
- `GET /api/summaries/history` - User's summary history

**Reading Lists**:
- `POST /api/reading-lists` - Create new reading list
- `POST /api/reading-lists/{id}/documents` - Add document to list
- `GET /api/reading-lists/{id}/summaries` - Get all summaries in a list
- `POST /api/reading-lists/{id}/share` - Generate shareable link

**Subscription & Usage**:
- `GET /api/usage/current` - Current month's usage statistics
- `GET /api/subscription/plans` - Available subscription tiers
- `POST /api/subscription/upgrade` - Upgrade subscription
- `GET /api/subscription/limits` - Check remaining summaries

### 4. Frontend Structure

```
src/
├── components/
│   ├── document/
│   │   ├── DocumentUploader.tsx
│   │   ├── DocumentPreview.tsx
│   │   ├── DocumentList.tsx
│   │   └── DocumentMetadata.tsx
│   ├── summary/
│   │   ├── SummaryDisplay.tsx
│   │   ├── SummaryOptions.tsx
│   │   ├── KeyPointsView.tsx
│   │   ├── SummaryActions.tsx
│   │   └── SummaryHistory.tsx
│   ├── navigation/
│   │   ├── TabBar.tsx
│   │   ├── HeaderBar.tsx
│   │   └── QuickActions.tsx
│   └── subscription/
│       ├── UsageIndicator.tsx
│       ├── UpgradePrompt.tsx
│       └── PlanSelector.tsx
├── screens/
│   ├── HomeScreen.tsx
│   ├── DocumentScreen.tsx
│   ├── SummaryScreen.tsx
│   ├── LibraryScreen.tsx
│   └── SettingsScreen.tsx
├── services/
│   ├── DocumentService.ts
│   ├── SummaryService.ts
│   ├── AIService.ts
│   └── CacheService.ts
└── hooks/
    ├── useDocument.ts
    ├── useSummary.ts
    ├── useOfflineSync.ts
    └── useSubscription.ts
```

## Feature Implementation Roadmap

### 1. Instant Document Summarization
**Description**: Core feature allowing users to upload documents and receive AI-generated summaries within seconds
**Priority**: High
**Complexity**: High

**Database Changes**:
- Initial schema creation for documents and summaries tables
- Indexes for fast retrieval by user and document

**API Endpoints**:
- `POST /api/documents/upload`
- `POST /api/summaries/generate`
- `GET /api/summaries/{id}`

**Frontend Components**:
- DocumentUploader with drag-drop support
- SummaryDisplay with loading states
- ProgressIndicator for processing feedback

### 2. Multi-Format Support
**Description**: Support for PDFs, Word docs, web articles, and images with OCR
**Priority**: High
**Complexity**: Medium

**Database Changes**:
- Add document_type and extraction_method fields
- Store OCR confidence scores

**API Endpoints**:
- `POST /api/documents/analyze` - Detect document type
- `POST /api/documents/ocr` - Process images

**Frontend Components**:
- FormatDetector component
- ImageCropper for OCR optimization
- URLInput with preview

### 3. Adjustable Summary Length
**Description**: Let users choose between brief (1 paragraph), standard (3-5 paragraphs), or detailed (full page) summaries
**Priority**: Medium
**Complexity**: Low

**Database Changes**:
- Add length_preference to summaries table
- Store multiple summary versions

**API Endpoints**:
- Update `POST /api/summaries/generate` with length parameter
- `PUT /api/summaries/{id}/adjust-length`

**Frontend Components**:
- LengthSelector slider
- SummaryComparison view
- PreferenceManager

### 4. Offline Reading Lists
**Description**: Save documents and summaries for offline access
**Priority**: Medium
**Complexity**: Medium

**Database Changes**:
- Add sync_status to documents and summaries
- Create offline_queue table

**API Endpoints**:
- `POST /api/sync/download`
- `POST /api/sync/upload`
- `GET /api/sync/status`

**Frontend Components**:
- OfflineIndicator
- SyncManager
- DownloadProgress
- LocalStorage wrapper

### 5. Smart Highlighting
**Description**: AI identifies and highlights the most important sentences in the original document
**Priority**: Low
**Complexity**: High

**Database Changes**:
- Create highlights table with position data
- Add importance_scores to summaries

**API Endpoints**:
- `POST /api/documents/{id}/analyze-importance`
- `GET /api/documents/{id}/highlights`

**Frontend Components**:
- DocumentAnnotator
- HighlightOverlay
- ImportanceHeatmap

## Technical Implementation Details

### AI Model Selection
- **Primary**: GPT-4 for research papers and technical documents
- **Secondary**: Claude 3 for creative and narrative content
- **Fallback**: Gemini Pro for general purpose summarization
- **Specialized**: Bio-medical model for healthcare documents

### Caching Strategy
- Cache summaries for 30 days
- Implement smart cache warming for popular documents
- Use Redis for hot summaries
- CDN for static document previews

### Performance Optimization
- Lazy load document sections
- Progressive summary rendering
- Background processing for large documents
- WebSocket updates for real-time progress

### Security Measures
- End-to-end encryption for sensitive documents
- Automatic PII detection and redaction
- Secure file upload with virus scanning
- JWT tokens with refresh rotation

## Monetization Strategy

### Freemium Model
- **Free Tier**: 10 summaries/month, max 10 pages per document
- **Pro Tier** ($9.99/month): 100 summaries, 50 pages, priority processing
- **Enterprise** (Custom): Unlimited, API access, custom models

### Additional Revenue Streams
- Team collaboration features
- Custom AI model training
- White-label solutions for educational institutions
- API access for developers

## Success Metrics
- Average summary generation time < 5 seconds
- User satisfaction rating > 4.5/5
- Monthly active users growth > 20%
- Subscription conversion rate > 5%
- Summary accuracy score > 90%