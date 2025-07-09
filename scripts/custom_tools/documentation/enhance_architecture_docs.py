"""
Enhanced Architecture Documentation Tool

This tool enhances existing ARCHITECTURE.md files to meet comprehensive standards including:
- Complete Admin/User system separation
- Detailed API endpoint documentation with examples
- Database schemas with relationships and indexes
- Redis and Laravel Echo configuration
- Security architecture with role-based access
- Performance optimizations and caching strategies

Can process single projects or batch enhance multiple projects.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import re

# Comprehensive template showing expected documentation standard
COMPREHENSIVE_ARCHITECTURE_TEMPLATE = """
# [Project Name] Architecture

## Overview
[Detailed overview explaining the project purpose, main features, and value proposition. Should be 3-5 paragraphs covering problem statement, solution approach, and key differentiators.]

### Problem Statement
[Specific problem the application addresses with context about why existing solutions fall short]

### Target Audience
[Primary and secondary user groups with their specific needs and use cases]

## Core Entities
- **Users**: Primary platform users with profiles, preferences, activity tracking, and subscription management
- **Admins**: System administrators with role-based access (super_admin, admin, moderator, support)
- **[Domain Entity 1]**: [Description with relationships to other entities]
- **[Domain Entity 2]**: [Description with business rules and constraints]
- **[Domain Entity 3]**: [Description of data lifecycle and state transitions]

## Design System

### Typography
- **Font**: Inter
- **Font Stack**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Font Sizes**: 
  - Heading 1: 2.5rem (40px)
  - Heading 2: 2rem (32px)
  - Heading 3: 1.5rem (24px)
  - Body: 1rem (16px)
  - Small: 0.875rem (14px)
- **Line Heights**: 1.5 for body text, 1.2 for headings
- **Font Weights**: 400 (regular), 500 (medium), 600 (semibold), 700 (bold)

### Color Palette
- **Primary Blue**: `#0066CC` - Primary actions, links, active states, and main CTA buttons
- **Secondary Gray**: `#6B7280` - Secondary text, inactive elements, and borders
- **Accent Orange**: `#F59E0B` - Highlights, notifications, and important alerts
- **Success Green**: `#10B981` - Success messages, positive states, and confirmations
- **Warning Yellow**: `#F59E0B` - Warning states, caution messages, and pending actions
- **Error Red**: `#EF4444` - Error messages, destructive actions, and validation errors
- **Background White**: `#FFFFFF` - Main content background
- **Background Gray**: `#F9FAFB` - Secondary background for sections
- **Text Primary**: `#111827` - Main text color for optimal readability
- **Text Secondary**: `#6B7280` - Secondary and helper text
- **Border Default**: `#E5E7EB` - Default borders and dividers
- **Border Focus**: `#0066CC` - Focus states for accessibility

## System Architecture

### 1. High-Level Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Admin Portal  │     │   User Portal   │     │   Mobile App    │
│   (React SPA)   │     │   (React SPA)   │     │ (React Native)  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    API Gateway         │
                    │  (Laravel Sanctum)     │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────┴────────┐    ┌────────┴────────┐    ┌────────┴────────┐
│  Admin Service  │    │  User Service   │    │ Notification    │
│   (Laravel)     │    │   (Laravel)     │    │    Service      │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │      Data Layer        │
                    ├─────────────────────────┤
                    │  MySQL  │ Redis │ S3   │
                    └─────────────────────────┘
```

### 2. Core Components

#### Admin System Architecture
```
app/Http/Controllers/Admin/
├── AdminDashboardController.php
│   ├── index() - Main dashboard with KPIs and real-time metrics
│   ├── analytics() - Detailed analytics with date ranges
│   ├── systemHealth() - Infrastructure and service health
│   └── exportReport() - Generate PDF/Excel reports
│
├── AdminUserController.php
│   ├── index() - Paginated user list with advanced filters
│   ├── show($id) - Detailed user profile with full history
│   ├── create() - Create new user form
│   ├── store() - Process new user creation
│   ├── edit($id) - Edit user form
│   ├── update($id) - Update user information
│   ├── destroy($id) - Soft delete user
│   ├── suspend($id) - Suspend with reason and duration
│   ├── restore($id) - Restore suspended/deleted user
│   ├── bulkAction() - Handle bulk operations (export, suspend, delete)
│   ├── activityLog($id) - User activity timeline
│   ├── loginAs($id) - Impersonate user for support
│   └── forceLogout($id) - Force logout all user sessions
│
├── AdminReportController.php
│   ├── index() - Available reports list
│   ├── generate() - Generate custom report
│   ├── schedule() - Schedule recurring reports
│   ├── download($id) - Download generated report
│   └── templates() - Manage report templates
│
├── AdminSystemController.php
│   ├── configuration() - System settings management
│   ├── updateConfig() - Update configuration values
│   ├── maintenance() - Toggle maintenance mode
│   ├── cache() - Cache management interface
│   ├── clearCache() - Clear specific cache keys
│   ├── logs() - View system logs
│   ├── queues() - Monitor queue status
│   └── scheduler() - View scheduled tasks
│
├── AdminContentController.php
│   ├── moderationQueue() - Content awaiting approval
│   ├── approve($id) - Approve content
│   ├── reject($id) - Reject with reason
│   ├── bulkModerate() - Bulk content actions
│   └── reportedContent() - User-reported items
│
└── AdminNotificationController.php
    ├── broadcast() - Send system-wide notification
    ├── targeted() - Send to specific user groups
    ├── scheduled() - Manage scheduled notifications
    └── templates() - Notification templates

app/Services/Admin/
├── UserManagementService.php
│   ├── searchUsers($criteria) - Advanced user search
│   ├── bulkExport($userIds) - Export user data
│   ├── bulkSuspend($userIds, $reason) - Suspend multiple users
│   ├── generateUserReport($userId) - Detailed user report
│   ├── mergeAccounts($primaryId, $secondaryId) - Merge duplicate accounts
│   └── calculateUserMetrics($userId) - User engagement metrics
│
├── SystemMonitoringService.php
│   ├── getSystemHealth() - Overall system status
│   ├── checkServiceStatus($service) - Individual service health
│   ├── getPerformanceMetrics() - Performance indicators
│   ├── detectAnomalies() - Anomaly detection
│   └── generateHealthReport() - System health report
│
├── ReportGenerationService.php
│   ├── generateReport($type, $parameters) - Create reports
│   ├── scheduleReport($config) - Schedule recurring reports
│   ├── exportToFormat($data, $format) - Export PDF/Excel/CSV
│   └── emailReport($report, $recipients) - Email distribution
│
├── AuditLogService.php
│   ├── logAdminAction($admin, $action, $target) - Log admin actions
│   ├── getAuditTrail($filters) - Retrieve audit logs
│   ├── exportAuditLogs($dateRange) - Export for compliance
│   └── detectSuspiciousActivity() - Security monitoring
│
└── BulkOperationService.php
    ├── processInBatches($items, $operation) - Batch processing
    ├── createJob($operation, $items) - Queue bulk operations
    ├── getJobStatus($jobId) - Check operation status
    └── downloadResults($jobId) - Get operation results
```

#### User System Architecture
```
app/Http/Controllers/User/
├── UserAuthController.php
│   ├── register() - User registration
│   ├── login() - User authentication
│   ├── logout() - End session
│   ├── refresh() - Refresh JWT token
│   ├── forgotPassword() - Password reset request
│   ├── resetPassword() - Process password reset
│   └── verifyEmail() - Email verification
│
├── UserProfileController.php
│   ├── show() - Display profile
│   ├── update() - Update profile information
│   ├── uploadAvatar() - Handle avatar upload
│   ├── updatePreferences() - User preferences
│   ├── privacySettings() - Privacy configuration
│   ├── exportData() - GDPR data export
│   └── deleteAccount() - Account deletion
│
├── UserDashboardController.php
│   ├── index() - Personalized dashboard
│   ├── widgets() - Dashboard widgets data
│   ├── quickStats() - Quick statistics
│   └── recentActivity() - Recent user activity
│
├── [DomainSpecificController].php
│   └── [Feature-specific methods]

app/Services/User/
├── AuthenticationService.php
│   ├── attemptLogin($credentials) - Login logic
│   ├── createToken($user) - JWT generation
│   ├── invalidateTokens($user) - Logout all devices
│   └── verify2FA($user, $code) - Two-factor auth
│
├── ProfileService.php
│   ├── updateProfile($user, $data) - Profile updates
│   ├── processAvatar($file) - Avatar processing
│   ├── exportUserData($user) - GDPR export
│   └── scheduleAccountDeletion($user) - Soft delete
│
├── NotificationService.php
│   ├── send($user, $notification) - Send notification
│   ├── markAsRead($notificationId) - Mark read
│   ├── getUserNotifications($user) - Get all notifications
│   └── updatePreferences($user, $preferences) - Notification settings
│
└── [DomainSpecificService].php
    └── [Business logic methods]
```

### 3. Database Schema Design

#### Admin-Specific Tables
```sql
-- Admin users table with role-based access
CREATE TABLE admins (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('super_admin', 'admin', 'moderator', 'support') NOT NULL DEFAULT 'support',
    permissions JSON COMMENT 'Additional granular permissions',
    two_factor_secret VARCHAR(255),
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    last_login_at TIMESTAMP NULL,
    last_login_ip VARCHAR(45),
    failed_login_attempts INT DEFAULT 0,
    locked_until TIMESTAMP NULL,
    created_by BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    FOREIGN KEY (created_by) REFERENCES admins(id),
    INDEX idx_email (email),
    INDEX idx_role (role),
    INDEX idx_deleted (deleted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Admin activity logging for audit trail
CREATE TABLE admin_activity_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    admin_id BIGINT NOT NULL,
    action VARCHAR(255) NOT NULL COMMENT 'e.g., user.suspended, config.updated',
    target_type VARCHAR(100) COMMENT 'Model type affected',
    target_id BIGINT COMMENT 'ID of affected record',
    previous_values JSON COMMENT 'State before change',
    new_values JSON COMMENT 'State after change',
    ip_address VARCHAR(45) NOT NULL,
    user_agent TEXT,
    metadata JSON COMMENT 'Additional context',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admins(id),
    INDEX idx_admin_action (admin_id, action),
    INDEX idx_target (target_type, target_id),
    INDEX idx_created (created_at),
    INDEX idx_ip (ip_address)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System configuration management
CREATE TABLE system_configurations (
    `key` VARCHAR(255) PRIMARY KEY,
    `value` JSON NOT NULL,
    `type` ENUM('string', 'integer', 'boolean', 'json', 'array') DEFAULT 'string',
    `category` VARCHAR(100) COMMENT 'Config category for grouping',
    `description` TEXT,
    `is_sensitive` BOOLEAN DEFAULT FALSE COMMENT 'Hide in logs',
    `updated_by` BIGINT,
    `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (updated_by) REFERENCES admins(id),
    INDEX idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Admin sessions tracking
CREATE TABLE admin_sessions (
    id VARCHAR(255) PRIMARY KEY,
    admin_id BIGINT NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admins(id) ON DELETE CASCADE,
    INDEX idx_admin (admin_id),
    INDEX idx_last_activity (last_activity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### User Tables with Relationships
```sql
-- Main users table
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email_verified_at TIMESTAMP NULL,
    remember_token VARCHAR(100),
    subscription_tier ENUM('free', 'basic', 'premium', 'pro', 'enterprise') DEFAULT 'free',
    subscription_expires_at TIMESTAMP NULL,
    trial_ends_at TIMESTAMP NULL,
    status ENUM('active', 'suspended', 'deleted', 'pending') DEFAULT 'pending',
    suspension_reason TEXT,
    suspended_at TIMESTAMP NULL,
    suspended_until TIMESTAMP NULL,
    referred_by BIGINT,
    referral_code VARCHAR(20) UNIQUE,
    last_login_at TIMESTAMP NULL,
    last_login_ip VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    FOREIGN KEY (referred_by) REFERENCES users(id),
    INDEX idx_email (email),
    INDEX idx_status_subscription (status, subscription_tier),
    INDEX idx_created (created_at),
    INDEX idx_deleted (deleted_at),
    INDEX idx_referral_code (referral_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- User profiles with detailed information
CREATE TABLE user_profiles (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    display_name VARCHAR(100),
    avatar_url VARCHAR(500),
    bio TEXT,
    date_of_birth DATE,
    gender ENUM('male', 'female', 'other', 'prefer_not_to_say'),
    phone_number VARCHAR(20),
    country_code VARCHAR(2),
    timezone VARCHAR(50) DEFAULT 'UTC',
    locale VARCHAR(10) DEFAULT 'en_US',
    metadata JSON COMMENT 'Additional profile data',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user (user_id),
    INDEX idx_name (first_name, last_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- User preferences and settings
CREATE TABLE user_preferences (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT UNIQUE NOT NULL,
    theme ENUM('light', 'dark', 'auto') DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'en',
    email_notifications JSON DEFAULT '{"marketing": true, "updates": true, "social": true}',
    push_notifications JSON DEFAULT '{"enabled": true, "sound": true, "vibrate": true}',
    privacy_settings JSON DEFAULT '{"profile_visibility": "public", "show_activity": true}',
    feature_flags JSON COMMENT 'User-specific feature toggles',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- User activity tracking
CREATE TABLE user_activities (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    action VARCHAR(255) NOT NULL,
    description TEXT,
    metadata JSON,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_action (user_id, action),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Add domain-specific tables here
```

### 4. API Design

#### Admin API Endpoints

##### Authentication & Authorization
```
POST   /api/admin/login
       Description: Admin login with optional 2FA
       Body: {
         "email": "admin@example.com",
         "password": "secure_password",
         "two_factor_code": "123456" // Optional, required if 2FA enabled
       }
       Response: {
         "success": true,
         "admin": {
           "id": 1,
           "name": "John Admin",
           "email": "admin@example.com",
           "role": "admin",
           "permissions": ["users.view", "users.edit", "reports.generate"]
         },
         "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
         "expires_at": "2024-01-15T10:00:00Z"
       }
       Rate Limit: 5 requests per minute per IP
       
POST   /api/admin/logout
       Description: Invalidate current session
       Headers: Authorization: Bearer {token}
       Response: { "success": true, "message": "Logged out successfully" }
       
GET    /api/admin/me
       Description: Get current admin information
       Headers: Authorization: Bearer {token}
       Response: {
         "id": 1,
         "name": "John Admin",
         "role": "admin",
         "permissions": [...],
         "last_login": "2024-01-14T15:30:00Z"
       }
```

##### User Management
```
GET    /api/admin/users
       Description: Get paginated user list with filters
       Headers: Authorization: Bearer {token}
       Query Parameters:
         - page: 1 (default: 1)
         - per_page: 20 (default: 20, max: 100)
         - search: "john" (searches email, name)
         - status: "active" (active|suspended|deleted|pending)
         - subscription: "premium" (free|basic|premium|pro|enterprise)
         - created_from: "2024-01-01"
         - created_to: "2024-01-31"
         - sort_by: "created_at" (created_at|email|last_login)
         - sort_order: "desc" (asc|desc)
       Response: {
         "data": [
           {
             "id": 1,
             "email": "user@example.com",
             "profile": {
               "first_name": "John",
               "last_name": "Doe",
               "avatar_url": "https://..."
             },
             "subscription_tier": "premium",
             "status": "active",
             "created_at": "2024-01-01T10:00:00Z",
             "last_login_at": "2024-01-14T09:00:00Z"
           }
         ],
         "meta": {
           "current_page": 1,
           "from": 1,
           "to": 20,
           "per_page": 20,
           "total": 150,
           "last_page": 8
         },
         "links": {
           "first": "/api/admin/users?page=1",
           "last": "/api/admin/users?page=8",
           "prev": null,
           "next": "/api/admin/users?page=2"
         }
       }
       Rate Limit: 100 requests per minute
       
GET    /api/admin/users/{id}
       Description: Get detailed user information
       Headers: Authorization: Bearer {token}
       Response: {
         "user": {
           "id": 1,
           "email": "user@example.com",
           "status": "active",
           "subscription_tier": "premium",
           "subscription_expires_at": "2024-12-31T23:59:59Z",
           "profile": { ... },
           "preferences": { ... },
           "statistics": {
             "total_sessions": 145,
             "total_actions": 1893,
             "last_30_days_activity": { ... }
           },
           "created_at": "2024-01-01T10:00:00Z"
         },
         "activity": [
           {
             "action": "login",
             "timestamp": "2024-01-14T09:00:00Z",
             "ip_address": "192.168.1.1",
             "metadata": { ... }
           }
         ],
         "admin_notes": [
           {
             "note": "VIP customer - handle with care",
             "created_by": "Admin Name",
             "created_at": "2024-01-10T14:00:00Z"
           }
         ]
       }
       
POST   /api/admin/users/{id}/suspend
       Description: Suspend a user account
       Headers: Authorization: Bearer {token}
       Body: {
         "reason": "Terms of service violation - spam",
         "duration_days": 30, // Optional, permanent if not specified
         "notify_user": true
       }
       Response: {
         "success": true,
         "user": {
           "id": 1,
           "status": "suspended",
           "suspended_until": "2024-02-14T10:00:00Z"
         }
       }
       
POST   /api/admin/users/bulk
       Description: Perform bulk operations on users
       Headers: Authorization: Bearer {token}
       Body: {
         "action": "export", // export|suspend|delete|update_subscription
         "user_ids": [1, 2, 3, 4, 5],
         "parameters": {
           // Action-specific parameters
           "format": "csv", // for export
           "reason": "Batch suspension", // for suspend
           "subscription_tier": "premium" // for update_subscription
         }
       }
       Response: {
         "success": true,
         "job_id": "job_123456",
         "affected_users": 5,
         "status_url": "/api/admin/jobs/job_123456"
       }
```

##### Analytics & Reporting
```
GET    /api/admin/analytics/dashboard
       Description: Get dashboard metrics
       Headers: Authorization: Bearer {token}
       Query Parameters:
         - period: "30d" (24h|7d|30d|90d|1y|custom)
         - from: "2024-01-01" (for custom period)
         - to: "2024-01-31" (for custom period)
       Response: {
         "metrics": {
           "users": {
             "total": 10000,
             "active": 7500,
             "new_this_period": 500,
             "growth_rate": 5.2
           },
           "revenue": {
             "total": 50000,
             "recurring": 45000,
             "growth_rate": 12.3,
             "churn_rate": 2.1
           },
           "engagement": {
             "daily_active_users": 3500,
             "average_session_duration": 1820,
             "actions_per_user": 45.6
           }
         },
         "charts": {
           "user_growth": [...],
           "revenue_trend": [...],
           "activity_heatmap": [...]
         }
       }
       
POST   /api/admin/reports/generate
       Description: Generate custom report
       Headers: Authorization: Bearer {token}
       Body: {
         "type": "user_engagement", // user_engagement|revenue|system_health|custom
         "format": "pdf", // pdf|excel|csv
         "parameters": {
           "date_from": "2024-01-01",
           "date_to": "2024-01-31",
           "segments": ["subscription_tier", "country"],
           "metrics": ["dau", "session_duration", "feature_usage"]
         },
         "email_to": ["admin@example.com"] // Optional
       }
       Response: {
         "success": true,
         "report_id": "report_789",
         "status": "processing",
         "estimated_completion": "2024-01-14T10:30:00Z",
         "download_url": null // Will be available when complete
       }
```

##### System Management
```
GET    /api/admin/system/config
       Description: Get system configuration
       Headers: Authorization: Bearer {token}
       Query Parameters:
         - category: "features" (features|limits|integrations|all)
       Response: {
         "configurations": {
           "features": {
             "user_registration": {
               "value": true,
               "type": "boolean",
               "description": "Allow new user registrations"
             },
             "maintenance_mode": {
               "value": false,
               "type": "boolean",
               "description": "Enable maintenance mode"
             }
           },
           "limits": {
             "max_upload_size": {
               "value": 10485760,
               "type": "integer",
               "description": "Maximum file upload size in bytes"
             }
           }
         }
       }
       
PUT    /api/admin/system/config/{key}
       Description: Update configuration value
       Headers: Authorization: Bearer {token}
       Body: {
         "value": true,
         "reason": "Enabling feature for beta testing"
       }
       Response: {
         "success": true,
         "configuration": {
           "key": "user_registration",
           "value": true,
           "updated_at": "2024-01-14T10:00:00Z",
           "updated_by": "Admin Name"
         }
       }
       
GET    /api/admin/system/health
       Description: Get system health status
       Headers: Authorization: Bearer {token}
       Response: {
         "status": "healthy", // healthy|degraded|critical
         "services": {
           "database": {
             "status": "healthy",
             "response_time": 15,
             "details": { ... }
           },
           "redis": {
             "status": "healthy",
             "memory_usage": "45%",
             "connected_clients": 23
           },
           "queue": {
             "status": "healthy",
             "pending_jobs": 145,
             "failed_jobs": 2,
             "workers": 5
           }
         },
         "metrics": {
           "cpu_usage": 35.2,
           "memory_usage": 62.1,
           "disk_usage": 41.5,
           "load_average": [1.2, 1.5, 1.8]
         }
       }
```

#### User API Endpoints

##### Authentication
```
POST   /api/user/register
       Description: Register new user account
       Body: {
         "email": "user@example.com",
         "password": "secure_password",
         "password_confirmation": "secure_password",
         "first_name": "John",
         "last_name": "Doe",
         "referral_code": "FRIEND123" // Optional
       }
       Response: {
         "success": true,
         "user": {
           "id": 1,
           "email": "user@example.com",
           "email_verified": false
         },
         "message": "Please check your email to verify your account"
       }
       Rate Limit: 3 requests per hour per IP
       
POST   /api/user/login
       Description: User login
       Body: {
         "email": "user@example.com",
         "password": "secure_password",
         "remember": true
       }
       Response: {
         "success": true,
         "user": {
           "id": 1,
           "email": "user@example.com",
           "subscription_tier": "premium",
           "profile": { ... }
         },
         "token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
         "expires_at": "2024-01-21T10:00:00Z"
       }
       Rate Limit: 10 requests per minute per IP
       
POST   /api/user/password/forgot
       Description: Request password reset
       Body: {
         "email": "user@example.com"
       }
       Response: {
         "success": true,
         "message": "If an account exists, a reset link has been sent"
       }
       Rate Limit: 3 requests per hour per IP
```

##### Profile Management
```
GET    /api/user/profile
       Description: Get current user profile
       Headers: Authorization: Bearer {token}
       Response: {
         "user": {
           "id": 1,
           "email": "user@example.com",
           "subscription_tier": "premium",
           "subscription_expires_at": "2024-12-31T23:59:59Z",
           "profile": {
             "first_name": "John",
             "last_name": "Doe",
             "display_name": "JohnD",
             "avatar_url": "https://...",
             "bio": "Software developer and fitness enthusiast"
           },
           "preferences": {
             "theme": "dark",
             "language": "en",
             "timezone": "America/New_York"
           },
           "statistics": {
             "member_since": "2024-01-01",
             "total_logins": 145,
             "last_login": "2024-01-14T09:00:00Z"
           }
         }
       }
       
PUT    /api/user/profile
       Description: Update profile information
       Headers: Authorization: Bearer {token}
       Body: {
         "first_name": "John",
         "last_name": "Doe",
         "display_name": "JohnD",
         "bio": "Updated bio",
         "date_of_birth": "1990-01-01",
         "phone_number": "+1234567890"
       }
       Response: {
         "success": true,
         "profile": { ... }
       }
       
POST   /api/user/profile/avatar
       Description: Upload profile avatar
       Headers: 
         - Authorization: Bearer {token}
         - Content-Type: multipart/form-data
       Body: {
         "avatar": [file]
       }
       Response: {
         "success": true,
         "avatar_url": "https://storage.example.com/avatars/user_1_avatar.jpg"
       }
       
DELETE /api/user/profile
       Description: Request account deletion
       Headers: Authorization: Bearer {token}
       Body: {
         "password": "current_password",
         "reason": "No longer using the service" // Optional
       }
       Response: {
         "success": true,
         "message": "Account scheduled for deletion in 30 days",
         "deletion_date": "2024-02-14T10:00:00Z"
       }
```

[Add domain-specific API endpoints here]

### 5. Frontend Architecture

#### Admin Frontend Structure
```
resources/js/Admin/
├── components/
│   ├── Dashboard/
│   │   ├── MetricsGrid.tsx              # Grid layout for KPI cards
│   │   ├── MetricCard.tsx               # Individual metric display
│   │   ├── RealtimeChart.tsx            # WebSocket-powered live charts
│   │   ├── UserActivityMap.tsx          # Geographic user distribution
│   │   ├── SystemHealthIndicator.tsx    # Service status indicators
│   │   └── QuickActions.tsx             # Common admin actions
│   │
│   ├── UserManagement/
│   │   ├── UserTable.tsx                # Main user data table
│   │   ├── UserTableRow.tsx             # Individual user row
│   │   ├── UserFilters.tsx              # Advanced filter panel
│   │   ├── UserDetailsModal.tsx         # Detailed user view
│   │   ├── BulkActionBar.tsx            # Bulk operation controls
│   │   ├── UserActivityTimeline.tsx     # User action history
│   │   ├── SuspendUserModal.tsx         # Suspension form
│   │   └── UserExportModal.tsx          # Export configuration
│   │
│   ├── Analytics/
│   │   ├── AnalyticsDashboard.tsx       # Main analytics page
│   │   ├── DateRangePicker.tsx          # Period selection
│   │   ├── MetricsChart.tsx             # Reusable chart component
│   │   ├── UserGrowthChart.tsx          # User acquisition trends
│   │   ├── RevenueChart.tsx             # Financial metrics
│   │   ├── EngagementHeatmap.tsx        # User activity patterns
│   │   ├── CohortAnalysis.tsx           # User retention cohorts
│   │   └── ExportAnalytics.tsx          # Export analytics data
│   │
│   ├── Reports/
│   │   ├── ReportList.tsx               # Available reports
│   │   ├── ReportGenerator.tsx          # Custom report builder
│   │   ├── ReportScheduler.tsx          # Schedule recurring reports
│   │   ├── ReportViewer.tsx             # View generated reports
│   │   └── ReportTemplates.tsx          # Manage report templates
│   │
│   ├── System/
│   │   ├── ConfigurationPanel.tsx       # System settings UI
│   │   ├── FeatureFlags.tsx             # Feature toggle management
│   │   ├── MaintenanceMode.tsx          # Maintenance controls
│   │   ├── CacheManager.tsx             # Cache operations
│   │   ├── QueueMonitor.tsx             # Queue status display
│   │   ├── LogViewer.tsx                # Real-time log viewer
│   │   └── HealthMonitor.tsx            # System health dashboard
│   │
│   └── shared/
│       ├── AdminLayout.tsx              # Main layout wrapper
│       ├── AdminSidebar.tsx             # Navigation sidebar
│       ├── AdminHeader.tsx              # Top navigation bar
│       ├── AdminBreadcrumbs.tsx         # Breadcrumb navigation
│       ├── PermissionGate.tsx           # Role-based component
│       ├── DataTable.tsx                # Reusable data table
│       ├── ConfirmDialog.tsx            # Confirmation modals
│       ├── LoadingSpinner.tsx           # Loading indicators
│       └── ErrorBoundary.tsx            # Error handling wrapper
│
├── pages/
│   ├── Login.tsx                        # Admin login page
│   ├── Dashboard.tsx                    # Main dashboard
│   ├── Users/
│   │   ├── Index.tsx                    # User listing
│   │   ├── Show.tsx                     # User details
│   │   └── Activity.tsx                 # User activity
│   ├── Analytics/
│   │   ├── Overview.tsx                 # Analytics overview
│   │   ├── Revenue.tsx                  # Revenue analytics
│   │   └── Engagement.tsx               # User engagement
│   ├── Reports/
│   │   ├── Index.tsx                    # Reports center
│   │   └── Generate.tsx                 # Report generation
│   └── System/
│       ├── Configuration.tsx            # System config
│       ├── Monitoring.tsx               # Health monitoring
│       └── Logs.tsx                     # System logs
│
├── hooks/
│   ├── useAdminAuth.ts                  # Admin authentication
│   ├── usePermissions.ts                # Permission checking
│   ├── useWebSocket.ts                  # WebSocket connection
│   ├── useRealtimeData.ts               # Real-time updates
│   ├── usePagination.ts                 # Pagination logic
│   ├── useFilters.ts                    # Filter management
│   ├── useDebounce.ts                   # Debounced values
│   └── useAsync.ts                      # Async operations
│
├── services/
│   ├── adminApi.ts                      # Admin API client
│   ├── websocket.ts                     # WebSocket service
│   ├── permissions.ts                   # Permission service
│   └── export.ts                        # Export utilities
│
├── store/
│   ├── adminStore.ts                    # Zustand admin store
│   ├── userManagementStore.ts           # User management state
│   ├── analyticsStore.ts                # Analytics state
│   └── systemStore.ts                   # System config state
│
├── types/
│   ├── admin.ts                         # Admin type definitions
│   ├── user.ts                          # User types
│   ├── analytics.ts                     # Analytics types
│   └── api.ts                           # API response types
│
└── utils/
    ├── formatters.ts                    # Data formatters
    ├── validators.ts                    # Input validation
    ├── constants.ts                     # App constants
    └── helpers.ts                       # Utility functions
```

#### User Frontend Structure
```
resources/js/User/
├── components/
│   ├── Dashboard/
│   │   ├── WelcomeCard.tsx             # Personalized welcome
│   │   ├── QuickStats.tsx              # Key user metrics
│   │   ├── RecentActivity.tsx          # Recent actions
│   │   ├── Recommendations.tsx          # AI recommendations
│   │   └── QuickActions.tsx            # Common actions
│   │
│   ├── Profile/
│   │   ├── ProfileForm.tsx             # Edit profile form
│   │   ├── AvatarUpload.tsx            # Avatar management
│   │   ├── PreferencesPanel.tsx        # User preferences
│   │   ├── PrivacySettings.tsx         # Privacy controls
│   │   ├── NotificationSettings.tsx    # Notification prefs
│   │   └── AccountDeletion.tsx         # Delete account
│   │
│   ├── Subscription/
│   │   ├── CurrentPlan.tsx             # Current subscription
│   │   ├── PlanComparison.tsx          # Compare plans
│   │   ├── UpgradeModal.tsx            # Upgrade flow
│   │   ├── PaymentMethod.tsx           # Payment management
│   │   └── BillingHistory.tsx          # Invoice history
│   │
│   ├── [DomainSpecific]/
│   │   └── [Feature components]
│   │
│   └── shared/
│       ├── UserLayout.tsx              # Main user layout
│       ├── UserNavigation.tsx          # Navigation menu
│       ├── UserHeader.tsx              # Header with user menu
│       ├── MobileNav.tsx               # Mobile navigation
│       ├── NotificationBell.tsx        # Notification icon
│       ├── SearchBar.tsx               # Global search
│       ├── LoadingState.tsx            # Loading indicators
│       └── EmptyState.tsx              # Empty data states
│
├── pages/
│   ├── Auth/
│   │   ├── Login.tsx                   # User login
│   │   ├── Register.tsx                # Registration
│   │   ├── ForgotPassword.tsx          # Password reset
│   │   └── VerifyEmail.tsx             # Email verification
│   ├── Dashboard.tsx                    # User dashboard
│   ├── Profile/
│   │   ├── Edit.tsx                    # Edit profile
│   │   ├── Settings.tsx                # User settings
│   │   └── Subscription.tsx            # Manage subscription
│   └── [DomainSpecific]/
│       └── [Feature pages]
│
├── hooks/
│   ├── useAuth.ts                      # User authentication
│   ├── useProfile.ts                   # Profile management
│   ├── useSubscription.ts              # Subscription state
│   ├── useNotifications.ts             # Notification handling
│   └── [Domain hooks]
│
├── services/
│   ├── api.ts                          # API client
│   ├── auth.ts                         # Auth service
│   ├── storage.ts                      # Local storage
│   └── [Domain services]
│
├── store/
│   ├── authStore.ts                    # Auth state
│   ├── profileStore.ts                 # Profile state
│   ├── notificationStore.ts            # Notifications
│   └── [Domain stores]
│
└── utils/
    ├── validators.ts                   # Form validation
    ├── formatters.ts                   # Data formatting
    └── constants.ts                    # App constants
```

### 6. State Management (Zustand)

#### Admin Store Implementation
```typescript
// store/adminStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { AdminApi } from '@/services/adminApi';

interface AdminState {
  // Authentication
  admin: Admin | null;
  token: string | null;
  permissions: string[];
  isAuthenticated: boolean;
  
  // Dashboard
  metrics: DashboardMetrics | null;
  isLoadingMetrics: boolean;
  
  // User Management
  users: PaginatedResponse<User> | null;
  selectedUser: User | null;
  userFilters: UserFilters;
  
  // System
  systemHealth: SystemHealth | null;
  configurations: SystemConfig[];
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  checkPermission: (permission: string) => boolean;
  
  // Dashboard Actions
  fetchMetrics: (period?: string) => Promise<void>;
  subscribeToRealtimeMetrics: () => void;
  
  // User Management Actions
  fetchUsers: (page?: number) => Promise<void>;
  setUserFilters: (filters: Partial<UserFilters>) => void;
  fetchUserDetails: (userId: number) => Promise<void>;
  suspendUser: (userId: number, reason: string, duration?: number) => Promise<void>;
  bulkUserAction: (action: BulkAction, userIds: number[]) => Promise<void>;
  
  // System Actions
  fetchSystemHealth: () => Promise<void>;
  updateConfiguration: (key: string, value: any) => Promise<void>;
  toggleMaintenanceMode: (enabled: boolean) => Promise<void>;
}

export const useAdminStore = create<AdminState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        admin: null,
        token: null,
        permissions: [],
        isAuthenticated: false,
        metrics: null,
        isLoadingMetrics: false,
        users: null,
        selectedUser: null,
        userFilters: {
          search: '',
          status: null,
          subscription: null,
          sortBy: 'created_at',
          sortOrder: 'desc'
        },
        systemHealth: null,
        configurations: [],
        
        // Auth Actions
        login: async (credentials) => {
          try {
            const response = await AdminApi.login(credentials);
            set({
              admin: response.admin,
              token: response.token,
              permissions: response.permissions,
              isAuthenticated: true
            });
            
            // Set auth header for future requests
            AdminApi.setAuthToken(response.token);
            
            // Subscribe to real-time updates
            get().subscribeToRealtimeMetrics();
          } catch (error) {
            throw error;
          }
        },
        
        logout: () => {
          AdminApi.logout();
          set({
            admin: null,
            token: null,
            permissions: [],
            isAuthenticated: false
          });
        },
        
        checkPermission: (permission) => {
          const { permissions, admin } = get();
          if (admin?.role === 'super_admin') return true;
          return permissions.includes(permission);
        },
        
        // Dashboard Actions
        fetchMetrics: async (period = '30d') => {
          set({ isLoadingMetrics: true });
          try {
            const metrics = await AdminApi.getMetrics(period);
            set({ metrics, isLoadingMetrics: false });
          } catch (error) {
            set({ isLoadingMetrics: false });
            throw error;
          }
        },
        
        subscribeToRealtimeMetrics: () => {
          // WebSocket subscription for real-time updates
          window.Echo.private('admin.metrics')
            .listen('MetricsUpdated', (e: any) => {
              set({ metrics: e.metrics });
            });
        },
        
        // User Management Actions
        fetchUsers: async (page = 1) => {
          const { userFilters } = get();
          const users = await AdminApi.getUsers({
            page,
            ...userFilters
          });
          set({ users });
        },
        
        setUserFilters: (filters) => {
          set(state => ({
            userFilters: { ...state.userFilters, ...filters }
          }));
          // Refetch users with new filters
          get().fetchUsers(1);
        },
        
        fetchUserDetails: async (userId) => {
          const user = await AdminApi.getUserDetails(userId);
          set({ selectedUser: user });
        },
        
        suspendUser: async (userId, reason, duration) => {
          await AdminApi.suspendUser(userId, { reason, duration_days: duration });
          // Refresh user list
          await get().fetchUsers();
        },
        
        bulkUserAction: async (action, userIds) => {
          await AdminApi.bulkUserAction({ action, user_ids: userIds });
          // Refresh user list
          await get().fetchUsers();
        },
        
        // System Actions
        fetchSystemHealth: async () => {
          const health = await AdminApi.getSystemHealth();
          set({ systemHealth: health });
        },
        
        updateConfiguration: async (key, value) => {
          await AdminApi.updateConfig(key, value);
          // Refresh configurations
          const configs = await AdminApi.getConfigurations();
          set({ configurations: configs });
        },
        
        toggleMaintenanceMode: async (enabled) => {
          await get().updateConfiguration('maintenance_mode', enabled);
        }
      }),
      {
        name: 'admin-storage',
        partialize: (state) => ({
          token: state.token,
          admin: state.admin,
          permissions: state.permissions
        })
      }
    )
  )
);
```

#### User Store Implementation
```typescript
// store/userStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { UserApi } from '@/services/api';

interface UserState {
  // Authentication
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  
  // Profile
  profile: UserProfile | null;
  preferences: UserPreferences | null;
  
  // Notifications
  notifications: Notification[];
  unreadCount: number;
  
  // [Domain-specific state]
  
  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => void;
  register: (data: RegisterData) => Promise<void>;
  
  // Profile Actions
  fetchProfile: () => Promise<void>;
  updateProfile: (data: Partial<UserProfile>) => Promise<void>;
  uploadAvatar: (file: File) => Promise<void>;
  updatePreferences: (prefs: Partial<UserPreferences>) => Promise<void>;
  
  // Notification Actions
  fetchNotifications: () => Promise<void>;
  markNotificationRead: (id: number) => Promise<void>;
  markAllRead: () => Promise<void>;
  
  // [Domain-specific actions]
}

export const useUserStore = create<UserState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        user: null,
        token: null,
        isAuthenticated: false,
        profile: null,
        preferences: null,
        notifications: [],
        unreadCount: 0,
        
        // Auth Actions
        login: async (credentials) => {
          const response = await UserApi.login(credentials);
          set({
            user: response.user,
            token: response.token,
            isAuthenticated: true
          });
          
          // Set auth header
          UserApi.setAuthToken(response.token);
          
          // Fetch additional data
          await Promise.all([
            get().fetchProfile(),
            get().fetchNotifications()
          ]);
        },
        
        logout: () => {
          UserApi.logout();
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            profile: null,
            preferences: null,
            notifications: [],
            unreadCount: 0
          });
        },
        
        register: async (data) => {
          const response = await UserApi.register(data);
          set({
            user: response.user,
            isAuthenticated: false // Email verification required
          });
        },
        
        // Profile Actions
        fetchProfile: async () => {
          const response = await UserApi.getProfile();
          set({
            profile: response.profile,
            preferences: response.preferences
          });
        },
        
        updateProfile: async (data) => {
          const profile = await UserApi.updateProfile(data);
          set({ profile });
        },
        
        uploadAvatar: async (file) => {
          const { avatar_url } = await UserApi.uploadAvatar(file);
          set(state => ({
            profile: state.profile ? { ...state.profile, avatar_url } : null
          }));
        },
        
        updatePreferences: async (prefs) => {
          const preferences = await UserApi.updatePreferences(prefs);
          set({ preferences });
        },
        
        // Notification Actions
        fetchNotifications: async () => {
          const { notifications, unread_count } = await UserApi.getNotifications();
          set({ notifications, unreadCount: unread_count });
        },
        
        markNotificationRead: async (id) => {
          await UserApi.markNotificationRead(id);
          set(state => ({
            notifications: state.notifications.map(n =>
              n.id === id ? { ...n, read: true } : n
            ),
            unreadCount: Math.max(0, state.unreadCount - 1)
          }));
        },
        
        markAllRead: async () => {
          await UserApi.markAllNotificationsRead();
          set(state => ({
            notifications: state.notifications.map(n => ({ ...n, read: true })),
            unreadCount: 0
          }));
        }
      }),
      {
        name: 'user-storage',
        partialize: (state) => ({
          token: state.token,
          user: state.user
        })
      }
    )
  )
);
```

### 7. Real-time Architecture (Redis & Laravel Echo)

#### Redis Configuration and Usage
```php
// config/database.php
'redis' => [
    'client' => env('REDIS_CLIENT', 'phpredis'),
    
    'options' => [
        'cluster' => env('REDIS_CLUSTER', 'redis'),
        'prefix' => env('REDIS_PREFIX', Str::slug(env('APP_NAME', 'laravel'), '_').'_database_'),
    ],
    
    'default' => [
        'url' => env('REDIS_URL'),
        'host' => env('REDIS_HOST', '127.0.0.1'),
        'password' => env('REDIS_PASSWORD', null),
        'port' => env('REDIS_PORT', '6379'),
        'database' => env('REDIS_DB', '0'),
        'read_timeout' => 60,
    ],
    
    'cache' => [
        'url' => env('REDIS_URL'),
        'host' => env('REDIS_HOST', '127.0.0.1'),
        'password' => env('REDIS_PASSWORD', null),
        'port' => env('REDIS_PORT', '6379'),
        'database' => env('REDIS_CACHE_DB', '1'),
    ],
    
    'session' => [
        'url' => env('REDIS_URL'),
        'host' => env('REDIS_HOST', '127.0.0.1'),
        'password' => env('REDIS_PASSWORD', null),
        'port' => env('REDIS_PORT', '6379'),
        'database' => env('REDIS_SESSION_DB', '2'),
    ],
    
    'queue' => [
        'url' => env('REDIS_URL'),
        'host' => env('REDIS_HOST', '127.0.0.1'),
        'password' => env('REDIS_PASSWORD', null),
        'port' => env('REDIS_PORT', '6379'),
        'database' => env('REDIS_QUEUE_DB', '3'),
    ],
],
```

#### Redis Key Structure Documentation
```yaml
# Cache Keys - Database 1
cache:admin:dashboard:metrics:{period}      # TTL: 5 minutes
  Format: JSON object with user counts, revenue, activity metrics
  Example: {"total_users": 10000, "active_users": 7500, ...}

cache:admin:users:page:{page}:{filters_hash} # TTL: 10 minutes
  Format: Serialized paginated user data
  Invalidation: On any user update, bulk operations

cache:user:{user_id}:profile                # TTL: 1 hour
  Format: JSON user profile data
  Invalidation: On profile update

cache:user:{user_id}:preferences            # TTL: 24 hours
  Format: JSON preferences object
  Invalidation: On preference update

cache:food:search:{query_hash}              # TTL: 24 hours
  Format: Array of food items matching search
  
cache:food:item:{food_id}                   # TTL: 7 days
  Format: Complete food item data with nutrients

# Session Keys - Database 2
session:admin:{session_id}                  # TTL: 2 hours (sliding)
  Format: Admin session data
  Fields: admin_id, ip, user_agent, last_activity

session:user:{session_id}                   # TTL: 24 hours (sliding)
  Format: User session data
  Fields: user_id, ip, user_agent, last_activity

# Queue Keys - Database 3
queues:high                                 # List of high priority jobs
queues:default                              # List of default priority jobs
queues:low                                  # List of low priority jobs
queues:notifications                        # List of notification jobs
queues:failed                               # List of failed jobs

# Real-time Data - Database 0
realtime:active_users                       # Sorted set
  Score: Last activity timestamp
  Member: user_id
  Used for: Online user count, activity tracking

realtime:admin_presence                     # Set
  Members: admin_ids currently online
  Used for: Admin collaboration features

realtime:system_metrics                     # Hash
  Fields: cpu_usage, memory_usage, queue_depth, etc.
  Updated: Every 30 seconds by monitoring job

# Rate Limiting - Database 0
throttle:admin:{admin_id}:{action}         # String counter
  TTL: 1 minute
  Limit: 100 requests per minute

throttle:user:{user_id}:{action}           # String counter
  TTL: 1 minute
  Limit: 60 (premium) or 20 (free) per minute

throttle:ip:{ip_address}:{action}          # String counter
  TTL: 1 minute
  Limit: 10 requests per minute for anonymous

# Distributed Locks - Database 0
lock:report_generation:{report_id}          # Prevent duplicate generation
lock:user_export:{user_id}                  # Prevent concurrent exports
lock:bulk_operation:{operation_id}          # Coordinate bulk operations
```

#### Laravel Echo Server Configuration
```json
{
  "authHost": "http://localhost:8000",
  "authEndpoint": "/broadcasting/auth",
  "clients": [
    {
      "appId": "your-app-id",
      "key": "your-app-key"
    }
  ],
  "database": "redis",
  "databaseConfig": {
    "redis": {
      "port": "6379",
      "host": "localhost",
      "keyPrefix": "laravel_echo_"
    },
    "sqlite": {
      "databasePath": "/database/laravel-echo-server.sqlite"
    }
  },
  "devMode": false,
  "host": null,
  "port": "6001",
  "protocol": "http",
  "socketio": {},
  "secureOptions": 67108864,
  "sslCertPath": "",
  "sslKeyPath": "",
  "sslCertChainPath": "",
  "sslPassphrase": "",
  "subscribers": {
    "http": true,
    "redis": true
  },
  "apiOriginAllow": {
    "allowCors": true,
    "allowOrigin": "http://localhost:3000",
    "allowMethods": "GET, POST",
    "allowHeaders": "Origin, Content-Type, X-Auth-Token, X-Requested-With, Accept, Authorization, X-CSRF-TOKEN, X-Socket-Id"
  }
}
```

#### Broadcasting Configuration
```php
// config/broadcasting.php
'connections' => [
    'redis' => [
        'driver' => 'redis',
        'connection' => 'default',
        'queue' => env('REDIS_QUEUE', 'default'),
        'retry_after' => 90,
        'block_for' => null,
    ],
],

// routes/channels.php
// Admin Channels
Broadcast::channel('admin.dashboard', function ($user) {
    return $user instanceof \App\Models\Admin;
});

Broadcast::channel('admin.presence', function ($user) {
    if ($user instanceof \App\Models\Admin) {
        return [
            'id' => $user->id,
            'name' => $user->name,
            'role' => $user->role,
            'avatar' => $user->avatar_url
        ];
    }
});

Broadcast::channel('admin.user.{userId}', function ($admin, $userId) {
    return $admin instanceof \App\Models\Admin && 
           $admin->can('view', User::find($userId));
});

// User Channels
Broadcast::channel('user.{id}', function ($user, $id) {
    return (int) $user->id === (int) $id;
});

Broadcast::channel('chat.{roomId}', function ($user, $roomId) {
    if ($user->canJoinRoom($roomId)) {
        return [
            'id' => $user->id,
            'name' => $user->profile->display_name,
            'avatar' => $user->profile->avatar_url
        ];
    }
});
```

#### Event Examples
```php
// app/Events/Admin/SystemAlertEvent.php
namespace App\Events\Admin;

use Illuminate\Broadcasting\Channel;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Broadcasting\PrivateChannel;
use Illuminate\Contracts\Broadcasting\ShouldBroadcast;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class SystemAlertEvent implements ShouldBroadcast
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    public string $type;
    public string $severity;
    public string $message;
    public array $metadata;

    public function __construct($type, $severity, $message, $metadata = [])
    {
        $this->type = $type;
        $this->severity = $severity;
        $this->message = $message;
        $this->metadata = $metadata;
    }

    public function broadcastOn(): array
    {
        return [
            new PrivateChannel('admin.dashboard'),
            new PrivateChannel('admin.alerts')
        ];
    }

    public function broadcastAs(): string
    {
        return 'system.alert';
    }

    public function broadcastWith(): array
    {
        return [
            'id' => uniqid('alert_'),
            'type' => $this->type,
            'severity' => $this->severity,
            'message' => $this->message,
            'metadata' => $this->metadata,
            'timestamp' => now()->toIso8601String()
        ];
    }
}

// app/Events/User/RealtimeNotificationEvent.php
namespace App\Events\User;

use App\Models\User;
use App\Models\Notification;
use Illuminate\Broadcasting\InteractsWithSockets;
use Illuminate\Broadcasting\PrivateChannel;
use Illuminate\Contracts\Broadcasting\ShouldBroadcast;
use Illuminate\Foundation\Events\Dispatchable;
use Illuminate\Queue\SerializesModels;

class RealtimeNotificationEvent implements ShouldBroadcast
{
    use Dispatchable, InteractsWithSockets, SerializesModels;

    private User $user;
    private Notification $notification;

    public function __construct(User $user, Notification $notification)
    {
        $this->user = $user;
        $this->notification = $notification;
    }

    public function broadcastOn(): array
    {
        return [
            new PrivateChannel('user.' . $this->user->id)
        ];
    }

    public function broadcastAs(): string
    {
        return 'notification.new';
    }

    public function broadcastWith(): array
    {
        return [
            'id' => $this->notification->id,
            'type' => $this->notification->type,
            'title' => $this->notification->title,
            'message' => $this->notification->message,
            'action_url' => $this->notification->action_url,
            'created_at' => $this->notification->created_at->toIso8601String()
        ];
    }

    public function broadcastWhen(): bool
    {
        // Only broadcast if user has real-time notifications enabled
        return $this->user->preferences->real_time_notifications ?? true;
    }
}
```

### 8. Background Jobs & Queue Configuration

#### Queue Configuration
```php
// config/queue.php
'connections' => [
    'redis' => [
        'driver' => 'redis',
        'connection' => 'queue',
        'queue' => env('REDIS_QUEUE', 'default'),
        'retry_after' => 90,
        'block_for' => null,
        'after_commit' => false,
    ],
],

'failed' => [
    'driver' => env('QUEUE_FAILED_DRIVER', 'database-uuids'),
    'database' => env('DB_CONNECTION', 'mysql'),
    'table' => 'failed_jobs',
],

// Queue worker configuration (supervisor)
[program:laravel-worker-high]
process_name=%(program_name)s_%(process_num)02d
command=php /path/to/artisan queue:work redis --queue=high --sleep=3 --tries=3 --max-time=3600
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
user=www-data
numprocs=2
redirect_stderr=true
stdout_logfile=/path/to/logs/worker-high.log
stopwaitsecs=3600

[program:laravel-worker-default]
process_name=%(program_name)s_%(process_num)02d
command=php /path/to/artisan queue:work redis --queue=default --sleep=3 --tries=3 --max-time=3600
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
user=www-data
numprocs=4
redirect_stderr=true
stdout_logfile=/path/to/logs/worker-default.log
stopwaitsecs=3600
```

#### Job Examples
```php
// app/Jobs/Admin/GenerateAnalyticsReportJob.php
namespace App\Jobs\Admin;

use App\Models\Admin;
use App\Services\Admin\ReportGenerationService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldBeUnique;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

class GenerateAnalyticsReportJob implements ShouldQueue, ShouldBeUnique
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $timeout = 600; // 10 minutes
    public $tries = 3;
    public $backoff = [60, 300, 900]; // 1 min, 5 min, 15 min

    private Admin $requestedBy;
    private string $reportType;
    private array $parameters;
    private string $format;

    public function __construct(Admin $requestedBy, string $reportType, array $parameters, string $format = 'pdf')
    {
        $this->requestedBy = $requestedBy;
        $this->reportType = $reportType;
        $this->parameters = $parameters;
        $this->format = $format;
        
        $this->onQueue('low'); // Reports are low priority
    }

    public function uniqueId(): string
    {
        // Prevent duplicate report generation
        return sprintf(
            'report_%s_%s_%s',
            $this->reportType,
            md5(json_encode($this->parameters)),
            $this->requestedBy->id
        );
    }

    public function handle(ReportGenerationService $reportService): void
    {
        // Log start
        activity()
            ->causedBy($this->requestedBy)
            ->withProperties([
                'report_type' => $this->reportType,
                'parameters' => $this->parameters
            ])
            ->log('Report generation started');

        // Generate report
        $report = $reportService->generate(
            $this->reportType,
            $this->parameters,
            $this->format
        );

        // Store report file
        $path = $report->store('reports/' . date('Y/m'));

        // Create database record
        $reportRecord = \App\Models\Report::create([
            'admin_id' => $this->requestedBy->id,
            'type' => $this->reportType,
            'parameters' => $this->parameters,
            'format' => $this->format,
            'file_path' => $path,
            'file_size' => $report->size(),
            'generated_at' => now()
        ]);

        // Notify admin
        $this->requestedBy->notify(
            new \App\Notifications\Admin\ReportReadyNotification($reportRecord)
        );

        // Broadcast completion
        broadcast(new \App\Events\Admin\ReportGeneratedEvent(
            $this->requestedBy,
            $reportRecord
        ));
    }

    public function failed(\Throwable $exception): void
    {
        // Log failure
        \Log::error('Report generation failed', [
            'admin_id' => $this->requestedBy->id,
            'report_type' => $this->reportType,
            'error' => $exception->getMessage(),
            'trace' => $exception->getTraceAsString()
        ]);

        // Notify admin of failure
        $this->requestedBy->notify(
            new \App\Notifications\Admin\ReportFailedNotification(
                $this->reportType,
                $exception->getMessage()
            )
        );

        // Record failed attempt
        activity()
            ->causedBy($this->requestedBy)
            ->withProperties([
                'report_type' => $this->reportType,
                'error' => $exception->getMessage()
            ])
            ->log('Report generation failed');
    }
}

// app/Jobs/User/ProcessUserActionJob.php
namespace App\Jobs\User;

use App\Models\User;
use App\Services\User\ActionProcessorService;
use Illuminate\Bus\Batchable;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

class ProcessUserActionJob implements ShouldQueue
{
    use Batchable, Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 3;
    public $maxExceptions = 3;

    private User $user;
    private string $action;
    private array $data;

    public function __construct(User $user, string $action, array $data = [])
    {
        $this->user = $user;
        $this->action = $action;
        $this->data = $data;
        
        // Set queue based on action priority
        $this->onQueue($this->determineQueue());
    }

    private function determineQueue(): string
    {
        $highPriorityActions = ['payment', 'subscription_change', 'account_deletion'];
        
        if (in_array($this->action, $highPriorityActions)) {
            return 'high';
        }
        
        return 'default';
    }

    public function handle(ActionProcessorService $processor): void
    {
        // Process the action
        $result = $processor->process($this->user, $this->action, $this->data);
        
        // Log the action
        activity()
            ->causedBy($this->user)
            ->withProperties([
                'action' => $this->action,
                'data' => $this->data,
                'result' => $result
            ])
            ->log('User action processed');
        
        // Trigger follow-up actions if needed
        if ($result->hasFollowUpActions()) {
            foreach ($result->getFollowUpActions() as $followUpAction) {
                ProcessUserActionJob::dispatch(
                    $this->user,
                    $followUpAction['action'],
                    $followUpAction['data']
                )->delay(now()->addSeconds($followUpAction['delay'] ?? 0));
            }
        }
        
        // Update user statistics
        $this->user->increment('total_actions');
        $this->user->update(['last_active_at' => now()]);
    }

    public function failed(\Throwable $exception): void
    {
        \Log::error('User action processing failed', [
            'user_id' => $this->user->id,
            'action' => $this->action,
            'error' => $exception->getMessage()
        ]);
        
        // Notify user if critical action failed
        $criticalActions = ['payment', 'subscription_change'];
        
        if (in_array($this->action, $criticalActions)) {
            $this->user->notify(
                new \App\Notifications\User\ActionFailedNotification(
                    $this->action,
                    $exception->getMessage()
                )
            );
        }
    }
}
```

### 9. Security Architecture

#### Authentication & Authorization
```php
// app/Http/Middleware/AdminAuthenticate.php
namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;

class AdminAuthenticate
{
    public function handle(Request $request, Closure $next, ...$guards)
    {
        if (!$request->user() || !$request->user() instanceof \App\Models\Admin) {
            return response()->json(['message' => 'Unauthenticated'], 401);
        }

        // Check if admin account is active
        if ($request->user()->status !== 'active') {
            return response()->json(['message' => 'Account inactive'], 403);
        }

        // Check 2FA if enabled
        if ($request->user()->two_factor_enabled && !$request->session()->get('2fa_verified')) {
            return response()->json(['message' => '2FA verification required'], 403);
        }

        // Log admin activity
        activity()
            ->causedBy($request->user())
            ->withProperties([
                'ip' => $request->ip(),
                'user_agent' => $request->userAgent(),
                'path' => $request->path()
            ])
            ->log('Admin accessed: ' . $request->path());

        return $next($request);
    }
}

// app/Http/Middleware/RequirePermission.php
namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;

class RequirePermission
{
    public function handle(Request $request, Closure $next, string $permission)
    {
        if (!$request->user()->hasPermission($permission)) {
            return response()->json([
                'message' => 'Insufficient permissions',
                'required' => $permission
            ], 403);
        }

        return $next($request);
    }
}

// app/Policies/UserPolicy.php
namespace App\Policies;

use App\Models\Admin;
use App\Models\User;

class UserPolicy
{
    public function viewAny(Admin $admin): bool
    {
        return $admin->hasPermission('users.view');
    }

    public function view(Admin $admin, User $user): bool
    {
        return $admin->hasPermission('users.view');
    }

    public function update(Admin $admin, User $user): bool
    {
        return $admin->hasPermission('users.edit');
    }

    public function suspend(Admin $admin, User $user): bool
    {
        // Super admins can suspend anyone
        if ($admin->role === 'super_admin') {
            return true;
        }
        
        // Regular admins can't suspend premium users
        if ($admin->role === 'admin' && $user->isPremium()) {
            return false;
        }
        
        return $admin->hasPermission('users.suspend');
    }

    public function delete(Admin $admin, User $user): bool
    {
        return $admin->role === 'super_admin';
    }

    public function bulkAction(Admin $admin): bool
    {
        return $admin->hasPermission('users.bulk_action');
    }
}
```

#### API Security Configuration
```php
// config/cors.php
return [
    'paths' => ['api/*', 'broadcasting/auth'],
    'allowed_methods' => ['*'],
    'allowed_origins' => explode(',', env('CORS_ALLOWED_ORIGINS', 'http://localhost:3000')),
    'allowed_origins_patterns' => [],
    'allowed_headers' => ['*'],
    'exposed_headers' => ['X-Request-ID'],
    'max_age' => 0,
    'supports_credentials' => true,
];

// Rate Limiting Configuration
// app/Providers/RouteServiceProvider.php
protected function configureRateLimiting()
{
    // Admin API rate limits
    RateLimiter::for('admin-api', function (Request $request) {
        $admin = $request->user();
        
        if ($admin && $admin->role === 'super_admin') {
            return Limit::none(); // No limit for super admins
        }
        
        return [
            Limit::perMinute(100)->by($admin ? $admin->id : $request->ip()),
            Limit::perHour(1000)->by($admin ? $admin->id : $request->ip())
        ];
    });
    
    // User API rate limits
    RateLimiter::for('user-api', function (Request $request) {
        $user = $request->user();
        
        // Different limits based on subscription
        if ($user) {
            switch ($user->subscription_tier) {
                case 'enterprise':
                    return Limit::perMinute(1000)->by($user->id);
                case 'pro':
                    return Limit::perMinute(200)->by($user->id);
                case 'premium':
                    return Limit::perMinute(100)->by($user->id);
                case 'basic':
                    return Limit::perMinute(60)->by($user->id);
                default:
                    return Limit::perMinute(20)->by($user->id);
            }
        }
        
        // Unauthenticated requests
        return Limit::perMinute(10)->by($request->ip());
    });
    
    // Auth endpoints
    RateLimiter::for('auth', function (Request $request) {
        return [
            Limit::perMinute(5)->by($request->ip()),
            Limit::perHour(20)->by($request->ip())
        ];
    });
}

// Security Headers Middleware
namespace App\Http\Middleware;

class SecurityHeaders
{
    public function handle($request, Closure $next)
    {
        $response = $next($request);
        
        $response->headers->set('X-Content-Type-Options', 'nosniff');
        $response->headers->set('X-Frame-Options', 'DENY');
        $response->headers->set('X-XSS-Protection', '1; mode=block');
        $response->headers->set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
        $response->headers->set('Content-Security-Policy', $this->cspPolicy());
        $response->headers->set('Referrer-Policy', 'strict-origin-when-cross-origin');
        $response->headers->set('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
        
        return $response;
    }
    
    private function cspPolicy(): string
    {
        return "default-src 'self'; " .
               "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://fonts.googleapis.com; " .
               "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; " .
               "font-src 'self' https://fonts.gstatic.com; " .
               "img-src 'self' data: https:; " .
               "connect-src 'self' wss://localhost:6001 https://api.example.com;";
    }
}
```

### 10. Development Workflow

#### Local Development Setup
```bash
# 1. Clone repository
git clone [repository-url]
cd project-name

# 2. Install PHP dependencies
composer install

# 3. Install Node dependencies
npm install

# 4. Environment setup
cp .env.example .env.local
php artisan key:generate

# 5. Configure local environment variables
# Edit .env.local with your local settings:
APP_ENV=local
APP_DEBUG=true
APP_URL=http://localhost:8000

DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=project_local
DB_USERNAME=root
DB_PASSWORD=

REDIS_HOST=127.0.0.1
REDIS_PASSWORD=null
REDIS_PORT=6379

BROADCAST_DRIVER=redis
CACHE_DRIVER=redis
SESSION_DRIVER=redis
QUEUE_CONNECTION=redis

# 6. Create database
mysql -u root -p -e "CREATE DATABASE project_local CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 7. Run migrations and seeders
php artisan migrate
php artisan db:seed --class=LocalDevelopmentSeeder

# 8. Generate IDE helper files
php artisan ide-helper:generate
php artisan ide-helper:models --write

# 9. Build frontend assets
npm run build

# 10. Create storage symlink
php artisan storage:link
```

#### Running Local Services
```bash
# Terminal 1: PHP Development Server
php artisan serve

# Terminal 2: Vite Dev Server (Hot Module Replacement)
npm run dev

# Terminal 3: Queue Worker
php artisan queue:work --tries=3 --timeout=90

# Terminal 4: Laravel Echo Server
laravel-echo-server start

# Terminal 5: Schedule Runner (optional for local)
while true; do php artisan schedule:run; sleep 60; done

# Terminal 6: Redis Monitor (optional)
redis-cli monitor
```

#### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/JIRA-123-admin-bulk-export

# Make changes
git add .
git commit -m "feat(admin): implement bulk user export functionality

- Add bulk export UI in user management
- Create export job with progress tracking
- Support CSV and Excel formats
- Add email notification on completion

JIRA-123"

# Push and create PR
git push origin feature/JIRA-123-admin-bulk-export

# After review and approval
git checkout main
git pull origin main
git merge --no-ff feature/JIRA-123-admin-bulk-export
git push origin main
```

### 11. Testing Strategy

#### Test Structure
```
tests/
├── Unit/
│   ├── Admin/
│   │   ├── Services/
│   │   │   ├── UserManagementServiceTest.php
│   │   │   ├── ReportGenerationServiceTest.php
│   │   │   └── SystemMonitoringServiceTest.php
│   │   └── Jobs/
│   │       ├── GenerateReportJobTest.php
│   │       └── BulkOperationJobTest.php
│   └── User/
│       ├── Services/
│       │   ├── ProfileServiceTest.php
│       │   └── NotificationServiceTest.php
│       └── Models/
│           ├── UserTest.php
│           └── ProfileTest.php
│
├── Feature/
│   ├── Admin/
│   │   ├── Auth/
│   │   │   ├── AdminLoginTest.php
│   │   │   └── TwoFactorAuthTest.php
│   │   ├── UserManagement/
│   │   │   ├── UserCrudTest.php
│   │   │   ├── BulkOperationsTest.php
│   │   │   └── UserSuspensionTest.php
│   │   └── Analytics/
│   │       ├── DashboardMetricsTest.php
│   │       └── ReportGenerationTest.php
│   └── User/
│       ├── Auth/
│       │   ├── RegistrationTest.php
│       │   ├── LoginTest.php
│       │   └── PasswordResetTest.php
│       └── Profile/
│           ├── ProfileUpdateTest.php
│           └── AvatarUploadTest.php
│
├── Browser/ (Dusk)
│   ├── Admin/
│   │   ├── AdminDashboardTest.php
│   │   └── UserManagementFlowTest.php
│   └── User/
│       ├── OnboardingFlowTest.php
│       └── ProfileCompletionTest.php
│
└── TestCase.php
```

#### Example Test Implementation
```php
// tests/Feature/Admin/UserManagement/BulkOperationsTest.php
namespace Tests\Feature\Admin\UserManagement;

use Tests\TestCase;
use App\Models\Admin;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Queue;

class BulkOperationsTest extends TestCase
{
    use RefreshDatabase;

    private Admin $superAdmin;
    private Admin $regularAdmin;

    protected function setUp(): void
    {
        parent::setUp();
        
        $this->superAdmin = Admin::factory()->create(['role' => 'super_admin']);
        $this->regularAdmin = Admin::factory()->create(['role' => 'admin']);
    }

    /** @test */
    public function admin_can_bulk_export_users()
    {
        // Arrange
        Queue::fake();
        $users = User::factory()->count(5)->create();
        $userIds = $users->pluck('id')->toArray();

        // Act
        $response = $this->actingAs($this->regularAdmin, 'admin')
            ->postJson('/api/admin/users/bulk', [
                'action' => 'export',
                'user_ids' => $userIds,
                'parameters' => [
                    'format' => 'csv',
                    'fields' => ['email', 'name', 'created_at']
                ]
            ]);

        // Assert
        $response->assertOk()
            ->assertJsonStructure([
                'success',
                'job_id',
                'affected_users',
                'status_url'
            ]);

        Queue::assertPushed(\App\Jobs\Admin\BulkExportUsersJob::class, function ($job) use ($userIds) {
            return count($job->userIds) === 5 && 
                   $job->format === 'csv';
        });
    }

    /** @test */
    public function regular_admin_cannot_bulk_delete_premium_users()
    {
        // Arrange
        $premiumUsers = User::factory()
            ->count(3)
            ->premium()
            ->create();

        // Act
        $response = $this->actingAs($this->regularAdmin, 'admin')
            ->postJson('/api/admin/users/bulk', [
                'action' => 'delete',
                'user_ids' => $premiumUsers->pluck('id')->toArray()
            ]);

        // Assert
        $response->assertForbidden();
        
        foreach ($premiumUsers as $user) {
            $this->assertDatabaseHas('users', [
                'id' => $user->id,
                'deleted_at' => null
            ]);
        }
    }

    /** @test */
    public function bulk_operation_validates_user_ids()
    {
        // Act
        $response = $this->actingAs($this->superAdmin, 'admin')
            ->postJson('/api/admin/users/bulk', [
                'action' => 'export',
                'user_ids' => ['invalid', 999999]
            ]);

        // Assert
        $response->assertUnprocessable()
            ->assertJsonValidationErrors(['user_ids.0', 'user_ids.1']);
    }
}
```

### 12. Performance Optimizations

#### Database Query Optimization
```php
// app/Services/Admin/UserManagementService.php
class UserManagementService
{
    public function getUsersWithStats(array $filters = [])
    {
        return User::query()
            // Eager load relationships
            ->with([
                'profile:id,user_id,first_name,last_name,avatar_url',
                'latestSubscription:id,user_id,tier,expires_at'
            ])
            // Add computed columns
            ->withCount(['orders', 'activities'])
            ->withSum('orders', 'total_amount')
            ->withAvg('reviews', 'rating')
            // Apply filters
            ->when($filters['search'] ?? null, function ($query, $search) {
                $query->where(function ($q) use ($search) {
                    $q->where('email', 'like', "%{$search}%")
                      ->orWhereHas('profile', function ($q) use ($search) {
                          $q->where('first_name', 'like', "%{$search}%")
                            ->orWhere('last_name', 'like', "%{$search}%");
                      });
                });
            })
            ->when($filters['status'] ?? null, function ($query, $status) {
                $query->where('status', $status);
            })
            // Use index
            ->orderBy('created_at', 'desc')
            ->paginate(20);
    }

    public function getDashboardMetrics(): array
    {
        return Cache::remember('admin:dashboard:metrics', 300, function () {
            // Use database aggregation instead of loading all records
            $userStats = DB::table('users')
                ->selectRaw('
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN status = "active" THEN 1 END) as active_users,
                    COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as new_users,
                    COUNT(CASE WHEN subscription_tier != "free" THEN 1 END) as paid_users
                ')
                ->first();

            $revenueStats = DB::table('payments')
                ->whereDate('created_at', '>=', now()->subDays(30))
                ->selectRaw('
                    SUM(amount) as total_revenue,
                    AVG(amount) as average_order_value,
                    COUNT(DISTINCT user_id) as paying_customers
                ')
                ->first();

            return [
                'users' => $userStats,
                'revenue' => $revenueStats,
                'calculated_at' => now()
            ];
        });
    }
}

// Avoid N+1 queries
// Bad:
$users = User::all();
foreach ($users as $user) {
    echo $user->profile->full_name; // N+1 query
}

// Good:
$users = User::with('profile')->get();
foreach ($users as $user) {
    echo $user->profile->full_name; // No additional queries
}

// Better: Only load what you need
$users = User::with('profile:id,user_id,first_name,last_name')
    ->select('id', 'email', 'status')
    ->get();
```

#### Caching Strategy
```php
// app/Traits/CacheableModel.php
namespace App\Traits;

trait CacheableModel
{
    public static function bootCacheableModel()
    {
        static::saved(function ($model) {
            $model->clearCache();
        });

        static::deleted(function ($model) {
            $model->clearCache();
        });
    }

    public function clearCache()
    {
        $tags = $this->getCacheTags();
        Cache::tags($tags)->flush();
    }

    public function getCacheTags(): array
    {
        return [
            $this->getTable(),
            $this->getTable() . ':' . $this->getKey()
        ];
    }

    public function remember(string $key, $ttl, Closure $callback)
    {
        return Cache::tags($this->getCacheTags())
            ->remember($key, $ttl, $callback);
    }
}

// Usage in model
class User extends Authenticatable
{
    use CacheableModel;

    public function getCachedProfile()
    {
        return $this->remember("user:{$this->id}:profile", 3600, function () {
            return $this->profile()->with('preferences')->first();
        });
    }
}
```

### 13. Monitoring & Logging

#### Structured Logging
```php
// config/logging.php
'channels' => [
    'stack' => [
        'driver' => 'stack',
        'channels' => ['daily', 'slack'],
        'ignore_exceptions' => false,
    ],

    'admin' => [
        'driver' => 'daily',
        'path' => storage_path('logs/admin/admin.log'),
        'level' => env('LOG_LEVEL', 'info'),
        'days' => 30,
        'permission' => 0664,
    ],

    'user_activity' => [
        'driver' => 'daily',
        'path' => storage_path('logs/user/activity.log'),
        'level' => 'info',
        'days' => 7,
    ],

    'security' => [
        'driver' => 'daily',
        'path' => storage_path('logs/security/security.log'),
        'level' => 'warning',
        'days' => 90,
    ],

    'performance' => [
        'driver' => 'daily',
        'path' => storage_path('logs/performance/performance.log'),
        'level' => 'info',
        'days' => 14,
    ],

    'api' => [
        'driver' => 'daily',
        'path' => storage_path('logs/api/api.log'),
        'level' => 'info',
        'days' => 30,
        'tap' => [App\Logging\ApiLogFormatter::class],
    ],
];

// app/Logging/ApiLogFormatter.php
namespace App\Logging;

use Monolog\Formatter\LineFormatter;
use Monolog\Handler\RotatingFileHandler;

class ApiLogFormatter
{
    public function __invoke($logger)
    {
        foreach ($logger->getHandlers() as $handler) {
            if ($handler instanceof RotatingFileHandler) {
                $handler->setFormatter(new LineFormatter(
                    "[%datetime%] %channel%.%level_name%: %message% %context% %extra%\n",
                    'Y-m-d H:i:s',
                    true,
                    true
                ));
            }
        }
    }
}

// Usage examples
Log::channel('admin')->info('Admin action performed', [
    'admin_id' => $admin->id,
    'action' => 'user.suspended',
    'target_user' => $userId,
    'reason' => $reason,
    'ip' => request()->ip(),
    'user_agent' => request()->userAgent()
]);

Log::channel('security')->warning('Failed login attempt', [
    'email' => $request->email,
    'ip' => $request->ip(),
    'attempts' => $attempts
]);

Log::channel('performance')->info('Slow query detected', [
    'query' => $sql,
    'bindings' => $bindings,
    'duration' => $duration,
    'threshold' => 1000 // ms
]);
```

#### Application Monitoring
```php
// app/Services/Monitoring/MetricsCollector.php
namespace App\Services\Monitoring;

use Prometheus\CollectorRegistry;
use Prometheus\Storage\Redis;

class MetricsCollector
{
    private CollectorRegistry $registry;

    public function __construct()
    {
        $this->registry = new CollectorRegistry(new Redis([
            'host' => config('database.redis.default.host'),
            'port' => config('database.redis.default.port'),
            'database' => 4 // Separate DB for metrics
        ]));
    }

    public function recordHttpRequest(string $method, string $route, int $status, float $duration)
    {
        $histogram = $this->registry->getOrRegisterHistogram(
            'app',
            'http_request_duration_seconds',
            'HTTP request latencies',
            ['method', 'route', 'status'],
            [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        );

        $histogram->observe($duration, [$method, $route, (string) $status]);

        $counter = $this->registry->getOrRegisterCounter(
            'app',
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'route', 'status']
        );

        $counter->inc([$method, $route, (string) $status]);
    }

    public function recordDatabaseQuery(string $type, float $duration)
    {
        $histogram = $this->registry->getOrRegisterHistogram(
            'app',
            'database_query_duration_seconds',
            'Database query latencies',
            ['type']
        );

        $histogram->observe($duration, [$type]);
    }

    public function recordQueueJob(string $queue, string $job, string $status, float $duration)
    {
        $histogram = $this->registry->getOrRegisterHistogram(
            'app',
            'queue_job_duration_seconds',
            'Queue job processing time',
            ['queue', 'job', 'status']
        );

        $histogram->observe($duration, [$queue, $job, $status]);
    }

    public function getMetrics(): string
    {
        $renderer = new \Prometheus\RenderTextFormat();
        return $renderer->render($this->registry->getMetricFamilySamples());
    }
}
```

### 14. Deployment & DevOps

#### Environment Configuration
```bash
# .env.production
APP_NAME="Application Name"
APP_ENV=production
APP_KEY=base64:...
APP_DEBUG=false
APP_URL=https://app.example.com

LOG_CHANNEL=stack
LOG_DEPRECATIONS_CHANNEL=null
LOG_LEVEL=warning

DB_CONNECTION=mysql
DB_HOST=db-cluster.example.com
DB_PORT=3306
DB_DATABASE=app_production
DB_USERNAME=app_user
DB_PASSWORD=${DB_PASSWORD}
DB_REPLICA_HOST=db-replica.example.com

BROADCAST_DRIVER=redis
CACHE_DRIVER=redis
FILESYSTEM_DISK=s3
QUEUE_CONNECTION=redis
SESSION_DRIVER=redis

REDIS_HOST=redis-cluster.example.com
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_PORT=6379

MAIL_MAILER=ses
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION=us-east-1

SENTRY_LARAVEL_DSN=${SENTRY_DSN}
SENTRY_ENVIRONMENT=production
```

#### Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

echo "Starting deployment..."

# Put application in maintenance mode
php artisan down --message="System upgrade in progress" --retry=60

# Pull latest code
git pull origin main

# Install dependencies
composer install --no-interaction --prefer-dist --optimize-autoloader --no-dev

# Run database migrations
php artisan migrate --force

# Clear and cache configurations
php artisan config:cache
php artisan route:cache
php artisan view:cache
php artisan event:cache

# Restart queue workers
php artisan queue:restart

# Clear OPcache
php artisan opcache:clear

# Build frontend assets
npm ci --production
npm run build

# Bring application back online
php artisan up

echo "Deployment completed successfully!"
```

### 15. Future Considerations

#### Planned Features
1. **GraphQL API** - For mobile app flexibility
2. **Microservices Migration** - Extract heavy processing services
3. **AI/ML Integration** - Predictive analytics and recommendations
4. **Real-time Collaboration** - Multi-user editing capabilities
5. **Advanced Analytics** - Business intelligence dashboards
6. **API Gateway** - Centralized API management
7. **Event Sourcing** - Complete audit trail
8. **CQRS Implementation** - Separate read/write models

#### Technical Debt
1. Migrate remaining jQuery components to React
2. Implement 100% TypeScript coverage
3. Add comprehensive E2E test suite
4. Optimize database indices based on query patterns
5. Implement API versioning strategy
6. Add request/response validation middleware
7. Create comprehensive API documentation
8. Implement circuit breakers for external services

#### Scalability Roadmap
1. **Phase 1**: Implement read replicas for reporting
2. **Phase 2**: Add Redis Cluster for caching
3. **Phase 3**: Implement database sharding
4. **Phase 4**: Extract services to microservices
5. **Phase 5**: Implement Kubernetes orchestration
6. **Phase 6**: Multi-region deployment

---

This architecture document serves as the technical blueprint for the application. It should be updated as the system evolves and new patterns are adopted.
"""


class ArchitectureEnhancer:
    """Enhances ARCHITECTURE.md files to comprehensive documentation standards."""
    
    def __init__(self):
        """Initialize the architecture enhancer."""
        self.logger = logging.getLogger(__name__)
        self.template = COMPREHENSIVE_ARCHITECTURE_TEMPLATE
        self.enhanced_count = 0
        self.skipped_count = 0
        self.errors = []
        
        # Initialize AI client
        try:
            from scripts.http_ai_client import HTTPAIClient
            self.ai_client = HTTPAIClient()
        except ImportError:
            self.logger.error("Could not import HTTPAIClient")
            self.ai_client = None
            
    def extract_section(self, content: str, section_name: str) -> str:
        """Extract a specific section from markdown content."""
        lines = content.split('\n')
        in_section = False
        section_content = []
        section_level = 0
        
        for line in lines:
            # Check if this is our section
            if section_name.lower() in line.lower() and line.strip().startswith('#'):
                in_section = True
                section_level = len(line.split()[0]) if line.split() else 0
                section_content.append(line)
                continue
                
            # If in section, check if we've hit a new section at same or higher level
            if in_section:
                if line.strip().startswith('#'):
                    current_level = len(line.split()[0]) if line.split() else 0
                    if current_level <= section_level:
                        break
                section_content.append(line)
        
        return '\n'.join(section_content)
    
    def analyze_existing_architecture(self, content: str) -> Dict[str, Any]:
        """Analyze existing architecture file to identify what's present and missing."""
        analysis = {
            'has_overview': bool(re.search(r'^#{1,2}\s+Overview', content, re.MULTILINE)),
            'has_core_entities': bool(re.search(r'^#{1,3}\s+Core Entities', content, re.MULTILINE)),
            'has_design_system': bool(re.search(r'^#{1,3}\s+Design System', content, re.MULTILINE)),
            'has_admin_system': bool(re.search(r'Admin.*Controller|admin.*api', content, re.IGNORECASE)),
            'has_user_system': bool(re.search(r'User.*Controller|user.*api', content, re.IGNORECASE)),
            'has_api_docs': bool(re.search(r'GET\s+/api/|POST\s+/api/', content)),
            'has_database_schema': bool(re.search(r'CREATE TABLE', content, re.IGNORECASE)),
            'has_redis_config': bool(re.search(r'Redis|cache:', content, re.IGNORECASE)),
            'has_echo_config': bool(re.search(r'Laravel Echo|Echo\.|broadcasting', content, re.IGNORECASE)),
            'has_security': bool(re.search(r'Security|Authentication|Authorization', content, re.IGNORECASE)),
            'has_frontend_structure': bool(re.search(r'components/|Frontend.*Structure', content, re.IGNORECASE)),
            'total_length': len(content),
            'missing_sections': []
        }
        
        # Identify missing critical sections
        required_sections = {
            'admin_system': 'Admin System Architecture',
            'user_system': 'User System Architecture', 
            'admin_api': 'Admin API Endpoints',
            'user_api': 'User API Endpoints',
            'database_schema': 'Database Schema with Admin tables',
            'redis_config': 'Redis Configuration',
            'echo_config': 'Laravel Echo Configuration',
            'security': 'Security Architecture',
            'frontend_structure': 'Frontend Architecture (Admin/User)'
        }
        
        for key, description in required_sections.items():
            if not analysis.get(f'has_{key}', False):
                analysis['missing_sections'].append(description)
                
        return analysis
    
    async def enhance_single_project(self, project_path: str) -> Dict[str, Any]:
        """Enhance a single project's ARCHITECTURE.md file."""
        try:
            project_path = Path(project_path)
            if not project_path.exists():
                return {
                    'success': False,
                    'error': f'Project path not found: {project_path}'
                }
            
            arch_path = project_path / 'ARCHITECTURE.md'
            if not arch_path.exists():
                return {
                    'success': False,
                    'error': f'No ARCHITECTURE.md found in {project_path}',
                    'suggestion': 'Run the documentation generator first to create initial documentation'
                }
            
            # Read existing content
            with open(arch_path, 'r') as f:
                existing_content = f.read()
            
            # Analyze what's missing
            analysis = self.analyze_existing_architecture(existing_content)
            
            if not analysis['missing_sections']:
                self.skipped_count += 1
                return {
                    'success': True,
                    'message': 'Architecture already comprehensive',
                    'analysis': analysis
                }
            
            # Extract sections to preserve
            core_entities = self.extract_section(existing_content, "Core Entities")
            design_system = self.extract_section(existing_content, "Design System")
            
            # Get project info
            project_name = project_path.name
            readme_path = project_path / 'README.md'
            readme_content = ""
            if readme_path.exists():
                with open(readme_path, 'r') as f:
                    readme_content = f.read()[:2000]
            
            # Generate enhanced architecture
            enhanced_content = await self._generate_enhanced_architecture(
                project_name=project_name,
                existing_content=existing_content,
                core_entities=core_entities,
                design_system=design_system,
                readme_content=readme_content,
                missing_sections=analysis['missing_sections']
            )
            
            if not enhanced_content:
                return {
                    'success': False,
                    'error': 'Failed to generate enhanced architecture'
                }
            
            # Validate enhancement
            validation = self.validate_architecture_completeness(enhanced_content)
            
            if validation['quality_score'] < 80:
                return {
                    'success': False,
                    'error': 'Generated architecture does not meet quality standards',
                    'validation': validation
                }
            
            # Create backup
            backup_path = arch_path.with_suffix('.md.backup')
            with open(backup_path, 'w') as f:
                f.write(existing_content)
            
            # Save enhanced version
            with open(arch_path, 'w') as f:
                f.write(enhanced_content)
            
            self.enhanced_count += 1
            
            return {
                'success': True,
                'project': project_name,
                'enhanced': True,
                'backup_created': str(backup_path),
                'analysis_before': analysis,
                'validation_after': validation,
                'improvements': len(analysis['missing_sections'])
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing {project_path}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    async def enhance_multiple_projects(self, 
                                      base_path: str = None,
                                      project_names: List[str] = None) -> Dict[str, Any]:
        """Enhance ARCHITECTURE.md files for multiple projects."""
        try:
            if not base_path:
                base_path = Path.cwd() / "local_repos/repositories/CodeWebMobile-AI"
            else:
                base_path = Path(base_path)
            
            if not base_path.exists():
                return {
                    'success': False,
                    'error': f'Base path not found: {base_path}'
                }
            
            # Find projects to enhance
            if project_names:
                projects = [base_path / name for name in project_names if (base_path / name).exists()]
            else:
                # Find all projects with ARCHITECTURE.md
                projects = [p for p in base_path.iterdir() 
                          if p.is_dir() and (p / 'ARCHITECTURE.md').exists()]
            
            if not projects:
                return {
                    'success': False,
                    'error': 'No projects found with ARCHITECTURE.md files'
                }
            
            results = {
                'total_projects': len(projects),
                'enhanced': 0,
                'skipped': 0,
                'failed': 0,
                'details': []
            }
            
            for project in projects:
                self.logger.info(f"Enhancing {project.name}...")
                result = await self.enhance_single_project(str(project))
                
                if result['success']:
                    if result.get('enhanced'):
                        results['enhanced'] += 1
                    else:
                        results['skipped'] += 1
                else:
                    results['failed'] += 1
                    self.errors.append({
                        'project': project.name,
                        'error': result.get('error')
                    })
                
                results['details'].append({
                    'project': project.name,
                    'result': result
                })
            
            results['success'] = results['failed'] == 0
            results['errors'] = self.errors
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch enhancement: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_enhanced_architecture(self, 
                                            project_name: str,
                                            existing_content: str,
                                            core_entities: str,
                                            design_system: str,
                                            readme_content: str,
                                            missing_sections: List[str]) -> Optional[str]:
        """Generate enhanced architecture content using AI."""
        if not self.ai_client:
            self.logger.error("AI client not available")
            return None
        
        try:
            # Build prompt
            preserved_sections = ""
            if core_entities:
                preserved_sections += f"\n\nPRESERVE THIS CORE ENTITIES SECTION:\n{core_entities}\n"
            if design_system:
                preserved_sections += f"\n\nPRESERVE THIS DESIGN SYSTEM SECTION:\n{design_system}\n"
            
            missing_list = "\n".join(f"- {section}" for section in missing_sections)
            
            prompt = f"""Enhance this ARCHITECTURE.md file to meet comprehensive documentation standards.

Project Name: {project_name}

EXISTING ARCHITECTURE (first 3000 chars):
{existing_content[:3000]}

README CONTEXT:
{readme_content}

{preserved_sections}

MISSING SECTIONS THAT MUST BE ADDED:
{missing_list}

EXAMPLE OF COMPREHENSIVE ARCHITECTURE:
{self.template[:8000]}

Please create an enhanced ARCHITECTURE.md that:
1. PRESERVES the existing Core Entities and Design System sections exactly as shown above
2. ADDS all missing sections listed above with the same level of detail as the example
3. Includes complete Admin/User system separation throughout:
   - Separate Admin and User controllers with method descriptions
   - Complete API endpoint documentation with request/response examples
   - Separate frontend structures for Admin and User portals
   - Role-based security architecture
4. Provides actual code examples and configurations
5. Includes Redis and Laravel Echo configuration for real-time features
6. Shows SQL schemas with proper indexes and relationships
7. Documents background jobs and queue configuration
8. Adds security architecture with middleware and policies

The enhanced documentation should be at least 25,000 characters and immediately useful for developers.
Format as a complete ARCHITECTURE.md file starting with # {project_name} Architecture."""

            response = await self.ai_client.generate_enhanced_response(
                prompt=prompt,
                model="gemini"
            )
            
            enhanced_content = response.get('content', '')
            
            # Ensure it starts properly
            if not enhanced_content.strip().startswith('#'):
                enhanced_content = f"# {project_name} Architecture\n\n{enhanced_content}"
            
            return enhanced_content
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced architecture: {e}")
            return None
    
    def validate_architecture_completeness(self, content: str) -> Dict[str, Any]:
        """Comprehensive validation of architecture documentation."""
        
        required_sections = {
            'core_entities': 'Core Entities',
            'design_system': 'Design System',
            'admin_system': 'Admin.*Controller|AdminSystem',
            'user_system': 'User.*Controller|UserSystem',
            'admin_api': r'/api/admin/',
            'user_api': r'/api/user/',
            'database_schema': 'CREATE TABLE',
            'redis_config': 'Redis.*Configuration|cache:',
            'echo_config': 'Laravel Echo|Echo\.',
            'security': 'Security.*Architecture|Authentication.*Authorization',
            'background_jobs': 'Jobs.*Queue|Background.*Processing',
            'state_management': 'State.*Management|Zustand'
        }
        
        quality_metrics = {
            'has_code_examples': bool(re.search(r'```', content)),
            'has_api_examples': bool(re.search(r'Body:|Response:', content)),
            'has_sql_indexes': bool(re.search(r'INDEX|KEY', content, re.IGNORECASE)),
            'has_frontend_structure': bool(re.search(r'components/|\.tsx|\.jsx', content)),
            'has_rate_limiting': bool(re.search(r'Rate Limit:|RateLimiter', content)),
            'has_real_examples': bool(re.search(r'POST\s+/api/|GET\s+/api/', content)),
            'has_error_handling': bool(re.search(r'try.*catch|error.*handling', content, re.IGNORECASE)),
            'has_testing': bool(re.search(r'test|spec|Test', content, re.IGNORECASE)),
            'sufficient_length': len(content) > 25000,
            'has_future_considerations': bool(re.search(r'Future.*Considerations|Roadmap', content, re.IGNORECASE))
        }
        
        # Check for missing sections
        missing_sections = []
        for key, pattern in required_sections.items():
            if not re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                missing_sections.append(key)
        
        # Calculate quality score
        quality_score = (sum(quality_metrics.values()) / len(quality_metrics)) * 100
        
        # Adjust score based on missing sections
        section_penalty = len(missing_sections) * 5
        quality_score = max(0, quality_score - section_penalty)
        
        return {
            'missing_sections': missing_sections,
            'quality_score': quality_score,
            'quality_metrics': quality_metrics,
            'character_count': len(content),
            'meets_standards': quality_score >= 80 and len(missing_sections) == 0
        }


# Tool function to be called by the AI assistant
async def enhance_architecture_documentation(
    project_path: str = None,
    enhance_all: bool = False,
    base_path: str = None,
    project_names: List[str] = None
) -> Dict[str, Any]:
    """
    Enhance ARCHITECTURE.md files to meet comprehensive documentation standards.
    
    Args:
        project_path: Path to specific project to enhance (optional)
        enhance_all: Enhance all projects in base_path (default: False)
        base_path: Base directory containing projects (optional)
        project_names: List of specific project names to enhance (optional)
        
    Returns:
        Dictionary with enhancement results including:
        - success: Whether enhancement succeeded
        - enhanced: Number of files enhanced
        - skipped: Number of files already comprehensive
        - failed: Number of failures
        - details: Detailed results for each project
        
    Examples:
        # Enhance single project
        enhance_architecture_documentation(project_path="/path/to/project")
        
        # Enhance specific projects
        enhance_architecture_documentation(
            base_path="/path/to/repos",
            project_names=["project1", "project2"]
        )
        
        # Enhance all projects
        enhance_architecture_documentation(enhance_all=True)
    """
    enhancer = ArchitectureEnhancer()
    
    if enhance_all or project_names:
        return await enhancer.enhance_multiple_projects(base_path, project_names)
    elif project_path:
        return await enhancer.enhance_single_project(project_path)
    else:
        return {
            "success": False,
            "error": "Specify project_path, set enhance_all=True, or provide project_names"
        }


# For testing
if __name__ == "__main__":
    async def test_enhancement():
        """Test the enhancement tool."""
        # Test single project
        result = await enhance_architecture_documentation(
            project_path=str(Path.cwd() / "local_repos" / "repositories"/CodeWebMobile-AI/micronutrient-optimizer"
        )
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_enhancement())