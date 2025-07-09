# System Architecture
## Executive Summary
**Project**: inventory-manager - AI-powered inventory management for small retailers
**Problem Domain**: Small retailers lose 20% of sales due to stockouts
**Solution Type**: AI-powered inventory management platform
**Architecture Pattern**: Inertia.js Monolith with Event-Driven AI Services
**Deployment Target**: Local Development → Cloud Production
## System Context
### Problem Analysis
**Problem**: Small retailers struggle with inventory across multiple channels
**Current Solutions**: Excel spreadsheets, basic POS systems
**Our Approach**: AI predictions based on local patterns and events
**Key Innovation**: Machine learning that adapts to local market conditions
### User Journey
- Discovery → Users find us through search and retail forums
- Onboarding → Simple signup with guided inventory import
- First Value → See AI predictions within 2 minutes
- Engagement → Daily inventory checks and order suggestions
- Retention → Users stay for accuracy and time savings

## Business Rules
### Core Business Rules
#### Inventory Management
**Stock Levels**:
- minimum_threshold
- reorder_point
- maximum_capacity

**Ordering**:
- lead_time: supplier_specific
- minimum_order: product_specific

**Pricing**:
- markup_rules: category_based
- discount_limits: role_based

#### User Management
**Roles**:
- owner
- manager
- staff

**Permissions**:
- owner: all
- manager: inventory_and_reports
- staff: view_only

### Validation Rules
```json
{
  "product_sku": {
    "type": "string",
    "required": true,
    "pattern": "^[A-Z0-9-]+$",
    "unique": true
  },
  "stock_quantity": {
    "type": "integer",
    "required": true,
    "min": 0,
    "max": 999999
  }
}
```
### Edge Cases
**Supplier out of stock**:
- Handling: Suggest alternative suppliers or products
- Recovery: Automated reorder when available

**Sudden demand spike**:
- Handling: Alert user, suggest emergency reorder
- Recovery: Adjust AI predictions for future

## State Machines
### Order Lifecycle
Manages the lifecycle of inventory orders

**States**:
- **draft**: Order being prepared
- **submitted**: Order sent to supplier
- **confirmed**: Supplier confirmed order
- **shipped**: Order in transit
- **received**: Order delivered
- **completed**: Order processed into inventory

**Transitions**:
```
draft → submitted (trigger: submit_order) [if: has_items, supplier_active]
submitted → confirmed (trigger: supplier_confirmation)
```
## Domain Models
### Product
**Purpose**: Represents inventory items

**Fields**:
- `id` (bigint): Primary key (required)
- `sku` (string): Stock keeping unit (required, unique)
- `name` (string): Product name (required, max: 255)
- `current_stock` (integer): Current stock level (required, min: 0)

**Relationships**:
- belongsTo:Category:category_id
- hasMany:StockMovement
- belongsToMany:Supplier:through:product_suppliers

**Key Methods**:
- `needsReorder()`: Check if reorder needed → boolean
- `calculateReorderQuantity()`: AI-based reorder suggestion → integer

### Supplier
**Purpose**: Represents product suppliers

**Fields**:
- `id` (bigint): Primary key
- `name` (string): Supplier name
- `lead_time_days` (integer): Typical delivery time

**Relationships**:
- belongsToMany:Product:through:product_suppliers
- hasMany:PurchaseOrder

### PurchaseOrder
**Purpose**: Represents orders to suppliers

**Fields**:
- `id` (bigint): Primary key
- `supplier_id` (bigint): Foreign key to suppliers
- `status` (enum): Order status
- `total_amount` (decimal): Order total

**Relationships**:
- belongsTo:Supplier
- hasMany:PurchaseOrderItem

### StockMovement
**Purpose**: Tracks all inventory changes

**Fields**:
- `id` (bigint): Primary key
- `product_id` (bigint): Foreign key to products
- `quantity` (integer): Positive for in, negative for out
- `reason` (string): Reason for movement

**Relationships**:
- belongsTo:Product
- belongsTo:User

### AIForecast
**Purpose**: Stores AI predictions for demand

**Fields**:
- `id` (bigint): Primary key
- `product_id` (bigint): Foreign key to products
- `forecast_date` (date): Date of forecast
- `predicted_demand` (integer): Predicted units
- `confidence_score` (decimal): AI confidence 0-1

**Relationships**:
- belongsTo:Product

### Category
**Purpose**: Product categorization

**Fields**:
- `id` (bigint): Primary key
- `name` (string): Category name
- `parent_id` (bigint): Parent category for hierarchy

**Relationships**:
- hasMany:Product
- belongsTo:Category:parent_id

## API Documentation
### Endpoints
#### GET /api/v1/products
List all products with filtering

**Authentication**: Bearer token
**Rate Limit**: 100/hour

**Request**:
```json
{
  "category": "electronics",
  "low_stock": true
}
```
**Responses**:
- **200**: Successful response
  ```json
  {
    "data": [
      {
        "id": 1,
        "sku": "ELEC-001",
        "name": "Wireless Mouse",
        "current_stock": 5
      }
    ],
    "meta": {
      "total": 150,
      "page": 1
    }
  }
  ```
**Example**:
```bash
curl -X GET 'https://api.example.com/api/v1/products?low_stock=true' -H 'Authorization: Bearer {token}'
```
## Implementation Examples
### Ai Forecast Service
Service that generates AI forecasts

```php
<?php
namespace App\Services;

class AIForecastService
{
    public function generateForecast(Product $product): array
    {
        // Gather historical data
        $historicalData = $this->gatherHistoricalData($product);
        
        // Call AI service
        $prediction = $this->aiClient->predict([
            'historical_sales' => $historicalData,
            'seasonality' => $this->getSeasonalFactors(),
            'local_events' => $this->getLocalEvents()
        ]);
        
        // Store forecast
        AIForecast::create([
            'product_id' => $product->id,
            'forecast_date' => now()->addDays(7),
            'predicted_demand' => $prediction['quantity'],
            'confidence_score' => $prediction['confidence']
        ]);
        
        return $prediction;
    }
}
```
## Infrastructure
### Environments
#### Local
**Setup**: Docker Compose with all services
**Services**:
- nginx
- php-fpm
- mysql
- redis
**Configuration**: Development optimized

#### Production
**Provider**: AWS
**Services**:
- ECS
- RDS
- ElastiCache
- CloudFront
**Configuration**: Auto-scaling, multi-AZ

## Testing Strategy
### Unit Tests
**Coverage Target**: 80%
**Tools**:
- PHPUnit
- Jest
**Example**:
```
test('product needs reorder', () => { ... });
```

### Integration Tests
**Coverage Target**: 70%
**Focus Areas**:
- API endpoints
- AI service integration

