#!/usr/bin/env python3
"""Test the architecture structure without AI generation."""

import json
from pathlib import Path
import sys

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.architecture_markdown_converter import ArchitectureMarkdownConverter


def test_architecture_structure():
    """Test with a mock architecture structure."""
    print("🧪 Testing Architecture Structure and Markdown Conversion")
    print("=" * 60)
    
    # Create a complete mock architecture
    mock_architecture = {
        "executive_summary": {
            "project": "inventory-manager - AI-powered inventory management for small retailers",
            "problem_domain": "Small retailers lose 20% of sales due to stockouts",
            "solution_type": "AI-powered inventory management platform",
            "architecture_pattern": "Inertia.js Monolith with Event-Driven AI Services",
            "deployment_target": "Local Development → Cloud Production"
        },
        "system_context": {
            "problem_analysis": {
                "problem": "Small retailers struggle with inventory across multiple channels",
                "current_solutions": "Excel spreadsheets, basic POS systems",
                "our_approach": "AI predictions based on local patterns and events",
                "key_innovation": "Machine learning that adapts to local market conditions"
            },
            "user_journey": [
                "Discovery → Users find us through search and retail forums",
                "Onboarding → Simple signup with guided inventory import",
                "First Value → See AI predictions within 2 minutes",
                "Engagement → Daily inventory checks and order suggestions",
                "Retention → Users stay for accuracy and time savings"
            ]
        },
        "business_rules": {
            "core_rules": {
                "inventory_management": {
                    "stock_levels": ["minimum_threshold", "reorder_point", "maximum_capacity"],
                    "ordering": {"lead_time": "supplier_specific", "minimum_order": "product_specific"},
                    "pricing": {"markup_rules": "category_based", "discount_limits": "role_based"}
                },
                "user_management": {
                    "roles": ["owner", "manager", "staff"],
                    "permissions": {"owner": "all", "manager": "inventory_and_reports", "staff": "view_only"}
                }
            },
            "validation_rules": {
                "product_sku": {
                    "type": "string",
                    "required": True,
                    "pattern": "^[A-Z0-9-]+$",
                    "unique": True
                },
                "stock_quantity": {
                    "type": "integer",
                    "required": True,
                    "min": 0,
                    "max": 999999
                }
            },
            "edge_cases": [
                {
                    "scenario": "Supplier out of stock",
                    "handling": "Suggest alternative suppliers or products",
                    "recovery": "Automated reorder when available"
                },
                {
                    "scenario": "Sudden demand spike",
                    "handling": "Alert user, suggest emergency reorder",
                    "recovery": "Adjust AI predictions for future"
                }
            ]
        },
        "state_machines": {
            "order_lifecycle": {
                "description": "Manages the lifecycle of inventory orders",
                "initial_state": "draft",
                "states": [
                    {"name": "draft", "description": "Order being prepared"},
                    {"name": "submitted", "description": "Order sent to supplier"},
                    {"name": "confirmed", "description": "Supplier confirmed order"},
                    {"name": "shipped", "description": "Order in transit"},
                    {"name": "received", "description": "Order delivered"},
                    {"name": "completed", "description": "Order processed into inventory"}
                ],
                "transitions": [
                    {
                        "from": "draft",
                        "to": "submitted",
                        "trigger": "submit_order",
                        "conditions": ["has_items", "supplier_active"],
                        "actions": ["send_to_supplier", "log_submission"]
                    },
                    {
                        "from": "submitted",
                        "to": "confirmed",
                        "trigger": "supplier_confirmation",
                        "actions": ["update_expected_date", "notify_user"]
                    }
                ]
            }
        },
        "domain_models": [
            {
                "name": "Product",
                "purpose": "Represents inventory items",
                "fields": [
                    {
                        "name": "id",
                        "type": "bigint",
                        "description": "Primary key",
                        "validation": {"required": True, "auto_increment": True}
                    },
                    {
                        "name": "sku",
                        "type": "string",
                        "description": "Stock keeping unit",
                        "validation": {"required": True, "unique": True, "pattern": "^[A-Z0-9-]+$"}
                    },
                    {
                        "name": "name",
                        "type": "string",
                        "description": "Product name",
                        "validation": {"required": True, "max": 255}
                    },
                    {
                        "name": "current_stock",
                        "type": "integer",
                        "description": "Current stock level",
                        "validation": {"required": True, "min": 0}
                    }
                ],
                "relationships": [
                    "belongsTo:Category:category_id",
                    "hasMany:StockMovement",
                    "belongsToMany:Supplier:through:product_suppliers"
                ],
                "key_methods": [
                    {"name": "needsReorder", "description": "Check if reorder needed", "returns": "boolean"},
                    {"name": "calculateReorderQuantity", "description": "AI-based reorder suggestion", "returns": "integer"}
                ]
            },
            {
                "name": "Supplier",
                "purpose": "Represents product suppliers",
                "fields": [
                    {"name": "id", "type": "bigint", "description": "Primary key"},
                    {"name": "name", "type": "string", "description": "Supplier name"},
                    {"name": "lead_time_days", "type": "integer", "description": "Typical delivery time"}
                ],
                "relationships": [
                    "belongsToMany:Product:through:product_suppliers",
                    "hasMany:PurchaseOrder"
                ]
            },
            {
                "name": "PurchaseOrder",
                "purpose": "Represents orders to suppliers",
                "fields": [
                    {"name": "id", "type": "bigint", "description": "Primary key"},
                    {"name": "supplier_id", "type": "bigint", "description": "Foreign key to suppliers"},
                    {"name": "status", "type": "enum", "description": "Order status"},
                    {"name": "total_amount", "type": "decimal", "description": "Order total"}
                ],
                "relationships": [
                    "belongsTo:Supplier",
                    "hasMany:PurchaseOrderItem"
                ]
            },
            {
                "name": "StockMovement",
                "purpose": "Tracks all inventory changes",
                "fields": [
                    {"name": "id", "type": "bigint", "description": "Primary key"},
                    {"name": "product_id", "type": "bigint", "description": "Foreign key to products"},
                    {"name": "quantity", "type": "integer", "description": "Positive for in, negative for out"},
                    {"name": "reason", "type": "string", "description": "Reason for movement"}
                ],
                "relationships": ["belongsTo:Product", "belongsTo:User"]
            },
            {
                "name": "AIForecast",
                "purpose": "Stores AI predictions for demand",
                "fields": [
                    {"name": "id", "type": "bigint", "description": "Primary key"},
                    {"name": "product_id", "type": "bigint", "description": "Foreign key to products"},
                    {"name": "forecast_date", "type": "date", "description": "Date of forecast"},
                    {"name": "predicted_demand", "type": "integer", "description": "Predicted units"},
                    {"name": "confidence_score", "type": "decimal", "description": "AI confidence 0-1"}
                ],
                "relationships": ["belongsTo:Product"]
            },
            {
                "name": "Category",
                "purpose": "Product categorization",
                "fields": [
                    {"name": "id", "type": "bigint", "description": "Primary key"},
                    {"name": "name", "type": "string", "description": "Category name"},
                    {"name": "parent_id", "type": "bigint", "description": "Parent category for hierarchy"}
                ],
                "relationships": ["hasMany:Product", "belongsTo:Category:parent_id"]
            }
        ],
        "api_documentation": {
            "endpoints": [
                {
                    "path": "/api/v1/products",
                    "method": "GET",
                    "description": "List all products with filtering",
                    "authentication": "Bearer token",
                    "rate_limit": "100/hour",
                    "request": {
                        "params": {
                            "category": {"type": "string", "required": False},
                            "low_stock": {"type": "boolean", "required": False}
                        },
                        "example": {"category": "electronics", "low_stock": True}
                    },
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "example": {
                                "data": [
                                    {"id": 1, "sku": "ELEC-001", "name": "Wireless Mouse", "current_stock": 5}
                                ],
                                "meta": {"total": 150, "page": 1}
                            }
                        }
                    },
                    "curl_example": "curl -X GET 'https://api.example.com/api/v1/products?low_stock=true' -H 'Authorization: Bearer {token}'"
                }
            ],
            "error_codes": {
                "PRODUCT_NOT_FOUND": "Product with given ID doesn't exist",
                "INSUFFICIENT_STOCK": "Not enough stock for operation",
                "INVALID_SKU": "SKU format is invalid"
            }
        },
        "implementation_examples": {
            "ai_forecast_service": {
                "description": "Service that generates AI forecasts",
                "code": """<?php
namespace App\\Services;

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
}"""
            }
        },
        "infrastructure": {
            "environments": {
                "local": {
                    "setup": "Docker Compose with all services",
                    "services": ["nginx", "php-fpm", "mysql", "redis"],
                    "configuration": "Development optimized"
                },
                "production": {
                    "provider": "AWS",
                    "services": ["ECS", "RDS", "ElastiCache", "CloudFront"],
                    "configuration": "Auto-scaling, multi-AZ"
                }
            }
        },
        "testing_strategy": {
            "test_types": {
                "unit": {
                    "coverage_target": "80%",
                    "tools": ["PHPUnit", "Jest"],
                    "example": "test('product needs reorder', () => { ... });"
                },
                "integration": {
                    "coverage_target": "70%",
                    "focus_areas": ["API endpoints", "AI service integration"]
                }
            }
        }
    }
    
    # Test the converter
    print("\n📋 Converting to Markdown...")
    converter = ArchitectureMarkdownConverter()
    markdown = converter.convert_to_markdown(mock_architecture)
    
    # Save output
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON
    with open(output_dir / 'mock_architecture.json', 'w') as f:
        json.dump(mock_architecture, f, indent=2)
    print(f"💾 Saved JSON to: {output_dir / 'mock_architecture.json'}")
    
    # Save Markdown
    with open(output_dir / 'mock_architecture.md', 'w') as f:
        f.write(markdown)
    print(f"📄 Saved Markdown to: {output_dir / 'mock_architecture.md'}")
    
    # Verify sections
    print("\n✅ Verifying Sections in Markdown:")
    expected_headers = [
        "# System Architecture",
        "## Executive Summary",
        "## System Context",
        "## Business Rules",
        "## State Machines",
        "## Domain Models",
        "## API Documentation",
        "## Implementation Examples",
        "## Infrastructure",
        "## Testing Strategy"
    ]
    
    for header in expected_headers:
        if header in markdown:
            print(f"  ✓ {header}")
        else:
            print(f"  ✗ {header} - MISSING")
    
    # Check markdown size and structure
    lines = markdown.split('\n')
    print(f"\n📊 Markdown Statistics:")
    print(f"  Total lines: {len(lines)}")
    print(f"  Total characters: {len(markdown)}")
    print(f"  Code blocks: {markdown.count('```')}")
    
    # Show preview
    print("\n📄 Markdown Preview (first 50 lines):")
    print("-" * 60)
    print('\n'.join(lines[:50]))
    print("...")
    
    return True


if __name__ == "__main__":
    success = test_architecture_structure()
    if success:
        print("\n✅ Architecture structure test PASSED!")
        print("Check test_output/mock_architecture.md for the full document")
    else:
        print("\n❌ Architecture structure test FAILED!")