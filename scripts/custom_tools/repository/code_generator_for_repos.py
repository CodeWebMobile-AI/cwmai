"""
AI-Generated Tool: code_generator_for_repos
Description: Generate code and boilerplate for repositories
Generated: 2025-01-22T00:00:00.000000+00:00
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from scripts.ai_brain import IntelligentAIBrain

__description__ = "Generate code, boilerplate files, and features for repositories"
__parameters__ = {
    "repo_name": {
        "type": "string",
        "required": True,
        "description": "Repository name"
    },
    "generate_type": {
        "type": "string",
        "required": True,
        "enum": ["controller", "model", "view", "api", "test", "component", "service", "migration", "feature"],
        "description": "Type of code to generate"
    },
    "name": {
        "type": "string",
        "required": True,
        "description": "Name of the component/feature to generate"
    },
    "framework": {
        "type": "string",
        "required": False,
        "description": "Framework (auto-detected if not specified)"
    },
    "options": {
        "type": "object",
        "required": False,
        "description": "Additional options for generation"
    }
}
__examples__ = [
    "code_generator_for_repos(repo_name='my-app', generate_type='controller', name='UserController')",
    "code_generator_for_repos(repo_name='my-app', generate_type='api', name='products')",
    "code_generator_for_repos(repo_name='my-app', generate_type='component', name='Header')",
    "code_generator_for_repos(repo_name='my-app', generate_type='test', name='UserService')"
]

# Base path
REPOSITORIES_PATH = Path(__file__).parent.parent.parent.parent / "local_repos" / "repositories" / "CodeWebMobile-AI"

async def code_generator_for_repos(**kwargs) -> Dict[str, Any]:
    """
    Generate code and boilerplate for repositories.
    """
    repo_name = kwargs.get('repo_name')
    generate_type = kwargs.get('generate_type')
    name = kwargs.get('name')
    framework = kwargs.get('framework')
    options = kwargs.get('options', {})
    
    if not all([repo_name, generate_type, name]):
        return {
            "status": "error",
            "message": "repo_name, generate_type, and name are required"
        }
    
    # Check if repository exists
    repo_path = REPOSITORIES_PATH / repo_name
    if not repo_path.exists():
        return {
            "status": "error",
            "message": f"Repository '{repo_name}' not found"
        }
    
    # Auto-detect framework if not specified
    if not framework:
        framework = await _detect_framework(repo_path)
    
    # Generate based on type
    generators = {
        'controller': generate_controller,
        'model': generate_model,
        'view': generate_view,
        'api': generate_api_endpoint,
        'test': generate_test,
        'component': generate_component,
        'service': generate_service,
        'migration': generate_migration,
        'feature': generate_feature
    }
    
    if generate_type not in generators:
        return {
            "status": "error",
            "message": f"Unknown generate_type: {generate_type}"
        }
    
    try:
        params = {
            'repo_path': repo_path,
            'name': name,
            'framework': framework,
            'options': options
        }
        return await generators[generate_type](params)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error generating {generate_type}: {str(e)}"
        }

async def generate_controller(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a controller."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    options = params['options']
    
    # Ensure controller name format
    if not name.endswith('Controller'):
        name += 'Controller'
    
    files_created = []
    
    if framework == 'laravel':
        # Laravel controller
        controller_path = repo_path / 'app' / 'Http' / 'Controllers' / f'{name}.php'
        controller_path.parent.mkdir(parents=True, exist_ok=True)
        
        resource_methods = options.get('resource', False)
        model_name = name.replace('Controller', '')
        
        content = f"""<?php

namespace App\\Http\\Controllers;

use App\\Models\\{model_name};
use Illuminate\\Http\\Request;
use Illuminate\\Http\\JsonResponse;

class {name} extends Controller
{{
"""
        if resource_methods:
            content += f"""    /**
     * Display a listing of the resource.
     */
    public function index(): JsonResponse
    {{
        ${model_name.lower()}s = {model_name}::paginate(15);
        return response()->json(${model_name.lower()}s);
    }}

    /**
     * Store a newly created resource.
     */
    public function store(Request $request): JsonResponse
    {{
        $validated = $request->validate([
            // Add validation rules
        ]);
        
        ${model_name.lower()} = {model_name}::create($validated);
        return response()->json(${model_name.lower()}, 201);
    }}

    /**
     * Display the specified resource.
     */
    public function show({model_name} ${model_name.lower()}): JsonResponse
    {{
        return response()->json(${model_name.lower()});
    }}

    /**
     * Update the specified resource.
     */
    public function update(Request $request, {model_name} ${model_name.lower()}): JsonResponse
    {{
        $validated = $request->validate([
            // Add validation rules
        ]);
        
        ${model_name.lower()}->update($validated);
        return response()->json(${model_name.lower()});
    }}

    /**
     * Remove the specified resource.
     */
    public function destroy({model_name} ${model_name.lower()}): JsonResponse
    {{
        ${model_name.lower()}->delete();
        return response()->json(['message' => 'Deleted successfully']);
    }}
"""
        else:
            content += """    // Add your controller methods here
"""
        
        content += "}\n"
        controller_path.write_text(content)
        files_created.append(str(controller_path.relative_to(repo_path)))
        
        # Create route file hint
        route_hint = f"""// Add to routes/api.php or routes/web.php:
use App\\Http\\Controllers\\{name};

"""
        if resource_methods:
            route_hint += f"Route::resource('{model_name.lower()}s', {name}::class);\n"
        else:
            route_hint += f"Route::get('/{model_name.lower()}s', [{name}::class, 'index']);\n"
        
        files_created.append("Route hint added")
        
    elif framework == 'express':
        # Express.js controller
        controller_path = repo_path / 'controllers' / f'{name}.js'
        controller_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""const {{ validationResult }} = require('express-validator');

class {name} {{
    /**
     * Get all items
     */
    async index(req, res) {{
        try {{
            // Implement listing logic
            res.json({{ success: true, data: [] }});
        }} catch (error) {{
            res.status(500).json({{ error: error.message }});
        }}
    }}

    /**
     * Get single item
     */
    async show(req, res) {{
        try {{
            const {{ id }} = req.params;
            // Implement show logic
            res.json({{ success: true, data: {{}} }});
        }} catch (error) {{
            res.status(500).json({{ error: error.message }});
        }}
    }}

    /**
     * Create new item
     */
    async store(req, res) {{
        try {{
            const errors = validationResult(req);
            if (!errors.isEmpty()) {{
                return res.status(400).json({{ errors: errors.array() }});
            }}
            
            // Implement create logic
            res.status(201).json({{ success: true, data: {{}} }});
        }} catch (error) {{
            res.status(500).json({{ error: error.message }});
        }}
    }}

    /**
     * Update item
     */
    async update(req, res) {{
        try {{
            const {{ id }} = req.params;
            // Implement update logic
            res.json({{ success: true, data: {{}} }});
        }} catch (error) {{
            res.status(500).json({{ error: error.message }});
        }}
    }}

    /**
     * Delete item
     */
    async destroy(req, res) {{
        try {{
            const {{ id }} = req.params;
            // Implement delete logic
            res.json({{ success: true, message: 'Deleted successfully' }});
        }} catch (error) {{
            res.status(500).json({{ error: error.message }});
        }}
    }}
}}

module.exports = new {name}();
"""
        controller_path.write_text(content)
        files_created.append(str(controller_path.relative_to(repo_path)))
        
    else:
        # Generic controller
        controller_path = repo_path / 'src' / 'controllers' / f'{name}.js'
        controller_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""// {name}
// Generated: {datetime.now().isoformat()}

export class {name} {{
    constructor() {{
        // Initialize controller
    }}

    async handleRequest(request, response) {{
        // Handle incoming request
        try {{
            // Your logic here
            return response.json({{ success: true }});
        }} catch (error) {{
            return response.status(500).json({{ error: error.message }});
        }}
    }}
}}
"""
        controller_path.write_text(content)
        files_created.append(str(controller_path.relative_to(repo_path)))
    
    return {
        "status": "success",
        "message": f"Controller '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created,
        "next_steps": [
            "Add routes for the controller",
            "Implement business logic",
            "Add validation rules",
            "Create tests"
        ]
    }

async def generate_model(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a model."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    options = params['options']
    
    files_created = []
    
    if framework == 'laravel':
        # Laravel model
        model_path = repo_path / 'app' / 'Models' / f'{name}.php'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""<?php

namespace App\\Models;

use Illuminate\\Database\\Eloquent\\Factories\\HasFactory;
use Illuminate\\Database\\Eloquent\\Model;

class {name} extends Model
{{
    use HasFactory;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        // Add fillable fields
    ];

    /**
     * The attributes that should be hidden for serialization.
     *
     * @var array<int, string>
     */
    protected $hidden = [
        // Add hidden fields
    ];

    /**
     * The attributes that should be cast.
     *
     * @var array<string, string>
     */
    protected $casts = [
        // Add casts
    ];

    // Add relationships here
}}
"""
        model_path.write_text(content)
        files_created.append(str(model_path.relative_to(repo_path)))
        
    elif framework in ['express', 'nextjs']:
        # Mongoose model for Node.js
        model_path = repo_path / 'models' / f'{name}.js'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""const mongoose = require('mongoose');

const {name}Schema = new mongoose.Schema({{
    // Define your schema fields
    name: {{
        type: String,
        required: true,
        trim: true
    }},
    description: {{
        type: String
    }},
    status: {{
        type: String,
        enum: ['active', 'inactive'],
        default: 'active'
    }},
    createdBy: {{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    }}
}}, {{
    timestamps: true
}});

// Add indexes
{name}Schema.index({{ name: 1 }});

// Add methods
{name}Schema.methods.toJSON = function() {{
    const obj = this.toObject();
    delete obj.__v;
    return obj;
}};

// Add statics
{name}Schema.statics.findActive = function() {{
    return this.find({{ status: 'active' }});
}};

module.exports = mongoose.model('{name}', {name}Schema);
"""
        model_path.write_text(content)
        files_created.append(str(model_path.relative_to(repo_path)))
    
    return {
        "status": "success",
        "message": f"Model '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created,
        "next_steps": [
            "Define model attributes/fields",
            "Add relationships",
            "Create migration",
            "Add validation rules"
        ]
    }

async def generate_component(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a React/Vue component."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    options = params['options']
    
    files_created = []
    
    # Determine if React or Vue
    is_typescript = (repo_path / 'tsconfig.json').exists()
    extension = '.tsx' if is_typescript else '.jsx'
    
    # React component
    component_dir = repo_path / 'src' / 'components' / name
    component_dir.mkdir(parents=True, exist_ok=True)
    
    # Main component file
    component_path = component_dir / f'{name}{extension}'
    
    if is_typescript:
        content = f"""import React, {{ FC, useState, useEffect }} from 'react';
import styles from './{name}.module.css';

interface {name}Props {{
    // Define props here
}}

export const {name}: FC<{name}Props> = (props) => {{
    const [state, setState] = useState<any>(null);
    
    useEffect(() => {{
        // Component logic
    }}, []);
    
    return (
        <div className={{styles.container}}>
            <h2>{name}</h2>
            {{/* Component content */}}
        </div>
    );
}};

export default {name};
"""
    else:
        content = f"""import React, {{ useState, useEffect }} from 'react';
import styles from './{name}.module.css';

export const {name} = (props) => {{
    const [state, setState] = useState(null);
    
    useEffect(() => {{
        // Component logic
    }}, []);
    
    return (
        <div className={{styles.container}}>
            <h2>{name}</h2>
            {{/* Component content */}}
        </div>
    );
}};

export default {name};
"""
    
    component_path.write_text(content)
    files_created.append(str(component_path.relative_to(repo_path)))
    
    # CSS module
    css_path = component_dir / f'{name}.module.css'
    css_content = f""".container {{
    padding: 1rem;
    margin: 0 auto;
    max-width: 1200px;
}}

/* Add your styles here */
"""
    css_path.write_text(css_content)
    files_created.append(str(css_path.relative_to(repo_path)))
    
    # Index file for easier imports
    index_path = component_dir / f'index{extension}'
    index_content = f"export {{ default }} from './{name}';\n"
    index_path.write_text(index_content)
    files_created.append(str(index_path.relative_to(repo_path)))
    
    # Test file
    test_ext = '.test.tsx' if is_typescript else '.test.jsx'
    test_path = component_dir / f'{name}{test_ext}'
    test_content = f"""import React from 'react';
import {{ render, screen }} from '@testing-library/react';
import {name} from './{name}';

describe('{name}', () => {{
    it('renders without crashing', () => {{
        render(<{name} />);
        expect(screen.getByText('{name}')).toBeInTheDocument();
    }});
    
    // Add more tests
}});
"""
    test_path.write_text(test_content)
    files_created.append(str(test_path.relative_to(repo_path)))
    
    return {
        "status": "success",
        "message": f"Component '{name}' generated successfully",
        "framework": "React" + (" TypeScript" if is_typescript else ""),
        "files_created": files_created,
        "component_path": f"src/components/{name}",
        "import_statement": f"import {name} from './components/{name}';"
    }

async def generate_api_endpoint(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate API endpoint."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    options = params['options']
    
    files_created = []
    
    if framework == 'laravel':
        # Laravel API route
        route_path = repo_path / 'routes' / 'api.php'
        
        # Read existing routes
        existing_content = route_path.read_text() if route_path.exists() else "<?php\n\nuse Illuminate\\Support\\Facades\\Route;\n\n"
        
        # Add new routes
        controller_name = f"{name.title()}Controller"
        route_content = f"""
// {name} API endpoints
Route::prefix('{name}')->group(function () {{
    Route::get('/', [{controller_name}::class, 'index']);
    Route::post('/', [{controller_name}::class, 'store']);
    Route::get('/{{id}}', [{controller_name}::class, 'show']);
    Route::put('/{{id}}', [{controller_name}::class, 'update']);
    Route::delete('/{{id}}', [{controller_name}::class, 'destroy']);
}});
"""
        
        route_path.write_text(existing_content + route_content)
        files_created.append("routes/api.php (updated)")
        
    elif framework in ['express', 'nextjs']:
        # Express/Next.js API route
        if framework == 'nextjs':
            # Next.js API route
            api_dir = repo_path / 'pages' / 'api' / name
            api_dir.mkdir(parents=True, exist_ok=True)
            
            # Index route
            index_path = api_dir / 'index.js'
            content = f"""export default async function handler(req, res) {{
    const {{ method }} = req;
    
    switch (method) {{
        case 'GET':
            // Get all {name}
            res.status(200).json({{ success: true, data: [] }});
            break;
        case 'POST':
            // Create new {name}
            res.status(201).json({{ success: true, data: {{}} }});
            break;
        default:
            res.setHeader('Allow', ['GET', 'POST']);
            res.status(405).end(`Method ${{method}} Not Allowed`);
    }}
}}
"""
            index_path.write_text(content)
            files_created.append(str(index_path.relative_to(repo_path)))
            
            # Dynamic route
            dynamic_path = api_dir / '[id].js'
            dynamic_content = f"""export default async function handler(req, res) {{
    const {{ method, query: {{ id }} }} = req;
    
    switch (method) {{
        case 'GET':
            // Get single {name}
            res.status(200).json({{ success: true, data: {{ id }} }});
            break;
        case 'PUT':
            // Update {name}
            res.status(200).json({{ success: true, data: {{ id }} }});
            break;
        case 'DELETE':
            // Delete {name}
            res.status(200).json({{ success: true, message: 'Deleted' }});
            break;
        default:
            res.setHeader('Allow', ['GET', 'PUT', 'DELETE']);
            res.status(405).end(`Method ${{method}} Not Allowed`);
    }}
}}
"""
            dynamic_path.write_text(dynamic_content)
            files_created.append(str(dynamic_path.relative_to(repo_path)))
            
        else:
            # Express route
            route_path = repo_path / 'routes' / f'{name}.js'
            route_path.parent.mkdir(parents=True, exist_ok=True)
            
            content = f"""const express = require('express');
const router = express.Router();
const {name}Controller = require('../controllers/{name}Controller');

// GET all {name}
router.get('/', {name}Controller.index);

// GET single {name}
router.get('/:id', {name}Controller.show);

// POST create {name}
router.post('/', {name}Controller.store);

// PUT update {name}
router.put('/:id', {name}Controller.update);

// DELETE {name}
router.delete('/:id', {name}Controller.destroy);

module.exports = router;
"""
            route_path.write_text(content)
            files_created.append(str(route_path.relative_to(repo_path)))
    
    return {
        "status": "success",
        "message": f"API endpoint '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created,
        "endpoints": [
            f"GET /{name} - List all",
            f"GET /{name}/{{id}} - Get single",
            f"POST /{name} - Create new",
            f"PUT /{name}/{{id}} - Update",
            f"DELETE /{name}/{{id}} - Delete"
        ]
    }

async def generate_test(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate test file."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    options = params['options']
    
    files_created = []
    
    if framework == 'laravel':
        # PHPUnit test
        test_path = repo_path / 'tests' / 'Feature' / f'{name}Test.php'
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""<?php

namespace Tests\\Feature;

use Tests\\TestCase;
use Illuminate\\Foundation\\Testing\\RefreshDatabase;
use Illuminate\\Foundation\\Testing\\WithFaker;

class {name}Test extends TestCase
{{
    use RefreshDatabase, WithFaker;

    /**
     * Test index endpoint
     */
    public function test_can_list_items()
    {{
        $response = $this->getJson('/api/{name.lower()}');
        
        $response->assertStatus(200)
                 ->assertJsonStructure([
                     'data',
                     'links',
                     'meta'
                 ]);
    }}

    /**
     * Test create endpoint
     */
    public function test_can_create_item()
    {{
        $data = [
            // Add test data
        ];
        
        $response = $this->postJson('/api/{name.lower()}', $data);
        
        $response->assertStatus(201)
                 ->assertJsonFragment($data);
    }}

    /**
     * Test show endpoint
     */
    public function test_can_show_item()
    {{
        // Create test item
        $item = factory(\\App\\Models\\{name}::class)->create();
        
        $response = $this->getJson("/api/{name.lower()}/{{$item->id}}");
        
        $response->assertStatus(200)
                 ->assertJson([
                     'id' => $item->id
                 ]);
    }}

    /**
     * Test update endpoint
     */
    public function test_can_update_item()
    {{
        $item = factory(\\App\\Models\\{name}::class)->create();
        
        $updateData = [
            // Add update data
        ];
        
        $response = $this->putJson("/api/{name.lower()}/{{$item->id}}", $updateData);
        
        $response->assertStatus(200);
        $this->assertDatabaseHas('{name.lower()}s', $updateData);
    }}

    /**
     * Test delete endpoint
     */
    public function test_can_delete_item()
    {{
        $item = factory(\\App\\Models\\{name}::class)->create();
        
        $response = $this->deleteJson("/api/{name.lower()}/{{$item->id}}");
        
        $response->assertStatus(200);
        $this->assertDatabaseMissing('{name.lower()}s', ['id' => $item->id]);
    }}
}}
"""
        test_path.write_text(content)
        files_created.append(str(test_path.relative_to(repo_path)))
        
    else:
        # Jest test
        test_path = repo_path / '__tests__' / f'{name}.test.js'
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""const {{ describe, it, expect, beforeEach, afterEach }} = require('@jest/globals');

describe('{name}', () => {{
    let testData;
    
    beforeEach(() => {{
        // Setup test data
        testData = {{
            // Add test data
        }};
    }});
    
    afterEach(() => {{
        // Cleanup
    }});
    
    describe('functionality', () => {{
        it('should work correctly', () => {{
            // Test implementation
            expect(true).toBe(true);
        }});
        
        it('should handle errors', () => {{
            // Test error handling
            expect(() => {{
                // Code that should throw
            }}).toThrow();
        }});
    }});
    
    describe('edge cases', () => {{
        it('should handle null input', () => {{
            // Test null handling
        }});
        
        it('should handle empty data', () => {{
            // Test empty data
        }});
    }});
}});
"""
        test_path.write_text(content)
        files_created.append(str(test_path.relative_to(repo_path)))
    
    return {
        "status": "success",
        "message": f"Test file for '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created,
        "test_command": "npm test" if framework != 'laravel' else "php artisan test"
    }

async def generate_service(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a service class."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    
    files_created = []
    
    if not name.endswith('Service'):
        name += 'Service'
    
    if framework == 'laravel':
        service_path = repo_path / 'app' / 'Services' / f'{name}.php'
        service_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""<?php

namespace App\\Services;

use Illuminate\\Support\\Facades\\Log;

class {name}
{{
    /**
     * Create a new service instance.
     */
    public function __construct()
    {{
        //
    }}

    /**
     * Process data
     */
    public function process($data)
    {{
        try {{
            // Business logic here
            
            return [
                'success' => true,
                'data' => $data
            ];
        }} catch (\\Exception $e) {{
            Log::error('{name} error: ' . $e->getMessage());
            throw $e;
        }}
    }}

    /**
     * Validate data
     */
    protected function validate($data)
    {{
        // Validation logic
        return true;
    }}
}}
"""
    else:
        service_path = repo_path / 'services' / f'{name}.js'
        service_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""class {name} {{
    constructor() {{
        // Initialize service
    }}

    /**
     * Main service method
     */
    async process(data) {{
        try {{
            // Validate input
            this.validate(data);
            
            // Business logic here
            const result = await this.performOperation(data);
            
            return {{
                success: true,
                data: result
            }};
        }} catch (error) {{
            console.error('{name} error:', error);
            throw error;
        }}
    }}

    /**
     * Validate input data
     */
    validate(data) {{
        if (!data) {{
            throw new Error('Data is required');
        }}
        // Add more validation
    }}

    /**
     * Perform the main operation
     */
    async performOperation(data) {{
        // Implementation here
        return data;
    }}
}}

module.exports = new {name}();
"""
    
    service_path.write_text(content)
    files_created.append(str(service_path.relative_to(repo_path)))
    
    return {
        "status": "success",
        "message": f"Service '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created
    }

async def generate_migration(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate database migration."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    
    files_created = []
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    
    if framework == 'laravel':
        table_name = name.lower() + 's'  # Pluralize
        migration_name = f"{timestamp}_create_{table_name}_table"
        migration_path = repo_path / 'database' / 'migrations' / f'{migration_name}.php'
        migration_path.parent.mkdir(parents=True, exist_ok=True)
        
        content = f"""<?php

use Illuminate\\Database\\Migrations\\Migration;
use Illuminate\\Database\\Schema\\Blueprint;
use Illuminate\\Support\\Facades\\Schema;

return new class extends Migration
{{
    /**
     * Run the migrations.
     */
    public function up(): void
    {{
        Schema::create('{table_name}', function (Blueprint $table) {{
            $table->id();
            $table->string('name');
            $table->text('description')->nullable();
            $table->enum('status', ['active', 'inactive'])->default('active');
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->timestamps();
            
            // Add indexes
            $table->index('status');
            $table->index('created_at');
        }});
    }}

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {{
        Schema::dropIfExists('{table_name}');
    }}
}};
"""
        migration_path.write_text(content)
        files_created.append(str(migration_path.relative_to(repo_path)))
        
    return {
        "status": "success",
        "message": f"Migration for '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created,
        "run_command": "php artisan migrate" if framework == 'laravel' else "npm run migrate"
    }

async def generate_feature(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a complete feature with all components."""
    repo_path = params['repo_path']
    name = params['name']
    framework = params['framework']
    options = params['options']
    
    # Initialize AI brain for intelligent feature generation
    ai_brain = IntelligentAIBrain()
    
    # Analyze what components are needed
    prompt = f"""
    Generate a complete feature implementation plan for '{name}' in a {framework} application.
    
    Include:
    1. Required components (model, controller, views, etc.)
    2. Database schema
    3. API endpoints
    4. Frontend components
    5. Tests
    
    Return as JSON with component list and implementation details.
    """
    
    response = await ai_brain.execute_capability('problem_analysis', {'prompt': prompt})
    
    files_created = []
    
    # Generate all components for the feature
    # Model
    model_result = await generate_model({
        'repo_path': repo_path,
        'name': name,
        'framework': framework,
        'options': {}
    })
    files_created.extend(model_result.get('files_created', []))
    
    # Controller
    controller_result = await generate_controller({
        'repo_path': repo_path,
        'name': name,
        'framework': framework,
        'options': {'resource': True}
    })
    files_created.extend(controller_result.get('files_created', []))
    
    # API endpoints
    api_result = await generate_api_endpoint({
        'repo_path': repo_path,
        'name': name.lower(),
        'framework': framework,
        'options': {}
    })
    files_created.extend(api_result.get('files_created', []))
    
    # Migration
    migration_result = await generate_migration({
        'repo_path': repo_path,
        'name': name,
        'framework': framework,
        'options': {}
    })
    files_created.extend(migration_result.get('files_created', []))
    
    # Frontend component
    if framework != 'laravel' or (repo_path / 'resources' / 'js').exists():
        component_result = await generate_component({
            'repo_path': repo_path,
            'name': name,
            'framework': framework,
            'options': {}
        })
        files_created.extend(component_result.get('files_created', []))
    
    # Tests
    test_result = await generate_test({
        'repo_path': repo_path,
        'name': name,
        'framework': framework,
        'options': {}
    })
    files_created.extend(test_result.get('files_created', []))
    
    return {
        "status": "success",
        "message": f"Feature '{name}' generated successfully",
        "framework": framework,
        "files_created": files_created,
        "components_generated": [
            "Model",
            "Controller with CRUD methods",
            "API endpoints",
            "Database migration",
            "Frontend component",
            "Test suite"
        ],
        "next_steps": [
            "Run migrations to create database table",
            "Implement business logic in controller",
            "Add validation rules",
            "Customize frontend component",
            "Write comprehensive tests"
        ]
    }

async def _detect_framework(repo_path: Path) -> str:
    """Detect the framework used in the repository."""
    # Check for Laravel
    if (repo_path / 'artisan').exists() and (repo_path / 'composer.json').exists():
        composer_json = json.loads((repo_path / 'composer.json').read_text())
        if 'laravel/framework' in composer_json.get('require', {}):
            return 'laravel'
    
    # Check for Next.js
    if (repo_path / 'next.config.js').exists() or (repo_path / 'next.config.mjs').exists():
        return 'nextjs'
    
    # Check for Express
    if (repo_path / 'package.json').exists():
        package_json = json.loads((repo_path / 'package.json').read_text())
        dependencies = package_json.get('dependencies', {})
        if 'express' in dependencies:
            return 'express'
    
    # Check for Django
    if (repo_path / 'manage.py').exists():
        return 'django'
    
    # Default to generic
    return 'generic'

# Test function
async def main():
    # Test generating a controller
    result = await code_generator_for_repos(
        repo_name='brand-guardian-ai',
        generate_type='controller',
        name='Brand'
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())