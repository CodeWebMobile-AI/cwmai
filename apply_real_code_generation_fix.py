#!/usr/bin/env python3
"""
Apply the real code generation fix to the evolutionary system
"""

import sys
import os

def apply_fix():
    """Apply the fix to make evolution generate real code."""
    
    print("🔧 Applying real code generation fix to evolutionary engine...")
    
    # Read the knowledge-driven evolutionary engine
    engine_file = 'scripts/knowledge_driven_evolutionary_engine.py'
    with open(engine_file, 'r') as f:
        content = f.read()
    
    # Check if fix already applied
    if 'evolutionary_code_generator_fix' in content:
        print("✅ Fix already applied")
        return
    
    # Add import for the fix
    import_line = "from scripts.evolutionary_code_generator_fix import RealCodeEvolutionFix, apply_real_code_generation_fixes"
    
    # Find where to add import (after other imports)
    lines = content.split('\n')
    import_index = 0
    for i, line in enumerate(lines):
        if line.startswith('from scripts.') or line.startswith('import '):
            import_index = i
    
    lines.insert(import_index + 1, import_line)
    
    # Add fix initialization in __init__ method
    init_fix = """
        # Initialize real code generation validator
        self.code_validator = apply_real_code_generation_fixes()
        self.logger.info("Real code generation validator initialized")"""
    
    # Find __init__ method
    for i, line in enumerate(lines):
        if 'def __init__(self' in line and 'KnowledgeDrivenEvolutionaryEngine' in lines[i-5:i+1]:
            # Find end of __init__
            indent_count = len(line) - len(line.lstrip())
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith(' ' * (indent_count + 4)):
                    lines.insert(j-1, init_fix)
                    break
            break
    
    # Modify the generate_improvement method to validate code
    validation_code = """
        # Validate generated code is real implementation
        if self.code_validator and 'code' in improvement:
            is_valid, error_msg, test_results = self.code_validator.validate_generated_code(
                improvement['code'], improvement
            )
            
            if not is_valid:
                self.logger.warning(f"Generated placeholder code: {error_msg}")
                # Regenerate with stricter requirements
                prompt = self.code_validator.create_implementation_prompt(improvement)
                # Re-attempt generation with explicit prompt
                improvement['code'] = await self._generate_real_code(prompt, improvement)
                
                # Validate again
                is_valid, error_msg, test_results = self.code_validator.validate_generated_code(
                    improvement['code'], improvement
                )
                
                if is_valid:
                    improvement['test_results'] = test_results
                    improvement['validated'] = True
                else:
                    improvement['fitness'] = 0.1  # Penalize placeholder code
                    improvement['validation_error'] = error_msg
            else:
                improvement['test_results'] = test_results
                improvement['validated'] = True
                # Boost fitness for real implementations
                improvement['fitness'] = improvement.get('fitness', 0.5) * 1.2"""
    
    # Find generate_improvement method
    for i, line in enumerate(lines):
        if 'async def generate_improvement' in line:
            # Find where to insert validation (before return)
            for j in range(i, len(lines)):
                if 'return improvement' in lines[j] and lines[j].strip().startswith('return'):
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    # Insert validation code before return
                    validation_lines = validation_code.split('\n')
                    for k, val_line in enumerate(validation_lines):
                        if val_line.strip():
                            lines.insert(j + k, ' ' * indent + val_line.strip())
                    break
            break
    
    # Add helper method for real code generation
    helper_method = '''
    async def _generate_real_code(self, prompt: str, improvement: Dict[str, Any]) -> str:
        """Generate real implementation code using explicit prompt."""
        try:
            response = await self.ai_brain.generate_improvement(
                prompt=prompt,
                context={
                    'type': improvement.get('type'),
                    'requirements': 'Generate complete, working code with real implementation',
                    'no_placeholders': True
                }
            )
            
            if response and 'code' in response:
                return response['code']
            else:
                # Fallback to more explicit generation
                code_prompt = f"""
Generate a Python class that implements: {improvement.get('name')}

The class must:
1. Have a working apply() method with REAL logic
2. Process input data and return meaningful results
3. Include error handling
4. NOT be a placeholder

Type: {improvement.get('type')}
Description: {improvement.get('description')}

Generate the complete implementation now:
"""
                code_response = await self.ai_brain.generate_code(code_prompt)
                return code_response.get('code', '')
                
        except Exception as e:
            self.logger.error(f"Error generating real code: {e}")
            return ""'''
    
    # Add helper method after generate_improvement
    for i, line in enumerate(lines):
        if 'async def generate_improvement' in line:
            # Find end of method
            indent = len(line) - len(line.lstrip())
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith(' ' * indent) and not lines[j].strip().startswith('@'):
                    # Insert helper method here
                    lines.insert(j, helper_method)
                    break
            break
    
    # Write the updated file
    with open(engine_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Real code generation fix applied to evolutionary engine")
    
    # Also update the fitness evaluation to check for real code
    print("\n🔧 Updating fitness evaluation...")
    
    # Read the fitness calculation section
    for i, line in enumerate(lines):
        if 'def calculate_fitness' in line or 'def _calculate_fitness' in line:
            # Add validation check to fitness
            fitness_check = """
        # Penalize placeholder code
        if hasattr(self, 'code_validator') and 'code' in improvement:
            if not improvement.get('validated', False):
                is_valid, _, _ = self.code_validator.validate_generated_code(
                    improvement['code'], improvement
                )
                if not is_valid:
                    fitness *= 0.1  # Severe penalty for placeholder code"""
            
            # Find where to insert (after initial fitness calculation)
            for j in range(i+1, min(i+50, len(lines))):
                if 'fitness' in lines[j] and '=' in lines[j]:
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    lines.insert(j+1, fitness_check)
                    break
            break
    
    # Write final version
    with open(engine_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print("✅ Fitness evaluation updated to penalize placeholder code")
    print("\n✨ Real code generation fix fully applied!")
    print("\nThe evolutionary system will now:")
    print("  • Validate that generated code is real implementation")
    print("  • Run tests on generated code")
    print("  • Penalize placeholder code with low fitness scores")
    print("  • Regenerate code that fails validation")
    print("  • Only mark improvements as successful if they pass tests")

if __name__ == "__main__":
    apply_fix()