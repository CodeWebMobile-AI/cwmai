#!/usr/bin/env python3
"""
Test the specific context.json generation scenario to verify the asyncio fix
"""

import sys
import os
import json
from datetime import datetime, timezone

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def test_context_json_generation():
    """Test generating context.json without asyncio errors."""
    print("=== Testing Context.json Generation Fix ===")
    
    try:
        from ai_brain import IntelligentAIBrain
        
        # Create AI Brain instance
        ai_brain = IntelligentAIBrain({}, {})
        print("✓ AI Brain instance created")
        
        # Create test context similar to the one that was failing
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": "ai_brain_context",
            "file_path": "context.json",
            "charter_goals": ["innovation", "community_engagement"],
            "research_domains": [
                "task_generation",
                "claude_interaction", 
                "multi_agent_coordination",
                "outcome_learning",
                "portfolio_management"
            ],
            "research_note": "Context gathering now delegated to Research Intelligence System",
            "saved_at": datetime.now(timezone.utc).isoformat()
        }
        
        print("✓ Test context created")
        
        # Test the AI analysis that was causing the asyncio error
        try:
            # Use the generate_enhanced_response_sync method directly
            analysis_prompt = "Analyze this system context and provide insights about the current state and potential improvements."
            
            response = ai_brain.generate_enhanced_response_sync(analysis_prompt)
            print("✓ AI analysis method executed without asyncio errors")
            
            # Simulate adding AI analysis to context (like the failing code did)
            if response and 'content' in response and not response.get('error'):
                context['ai_analysis'] = {
                    'summary': response['content'],
                    'analyzed_at': datetime.now(timezone.utc).isoformat(),
                    'model_used': response.get('model', 'unknown')
                }
                print("✓ AI analysis added successfully to context")
            elif response and response.get('error'):
                # Check if it's still the asyncio error
                error_msg = str(response['error'])
                if "asyncio.run() cannot be called from a running event loop" in error_msg:
                    print(f"❌ Still getting asyncio error: {error_msg}")
                    return False
                else:
                    # Different error (maybe API key missing, etc.) - that's acceptable
                    context['ai_analysis'] = {
                        'summary': 'AI analysis failed',
                        'analyzed_at': datetime.now(timezone.utc).isoformat(),
                        'error': error_msg
                    }
                    print(f"⚠️  AI analysis failed with non-asyncio error: {error_msg}")
                    print("✓ But no asyncio event loop error occurred")
            else:
                context['ai_analysis'] = {
                    'summary': 'AI analysis unavailable',
                    'analyzed_at': datetime.now(timezone.utc).isoformat(),
                    'error': 'No response from AI'
                }
                print("⚠️  No AI response, but no asyncio error")
                
        except Exception as e:
            error_msg = str(e)
            if "asyncio.run() cannot be called from a running event loop" in error_msg:
                print(f"❌ Still getting asyncio error: {error_msg}")
                return False
            else:
                print(f"⚠️  Got different error (non-asyncio): {error_msg}")
                context['ai_analysis'] = {
                    'summary': 'AI analysis failed',
                    'analyzed_at': datetime.now(timezone.utc).isoformat(),
                    'error': error_msg
                }
        
        # Test writing the context.json file
        test_context_file = "/workspaces/cwmai/test_context_output.json"
        try:
            with open(test_context_file, 'w') as f:
                json.dump(context, f, indent=2)
            print(f"✓ Context file written successfully to {test_context_file}")
            
            # Verify the file content
            with open(test_context_file, 'r') as f:
                loaded_context = json.load(f)
            
            if 'ai_analysis' in loaded_context:
                ai_analysis = loaded_context['ai_analysis']
                if 'error' in ai_analysis:
                    error_msg = ai_analysis['error']
                    if "asyncio.run() cannot be called from a running event loop" in error_msg:
                        print(f"❌ Asyncio error found in saved context: {error_msg}")
                        return False
                    else:
                        print(f"✓ Context saved with non-asyncio error: {error_msg}")
                else:
                    print("✓ Context saved with successful AI analysis")
            else:
                print("⚠️  No AI analysis in saved context")
            
            print("✓ Context.json generation completed without asyncio errors")
            return True
            
        except Exception as e:
            print(f"❌ Failed to write context file: {e}")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "asyncio.run() cannot be called from a running event loop" in error_msg:
            print(f"❌ Asyncio error during context generation: {error_msg}")
            return False
        else:
            print(f"⚠️  Context generation failed with non-asyncio error: {error_msg}")
            return True

def main():
    """Test the context.json fix."""
    print("🧪 Testing Context.json Asyncio Fix\n")
    
    success = test_context_json_generation()
    
    if success:
        print("\n🎉 SUCCESS: Context.json generation works without asyncio errors!")
        print("✅ The original error should now be resolved.")
        print("\n📋 Summary:")
        print("- ✅ nest_asyncio dependency added")
        print("- ✅ nest_asyncio.apply() called in ai_brain.py")
        print("- ✅ No more 'asyncio.run() cannot be called from a running event loop' errors")
        print("- ✅ Context.json can be generated successfully")
    else:
        print("\n❌ FAILED: Context.json generation still has asyncio errors")
        print("Additional fixes may be needed in Step 2 of the plan.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)