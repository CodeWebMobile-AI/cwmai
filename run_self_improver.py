#!/usr/bin/env python3
"""
Run the Safe Self-Improver on the codebase

This script analyzes the codebase for improvement opportunities and applies
safe, validated improvements with full logging and rollback capabilities.
"""

import sys
import os
import logging
from datetime import datetime
import json

# Add scripts directory to path
sys.path.insert(0, '/workspaces/cwmai/scripts')

from safe_self_improver import SafeSelfImprover, ModificationType

# Configure logging
log_filename = f'self_improvement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run self-improvement analysis and application."""
    logger.info("="*60)
    logger.info("STARTING SAFE SELF-IMPROVER")
    logger.info("="*60)
    
    # Initialize the self-improver
    repo_path = "/workspaces/cwmai"
    max_daily_changes = 3  # Conservative limit for safety
    
    logger.info(f"Repository: {repo_path}")
    logger.info(f"Max daily changes: {max_daily_changes}")
    
    try:
        improver = SafeSelfImprover(repo_path=repo_path, max_changes_per_day=max_daily_changes)
        logger.info("✅ Self-improver initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize self-improver: {e}")
        return 1
    
    # Analyze the codebase for improvements
    logger.info("\n🔍 Analyzing codebase for improvement opportunities...")
    opportunities = improver.analyze_improvement_opportunities()
    
    logger.info(f"\n📊 Found {len(opportunities)} improvement opportunities:")
    for i, opp in enumerate(opportunities, 1):
        logger.info(f"  {i}. [{opp['type'].value}] {opp['file']}: {opp['description']} (priority: {opp['priority']:.2f})")
    
    if not opportunities:
        logger.info("No improvement opportunities found at this time.")
        return 0
    
    # Sort by priority
    opportunities.sort(key=lambda x: x['priority'], reverse=True)
    
    # Prioritize our test files for demonstration
    test_file_opportunities = [opp for opp in opportunities if 'simple_optimization_target.py' in opp['file']]
    other_opportunities = [opp for opp in opportunities if 'simple_optimization_target.py' not in opp['file']]
    opportunities = test_file_opportunities + other_opportunities
    
    # Process top opportunities (limited for safety)
    improvements_to_try = min(3, len(opportunities))
    logger.info(f"\n🚀 Attempting to apply top {improvements_to_try} improvements...")
    
    successful_improvements = 0
    failed_improvements = 0
    
    for i, opp in enumerate(opportunities[:improvements_to_try], 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing improvement {i}/{improvements_to_try}")
        logger.info(f"{'='*50}")
        logger.info(f"File: {opp['file']}")
        logger.info(f"Type: {opp['type'].value}")
        logger.info(f"Description: {opp['description']}")
        
        # Propose the improvement
        logger.info("\n📝 Proposing improvement...")
        modification = improver.propose_improvement(
            target_file=opp['file'],
            improvement_type=opp['type'],
            description=opp['description']
        )
        
        if not modification:
            logger.warning("❌ No modification generated (possibly unsafe or limit reached)")
            failed_improvements += 1
            continue
        
        logger.info(f"✅ Modification proposed: {modification.id}")
        logger.info(f"   Safety score: {modification.safety_score:.2f}")
        logger.info(f"   Changes: {len(modification.changes)} modifications")
        
        # Show the changes
        for j, (old, new) in enumerate(modification.changes[:2], 1):  # Show first 2 changes
            logger.info(f"\n   Change {j}:")
            logger.info(f"   OLD: {old[:100]}..." if len(old) > 100 else f"   OLD: {old}")
            logger.info(f"   NEW: {new[:100]}..." if len(new) > 100 else f"   NEW: {new}")
        
        if modification.safety_score < 0.8:
            logger.warning("❌ Safety score too low, skipping application")
            failed_improvements += 1
            continue
        
        # Apply the improvement
        logger.info("\n🔧 Applying improvement...")
        success = improver.apply_improvement(modification)
        
        if success:
            logger.info(f"✅ Successfully applied improvement: {modification.id}")
            successful_improvements += 1
            
            # Log performance impact
            if modification.performance_impact:
                logger.info("📈 Performance impact:")
                for metric, value in modification.performance_impact.items():
                    logger.info(f"   - {metric}: {value}")
            
            # Log test results
            if modification.test_results:
                logger.info(f"🧪 Test results: {modification.test_results.get('pass_rate', 0)*100:.1f}% pass rate")
        else:
            logger.error(f"❌ Failed to apply improvement: {modification.id}")
            failed_improvements += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SELF-IMPROVEMENT SESSION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful improvements: {successful_improvements}")
    logger.info(f"❌ Failed improvements: {failed_improvements}")
    logger.info(f"📁 Log file: {log_filename}")
    
    # Check if modifications directory exists
    modifications_dir = os.path.join(repo_path, '.self_improver')
    if os.path.exists(modifications_dir):
        logger.info(f"📂 Modifications history: {modifications_dir}")
        
        # List modification files
        mod_files = [f for f in os.listdir(modifications_dir) if f.startswith('modifications_')]
        if mod_files:
            logger.info("   Available history files:")
            for f in sorted(mod_files):
                logger.info(f"   - {f}")
    
    logger.info(f"\n✨ Self-improvement session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())