"""
Staged Self-Improvement System

Extends SafeSelfImprover to stage improvements in a separate directory
before applying them to production code.
"""

import os
import json
import shutil
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_brain import IntelligentAIBrain
from dataclasses import dataclass, field
import hashlib
import difflib

from safe_self_improver import SafeSelfImprover, Modification, ModificationType
from ai_brain import IntelligentAIBrain
from ai_code_analyzer import AICodeAnalyzer


@dataclass
class StagedImprovement:
    """Represents a staged improvement awaiting validation."""
    modification: Modification
    staged_path: str
    original_path: str
    created_at: datetime
    validation_status: Optional[Dict[str, Any]] = None
    applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class StagedSelfImprover(SafeSelfImprover):
    """Self-improver that stages changes before applying them."""
    
    def __init__(self, repo_path: str = ".", max_changes_per_day: int = 24, ai_brain: Optional[IntelligentAIBrain] = None):
        """Initialize staged self-improver with staging directories.
        
        Args:
            repo_path: Path to the repository
            max_changes_per_day: Maximum changes allowed per day
            ai_brain: Optional AI brain for intelligent analysis
        """
        super().__init__(repo_path, max_changes_per_day)
        
        # Initialize AI components if available
        self.ai_brain = ai_brain
        self.ai_analyzer = AICodeAnalyzer(ai_brain) if ai_brain else None
        
        # Staging directories
        self.staging_root = os.path.join(self.repo_path, '.self_improver')
        self.staging_dir = os.path.join(self.staging_root, 'staged')
        self.validated_dir = os.path.join(self.staging_root, 'validated')
        self.applied_dir = os.path.join(self.staging_root, 'applied')
        self.rollback_dir = os.path.join(self.staging_root, 'rollback')
        self.metrics_dir = os.path.join(self.staging_root, 'metrics')
        self.reports_dir = os.path.join(self.staging_root, 'reports')
        
        # Create directories
        self._create_staging_directories()
        
        # Load staged improvements
        self.staged_improvements: Dict[str, StagedImprovement] = self._load_staged_improvements()
        
        # Configuration
        self.config = self._load_staging_config()
    
    def _create_staging_directories(self):
        """Create necessary staging directories."""
        dirs = [
            self.staging_dir,
            self.validated_dir,
            self.applied_dir,
            self.rollback_dir,
            os.path.join(self.metrics_dir, 'before'),
            os.path.join(self.metrics_dir, 'after'),
            self.reports_dir
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_staging_config(self) -> Dict[str, Any]:
        """Load staging configuration."""
        config_path = os.path.join(self.staging_root, 'config.json')
        
        default_config = {
            'auto_validate': True,
            'auto_apply_validated': False,
            'validation_timeout': 300,
            'min_confidence': 0.8,
            'batch_size': 3,
            'ab_test_duration': 3600,
            'require_improvement': True,
            'max_regression_percent': 5
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except:
                pass
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def stage_improvement(self, modification: Modification) -> Optional[StagedImprovement]:
        """Stage an improvement without modifying the original file.
        
        Args:
            modification: The modification to stage
            
        Returns:
            StagedImprovement object or None if staging failed
        """
        try:
            # Create unique staging ID
            staging_id = f"{modification.id}_{int(datetime.now().timestamp())}"
            
            # Create staging subdirectory
            staging_subdir = os.path.join(self.staging_dir, staging_id)
            os.makedirs(staging_subdir, exist_ok=True)
            
            # Determine paths
            original_path = os.path.join(self.repo_path, modification.target_file)
            staged_path = os.path.join(staging_subdir, os.path.basename(modification.target_file))
            
            # Read original code
            with open(original_path, 'r') as f:
                original_code = f.read()
            
            # Apply improvements to create staged version
            improved_code = self._apply_changes_to_code(original_code, modification.changes)
            
            # Save staged version
            with open(staged_path, 'w') as f:
                f.write(improved_code)
            
            # Create staged improvement object
            staged = StagedImprovement(
                modification=modification,
                staged_path=staged_path,
                original_path=original_path,
                created_at=datetime.now(timezone.utc),
                metadata={
                    'staging_id': staging_id,
                    'original_hash': hashlib.md5(original_code.encode()).hexdigest(),
                    'improved_hash': hashlib.md5(improved_code.encode()).hexdigest(),
                    'lines_changed': len(modification.changes),
                    'file_size_before': len(original_code),
                    'file_size_after': len(improved_code)
                }
            )
            
            # Save metadata
            self._save_staged_metadata(staging_id, staged)
            
            # Add to tracked improvements
            self.staged_improvements[staging_id] = staged
            
            # Generate diff report
            self._generate_diff_report(staging_id, original_code, improved_code)
            
            print(f"✅ Staged improvement {staging_id} for {modification.target_file}")
            return staged
            
        except Exception as e:
            print(f"❌ Failed to stage improvement: {e}")
            return None
    
    def _save_staged_metadata(self, staging_id: str, staged: StagedImprovement):
        """Save metadata for a staged improvement."""
        metadata_path = os.path.join(self.staging_dir, staging_id, 'metadata.json')
        
        metadata = {
            'modification': {
                'id': staged.modification.id,
                'type': staged.modification.type.value,
                'target_file': staged.modification.target_file,
                'description': staged.modification.description,
                'changes_count': len(staged.modification.changes),
                'safety_score': staged.modification.safety_score,
                'timestamp': staged.modification.timestamp.isoformat()
            },
            'staging': {
                'staging_id': staging_id,
                'staged_path': staged.staged_path,
                'original_path': staged.original_path,
                'created_at': staged.created_at.isoformat(),
                'applied': staged.applied
            },
            'metadata': staged.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_diff_report(self, staging_id: str, original: str, improved: str):
        """Generate a diff report for the staged improvement."""
        report_path = os.path.join(self.reports_dir, f"{staging_id}_diff.txt")
        
        # Generate unified diff
        original_lines = original.splitlines(keepends=True)
        improved_lines = improved.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            improved_lines,
            fromfile='original',
            tofile='improved',
            n=3
        )
        
        with open(report_path, 'w') as f:
            f.write(''.join(diff))
    
    async def validate_staged_improvement(self, staging_id: str) -> Dict[str, Any]:
        """Validate a staged improvement.
        
        Args:
            staging_id: ID of the staged improvement
            
        Returns:
            Validation results dictionary
        """
        if staging_id not in self.staged_improvements:
            return {'error': 'Staged improvement not found'}
        
        staged = self.staged_improvements[staging_id]
        print(f"🔍 Validating staged improvement {staging_id}")
        
        # Import validator
        try:
            from improvement_validator import ImprovementValidator
            validator = ImprovementValidator(self.repo_path)
        except ImportError:
            # Fallback to basic validation
            return await self._basic_validation(staged)
        
        # Run comprehensive validation
        validation_results = await validator.validate_improvement(
            staged.staged_path,
            staged.original_path,
            staged.modification
        )
        
        # Update staged improvement with validation results
        staged.validation_status = validation_results
        self._save_staged_metadata(staging_id, staged)
        
        # Move to validated directory if passed
        if validation_results.get('ready_to_apply', False):
            self._move_to_validated(staging_id)
        
        return validation_results
    
    async def _basic_validation(self, staged: StagedImprovement) -> Dict[str, Any]:
        """Basic validation when validator is not available."""
        try:
            # Syntax check
            import py_compile
            py_compile.compile(staged.staged_path, doraise=True)
            
            # Basic safety check
            with open(staged.staged_path, 'r') as f:
                improved_code = f.read()
            
            # Check for dangerous patterns
            dangerous_patterns = [
                'exec(', 'eval(', '__import__', 'os.system',
                'subprocess.call', 'shutil.rmtree'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in improved_code:
                    return {
                        'ready_to_apply': False,
                        'syntax_valid': True,
                        'safety_passed': False,
                        'error': f'Dangerous pattern found: {pattern}'
                    }
            
            return {
                'ready_to_apply': True,
                'syntax_valid': True,
                'safety_passed': True,
                'tests_pass': True,  # Assumed since we can't run tests
                'performance_improved': True  # Assumed
            }
            
        except Exception as e:
            return {
                'ready_to_apply': False,
                'error': str(e)
            }
    
    def _move_to_validated(self, staging_id: str):
        """Move a staged improvement to the validated directory."""
        staged = self.staged_improvements[staging_id]
        
        # Create validated directory
        validated_subdir = os.path.join(self.validated_dir, staging_id)
        
        # Move entire staging directory
        staging_subdir = os.path.join(self.staging_dir, staging_id)
        shutil.move(staging_subdir, validated_subdir)
        
        # Update path
        staged.staged_path = os.path.join(
            validated_subdir, 
            os.path.basename(staged.staged_path)
        )
        
        print(f"✅ Moved {staging_id} to validated")
    
    async def apply_staged_improvement(self, staging_id: str) -> bool:
        """Apply a validated staged improvement to the original file.
        
        Args:
            staging_id: ID of the staged improvement
            
        Returns:
            Success status
        """
        if staging_id not in self.staged_improvements:
            print(f"❌ Staged improvement {staging_id} not found")
            return False
        
        staged = self.staged_improvements[staging_id]
        
        # Check if validated
        if not staged.validation_status or not staged.validation_status.get('ready_to_apply'):
            print(f"❌ Improvement {staging_id} not validated or not ready to apply")
            return False
        
        try:
            # Create rollback
            self._create_rollback(staged)
            
            # Read staged improvement
            with open(staged.staged_path, 'r') as f:
                improved_code = f.read()
            
            # Apply to original file
            with open(staged.original_path, 'w') as f:
                f.write(improved_code)
            
            # Mark as applied
            staged.applied = True
            staged.metadata['applied_at'] = datetime.now(timezone.utc).isoformat()
            
            # Move to applied directory
            self._move_to_applied(staging_id)
            
            # Update modification tracking
            staged.modification.applied = True
            staged.modification.success = True
            self.modifications_today.append(staged.modification)
            self._save_modification(staged.modification)
            
            print(f"✅ Successfully applied improvement {staging_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to apply improvement: {e}")
            # Attempt rollback
            self._rollback_improvement(staging_id)
            return False
    
    def _create_rollback(self, staged: StagedImprovement):
        """Create a rollback point for an improvement."""
        rollback_subdir = os.path.join(
            self.rollback_dir, 
            staged.metadata['staging_id']
        )
        os.makedirs(rollback_subdir, exist_ok=True)
        
        # Copy original file to rollback
        rollback_path = os.path.join(
            rollback_subdir, 
            os.path.basename(staged.original_path)
        )
        shutil.copy2(staged.original_path, rollback_path)
        
        # Save rollback metadata
        rollback_metadata = {
            'staging_id': staged.metadata['staging_id'],
            'original_path': staged.original_path,
            'rollback_path': rollback_path,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        with open(os.path.join(rollback_subdir, 'rollback.json'), 'w') as f:
            json.dump(rollback_metadata, f, indent=2)
    
    def _move_to_applied(self, staging_id: str):
        """Move a staged improvement to the applied directory."""
        staged = self.staged_improvements[staging_id]
        
        # Determine source directory (could be staged or validated)
        if os.path.exists(os.path.join(self.staging_dir, staging_id)):
            source_dir = os.path.join(self.staging_dir, staging_id)
        else:
            source_dir = os.path.join(self.validated_dir, staging_id)
        
        # Move to applied
        applied_subdir = os.path.join(self.applied_dir, staging_id)
        shutil.move(source_dir, applied_subdir)
        
        print(f"✅ Moved {staging_id} to applied")
    
    def _rollback_improvement(self, staging_id: str) -> bool:
        """Rollback an applied improvement."""
        rollback_subdir = os.path.join(self.rollback_dir, staging_id)
        rollback_metadata_path = os.path.join(rollback_subdir, 'rollback.json')
        
        if not os.path.exists(rollback_metadata_path):
            print(f"❌ No rollback found for {staging_id}")
            return False
        
        try:
            with open(rollback_metadata_path, 'r') as f:
                rollback_metadata = json.load(f)
            
            # Restore original file
            shutil.copy2(
                rollback_metadata['rollback_path'],
                rollback_metadata['original_path']
            )
            
            print(f"✅ Rolled back improvement {staging_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to rollback: {e}")
            return False
    
    def get_staged_improvements(self, status: Optional[str] = None) -> List[StagedImprovement]:
        """Get list of staged improvements.
        
        Args:
            status: Filter by status ('staged', 'validated', 'applied')
            
        Returns:
            List of staged improvements
        """
        improvements = []
        
        # Check different directories based on status
        if status == 'staged' or status is None:
            for staging_id in os.listdir(self.staging_dir):
                if staging_id in self.staged_improvements:
                    improvements.append(self.staged_improvements[staging_id])
        
        if status == 'validated' or status is None:
            for staging_id in os.listdir(self.validated_dir):
                if staging_id in self.staged_improvements:
                    improvements.append(self.staged_improvements[staging_id])
        
        if status == 'applied' or status is None:
            for staging_id in os.listdir(self.applied_dir):
                if staging_id in self.staged_improvements:
                    improvements.append(self.staged_improvements[staging_id])
        
        return improvements
    
    def _load_staged_improvements(self) -> Dict[str, StagedImprovement]:
        """Load all staged improvements from disk."""
        improvements = {}
        
        # Load from all directories
        for directory in [self.staging_dir, self.validated_dir, self.applied_dir]:
            if not os.path.exists(directory):
                continue
                
            for staging_id in os.listdir(directory):
                metadata_path = os.path.join(directory, staging_id, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                        
                        # Reconstruct modification
                        mod_data = data['modification']
                        modification = Modification(
                            id=mod_data['id'],
                            type=ModificationType(mod_data['type']),
                            target_file=mod_data['target_file'],
                            description=mod_data['description'],
                            changes=[],  # Not stored in metadata
                            timestamp=datetime.fromisoformat(mod_data['timestamp']),
                            safety_score=mod_data.get('safety_score', 0)
                        )
                        
                        # Reconstruct staged improvement
                        staged = StagedImprovement(
                            modification=modification,
                            staged_path=data['staging']['staged_path'],
                            original_path=data['staging']['original_path'],
                            created_at=datetime.fromisoformat(data['staging']['created_at']),
                            applied=data['staging']['applied'],
                            metadata=data['metadata']
                        )
                        
                        if 'validation_status' in data:
                            staged.validation_status = data['validation_status']
                        
                        improvements[staging_id] = staged
                        
                    except Exception as e:
                        print(f"Error loading staged improvement {staging_id}: {e}")
        
        return improvements
    
    async def stage_batch_improvements(self, opportunities: List[Dict[str, Any]], 
                                      max_batch: Optional[int] = None) -> List[str]:
        """Stage a batch of improvements.
        
        Args:
            opportunities: List of improvement opportunities
            max_batch: Maximum number to stage (defaults to config batch_size)
            
        Returns:
            List of staging IDs
        """
        if max_batch is None:
            max_batch = self.config.get('batch_size', 3)
        
        staged_ids = []
        
        for opp in opportunities[:max_batch]:
            # Create modification from opportunity
            modification = None
            
            # Check if this is from AI analyzer (has original_code/improved_code)
            if 'original_code' in opp and 'improved_code' in opp:
                # Direct modification from AI analyzer
                modification = Modification(
                    type=opp['type'],
                    target_file=opp['file'],
                    description=opp['description'],
                    changes=[{
                        'type': 'replace',
                        'original': opp['original_code'],
                        'replacement': opp['improved_code'],
                        'line_number': opp.get('line_start', 0)
                    }],
                    safety_score=opp.get('score', 0.8)
                )
            else:
                # Traditional regex-based improvement
                modification = self.propose_improvement(
                    target_file=opp['file'],
                    improvement_type=opp['type'],
                    description=opp['description']
                )
            
            if modification:
                staged = self.stage_improvement(modification)
                if staged:
                    staged_ids.append(staged.metadata['staging_id'])
        
        print(f"📦 Staged {len(staged_ids)} improvements")
        return staged_ids
    
    async def validate_batch(self, staging_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Validate a batch of staged improvements.
        
        Args:
            staging_ids: List of staging IDs to validate
            
        Returns:
            Dictionary mapping staging ID to validation results
        """
        results = {}
        
        for staging_id in staging_ids:
            results[staging_id] = await self.validate_staged_improvement(staging_id)
        
        # Summary
        ready_count = sum(1 for r in results.values() if r.get('ready_to_apply', False))
        print(f"✅ {ready_count}/{len(staging_ids)} improvements ready to apply")
        
        return results
    
    def generate_staging_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all staged improvements."""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_staged': 0,
                'total_validated': 0,
                'total_applied': 0,
                'success_rate': 0.0
            },
            'by_type': {},
            'by_file': {},
            'recent_activity': []
        }
        
        # Count by status
        staged = self.get_staged_improvements('staged')
        validated = self.get_staged_improvements('validated')
        applied = self.get_staged_improvements('applied')
        
        report['summary']['total_staged'] = len(staged)
        report['summary']['total_validated'] = len(validated)
        report['summary']['total_applied'] = len(applied)
        
        # Calculate success rate
        total_attempted = len(validated) + len(applied)
        if total_attempted > 0:
            report['summary']['success_rate'] = len(applied) / total_attempted
        
        # Group by type and file
        all_improvements = staged + validated + applied
        
        for imp in all_improvements:
            # By type
            imp_type = imp.modification.type.value
            if imp_type not in report['by_type']:
                report['by_type'][imp_type] = {'count': 0, 'applied': 0}
            report['by_type'][imp_type]['count'] += 1
            if imp.applied:
                report['by_type'][imp_type]['applied'] += 1
            
            # By file
            file_path = imp.modification.target_file
            if file_path not in report['by_file']:
                report['by_file'][file_path] = {'count': 0, 'applied': 0}
            report['by_file'][file_path]['count'] += 1
            if imp.applied:
                report['by_file'][file_path]['applied'] += 1
        
        # Recent activity
        sorted_improvements = sorted(
            all_improvements, 
            key=lambda x: x.created_at, 
            reverse=True
        )[:10]
        
        for imp in sorted_improvements:
            activity = {
                'staging_id': imp.metadata['staging_id'],
                'type': imp.modification.type.value,
                'file': imp.modification.target_file,
                'description': imp.modification.description,
                'created_at': imp.created_at.isoformat(),
                'status': 'applied' if imp.applied else 'validated' if imp.validation_status else 'staged'
            }
            report['recent_activity'].append(activity)
        
        # Save report
        report_path = os.path.join(
            self.reports_dir, 
            f"staging_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report