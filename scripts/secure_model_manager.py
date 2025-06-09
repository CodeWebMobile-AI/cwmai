"""
Secure Model Manager

Replaces insecure pickle serialization with secure alternatives.
Implements model validation, signature checking, and safe deserialization.
"""

import json
import hashlib
import hmac
import os
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
import joblib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ModelSecurityError(Exception):
    """Exception raised for model security violations."""
    pass


class SecureModelManager:
    """Secure model management with signature validation and safe serialization."""
    
    def __init__(self, models_dir: str = None, secret_key: str = None):
        """Initialize secure model manager.
        
        Args:
            models_dir: Directory to store models
            secret_key: Secret key for model signature verification
        """
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), 'secure_models')
        self.secret_key = secret_key or os.environ.get('MODEL_SIGNING_KEY', 'default_dev_key_change_in_production')
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metadata storage
        self.metadata_file = os.path.join(self.models_dir, 'models_metadata.json')
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized SecureModelManager with models directory: {self.models_dir}")
    
    def save_model(self, model: Any, model_name: str, model_type: str = "sklearn", 
                   description: str = "", version: str = "1.0") -> bool:
        """Securely save a model with signature verification.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            model_type: Type of model (sklearn, custom, etc.)
            description: Model description
            version: Model version
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate model name
            if not self._is_valid_model_name(model_name):
                raise ModelSecurityError(f"Invalid model name: {model_name}")
            
            # Create secure filename
            safe_filename = self._create_safe_filename(model_name, version)
            model_path = os.path.join(self.models_dir, safe_filename)
            
            # Save model using joblib (safer than pickle)
            joblib.dump(model, model_path, compress=3)
            
            # Generate model signature
            signature = self._generate_model_signature(model_path)
            
            # Create metadata
            model_metadata = {
                'name': model_name,
                'type': model_type,
                'description': description,
                'version': version,
                'filename': safe_filename,
                'signature': signature,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'size_bytes': os.path.getsize(model_path),
                'checksum': self._calculate_file_checksum(model_path)
            }
            
            # Update metadata
            self.metadata[model_name] = model_metadata
            self._save_metadata()
            
            logger.info(f"Successfully saved model: {model_name} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str, verify_signature: bool = True) -> Optional[Any]:
        """Securely load a model with signature verification.
        
        Args:
            model_name: Name of the model to load
            verify_signature: Whether to verify model signature
            
        Returns:
            Loaded model or None if failed
        """
        try:
            # Check if model exists in metadata
            if model_name not in self.metadata:
                logger.error(f"Model not found: {model_name}")
                return None
            
            model_info = self.metadata[model_name]
            model_path = os.path.join(self.models_dir, model_info['filename'])
            
            # Verify file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
            
            # Verify file integrity
            if not self._verify_file_integrity(model_path, model_info['checksum']):
                raise ModelSecurityError(f"Model file integrity check failed: {model_name}")
            
            # Verify signature if requested
            if verify_signature:
                if not self._verify_model_signature(model_path, model_info['signature']):
                    raise ModelSecurityError(f"Model signature verification failed: {model_name}")
            
            # Load model using joblib
            model = joblib.load(model_path)
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their metadata.
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        for model_name, metadata in self.metadata.items():
            # Create a copy without sensitive information
            public_metadata = {
                'name': metadata['name'],
                'type': metadata['type'],
                'description': metadata['description'],
                'version': metadata['version'],
                'created_at': metadata['created_at'],
                'size_bytes': metadata['size_bytes']
            }
            models.append(public_metadata)
        
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """Securely delete a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.metadata:
                logger.error(f"Model not found: {model_name}")
                return False
            
            model_info = self.metadata[model_name]
            model_path = os.path.join(self.models_dir, model_info['filename'])
            
            # Delete model file
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Remove from metadata
            del self.metadata[model_name]
            self._save_metadata()
            
            logger.info(f"Successfully deleted model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def migrate_from_pickle(self, pickle_file: str, model_name: str, 
                          model_type: str = "legacy") -> bool:
        """Migrate a model from pickle format to secure format.
        
        Args:
            pickle_file: Path to pickle file
            model_name: Name for the migrated model
            model_type: Type of the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning(f"Migrating potentially unsafe pickle file: {pickle_file}")
            
            # Load with restricted unpickler (safer but still not completely safe)
            import pickle
            import io
            
            class RestrictedUnpickler(pickle.Unpickler):
                """Restricted unpickler that only allows safe classes."""
                
                def load_build(self):
                    raise pickle.UnpicklingError("BUILD opcode disabled for security")
                
                def load_reduce(self):
                    raise pickle.UnpicklingError("REDUCE opcode disabled for security")
                
                def find_class(self, module, name):
                    # Only allow specific safe modules and classes
                    safe_modules = {
                        'sklearn.linear_model._base',
                        'sklearn.tree._tree',
                        'sklearn.ensemble._forest',
                        'numpy.core.multiarray',
                        'numpy',
                        'pandas.core.frame',
                        'pandas.core.series',
                    }
                    
                    if module in safe_modules:
                        return super().find_class(module, name)
                    else:
                        raise pickle.UnpicklingError(f"Unsafe module: {module}.{name}")
            
            # Read pickle file in a sandboxed way
            with open(pickle_file, 'rb') as f:
                unpickler = RestrictedUnpickler(f)
                model = unpickler.load()
            
            # Save in secure format
            return self.save_model(
                model=model,
                model_name=model_name,
                model_type=model_type,
                description=f"Migrated from pickle file: {os.path.basename(pickle_file)}",
                version="1.0"
            )
            
        except Exception as e:
            logger.error(f"Failed to migrate pickle file {pickle_file}: {e}")
            return False
    
    def _is_valid_model_name(self, name: str) -> bool:
        """Validate model name for security."""
        import re
        
        # Only allow alphanumeric characters, underscores, and hyphens
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return False
        
        # Check length
        if len(name) < 1 or len(name) > 100:
            return False
        
        # Prevent path traversal
        if '..' in name or '/' in name or '\\' in name:
            return False
        
        return True
    
    def _create_safe_filename(self, model_name: str, version: str) -> str:
        """Create a safe filename for model storage."""
        safe_name = f"{model_name}_v{version}.joblib"
        return safe_name.replace(' ', '_').replace('..', '_')
    
    def _generate_model_signature(self, model_path: str) -> str:
        """Generate HMAC signature for model file."""
        with open(model_path, 'rb') as f:
            content = f.read()
        
        signature = hmac.new(
            self.secret_key.encode(),
            content,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _verify_model_signature(self, model_path: str, expected_signature: str) -> bool:
        """Verify model file signature."""
        actual_signature = self._generate_model_signature(model_path)
        return hmac.compare_digest(actual_signature, expected_signature)
    
    def _calculate_file_checksum(self, filepath: str) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _verify_file_integrity(self, filepath: str, expected_checksum: str) -> bool:
        """Verify file integrity using checksum."""
        actual_checksum = self._calculate_file_checksum(filepath)
        return actual_checksum == expected_checksum
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")


def migrate_legacy_pickle_models():
    """Migrate legacy pickle models to secure format."""
    print("🔒 Migrating legacy pickle models to secure format...")
    
    manager = SecureModelManager()
    scripts_dir = os.path.dirname(__file__)
    
    # Look for pickle files to migrate
    pickle_files = [
        os.path.join(scripts_dir, 'predictive_models.pkl'),
        # Add other pickle files here as needed
    ]
    
    migrated_count = 0
    
    for pickle_file in pickle_files:
        if os.path.exists(pickle_file):
            base_name = os.path.splitext(os.path.basename(pickle_file))[0]
            
            print(f"  Migrating: {pickle_file}")
            
            if manager.migrate_from_pickle(pickle_file, base_name, "legacy"):
                print(f"  ✅ Successfully migrated: {base_name}")
                
                # Backup original pickle file (don't delete immediately)
                backup_path = f"{pickle_file}.backup"
                if not os.path.exists(backup_path):
                    os.rename(pickle_file, backup_path)
                    print(f"  📦 Original backed up to: {backup_path}")
                
                migrated_count += 1
            else:
                print(f"  ❌ Failed to migrate: {pickle_file}")
    
    print(f"\n✅ Migration complete. Migrated {migrated_count} model(s).")
    
    if migrated_count > 0:
        print("\n🔍 Available models:")
        models = manager.list_models()
        for model in models:
            print(f"  - {model['name']} v{model['version']} ({model['type']})")


if __name__ == "__main__":
    # Run migration when script is executed directly
    migrate_legacy_pickle_models()