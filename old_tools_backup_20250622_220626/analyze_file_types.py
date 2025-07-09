"""
AI-Generated Tool: analyze_file_types
Description: Analyze data for file types
Generated: 2025-06-15T11:05:48.161889+00:00
Requirements: 
        Tool Name: analyze_file_types
        Intent: Analyze data for file types
        Expected Parameters: {}
        Category: analytics
        
        The tool should:
        1. Perform comprehensive analysis of items
2. Generate insights and recommendations
3. Return detailed report
        
"""

"""
File Type Analysis Module

This module provides functionality to analyze file types within a system or dataset.
It performs comprehensive analysis, generates insights, and returns detailed reports
about file type distributions, potential issues, and optimization recommendations.
"""

from scripts.state_manager import StateManager  # Assuming this exists
from typing import Dict, Any, List, Optional
import logging
import functools
import os
import hashlib
import json  # Import json for explicit use
import datetime # Import datetime for use in cache
import mimetypes

# Import for entropy calculation
import math

# Import for statistical analysis
import numpy as np

__description__ = "Analyze data for file types and generate insights"
__parameters__ = {}
__examples__ = [
    {"description": "Basic file type analysis", "parameters": {}},
]

# Configure logging (consider moving to a separate module)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a cache directory (can be configurable)
CACHE_DIR = ".file_analysis_cache"
os.makedirs(CACHE_DIR, exist_ok=True)  # Create if it doesn't exist

# Cache expiration time (in seconds)
CACHE_EXPIRATION = 3600  # 1 hour

# --- Utility Functions ---

def get_file_extension(filename: str) -> str:
    """Extracts the file extension from a filename."""
    try:
        return os.path.splitext(filename)[1].lstrip('.').lower()
    except Exception as e:
        logging.warning(f"Failed to extract extension from {filename}: {e}")
        return ""  # or "unknown" if desired

def calculate_file_size_in_mb(size_in_bytes: int) -> float:
    """Converts file size from bytes to megabytes."""
    return round(size_in_bytes / (1024 * 1024), 2)

def format_file_size(size_in_bytes: int) -> str:
    """Formats file size into a human-readable format (KB, MB, GB)."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"


def is_text_file(file_path: str) -> bool:
    """Tries to determine if a file is a text file based on its content."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.read(512)  # Read the first 512 bytes
        return True
    except UnicodeDecodeError:
        return False
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return False
    except Exception as e:
        logging.warning(f"Error checking if {file_path} is text: {e}")
        return False

def get_file_creation_date(file_path: str) -> Optional[datetime.datetime]:
    """Attempts to retrieve the file's creation date."""
    try:
        timestamp = os.path.getctime(file_path)  # Creation time on Windows, last metadata change on Unix
        return datetime.datetime.fromtimestamp(timestamp)
    except OSError as e:
        logging.warning(f"Could not get creation date for {file_path}: {e}")
        return None

def calculate_file_entropy(file_path: str) -> Optional[float]:
    """Calculates the entropy of a file."""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return 0.0 # Empty file has no entropy
        entropy = 0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log2(p_x)
        return entropy
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.warning(f"Error calculating entropy for {file_path}: {e}")
        return None

def get_mime_type(file_path: str) -> str:
    """
    Determines the MIME type of a file based on its extension.
    """
    mime_type, encoding = mimetypes.guess_type(file_path)
    return mime_type if mime_type else "application/octet-stream"  # Default MIME type

def get_file_owner(file_path: str) -> Optional[str]:
    """Attempts to retrieve the file's owner (username)."""
    try:
        import pwd  # Requires `pwd` module (usually Unix-like systems)
        uid = os.stat(file_path).st_uid
        return pwd.getpwuid(uid).pw_name
    except ImportError:
        logging.warning("`pwd` module not available. Cannot determine file owner.")
        return None
    except KeyError:
        logging.warning(f"Could not determine owner for file {file_path} (UID not found).")
        return None
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.warning(f"Error getting file owner for {file_path}: {e}")
        return None

def get_file_last_accessed_date(file_path: str) -> Optional[datetime.datetime]:
    """Attempts to retrieve the file's last accessed date."""
    try:
        timestamp = os.path.getatime(file_path)  # Last accessed time
        return datetime.datetime.fromtimestamp(timestamp)
    except OSError as e:
        logging.warning(f"Could not get last accessed date for {file_path}: {e}")
        return None

def is_binary_file(file_path: str, block_size: int = 512) -> bool:
    """
    Detects if a file is likely a binary file by checking for null bytes.
    Reads the file in chunks to avoid loading large files into memory.
    """
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                if b'\0' in chunk:
                    return True
        return False
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return False
    except Exception as e:
        logging.warning(f"Error checking if {file_path} is binary: {e}")
        return False

def calculate_shannon_entropy(data: bytes) -> float:
    """Calculates the Shannon entropy of the given data."""
    if not data:
        return 0.0

    entropy = 0
    data_length = len(data)
    seen_bytes = set()  # Keep track of bytes already processed

    for byte in data:
        if byte not in seen_bytes:
            p_x = float(data.count(byte)) / data_length
            if p_x > 0:
                entropy -= p_x * math.log2(p_x)
            seen_bytes.add(byte)  # Add the byte to the set of processed bytes

    return entropy

def get_file_encoding(file_path: str) -> Optional[str]:
    """Attempts to determine the file encoding."""
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result['encoding']
    except ImportError:
        logging.warning("chardet module not available. Cannot determine file encoding.")
        return None
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.warning(f"Error determining file encoding for {file_path}: {e}")
        return None


# --- Caching ---

def generate_cache_key(files: List[Dict[str, Any]]) -> str:
    """Generates a cache key based on the state of the files being analyzed."""
    # Create a hash from the sorted list of file paths and sizes to ensure consistency.
    file_data = sorted([(file.get('path', ''), file.get('size', 0), file.get('last_modified', 0)) for file in files]) # Include last_modified for more robust cache invalidation
    hash_input = str(file_data).encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()


def cached_analysis(func):
    """Decorator to cache the results of the file type analysis."""
    @functools.wraps(func)
    async def wrapper(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper function to handle caching logic."""
        try:
            state_manager = StateManager()
            state = state_manager.load_state()
            files = state.get('files', [])

            cache_key = generate_cache_key(files)
            cache_file = os.path.join(CACHE_DIR, f"analysis_cache_{cache_key}.json")

            # Check if cache file exists and is not expired
            if os.path.exists(cache_file):
                # Get the file's modification time
                file_modified_time = os.path.getmtime(cache_file)
                # Calculate the age of the cache file
                cache_age = datetime.datetime.now().timestamp() - file_modified_time

                if cache_age < CACHE_EXPIRATION:
                    logging.info("Cache hit! Loading analysis from cache.")
                    try:
                        with open(cache_file, 'r') as f:
                            return json.load(f)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Cache file corrupted: {e}.  Invalidating cache.")
                        os.remove(cache_file)  # Remove the corrupted cache file
                        return await func(**kwargs)  # Re-run analysis
                else:
                    logging.info("Cache expired. Re-running analysis.")
                    os.remove(cache_file) # Remove the expired cache file
                    return await func(**kwargs)

            # If not in cache, perform the analysis
            result = await func(**kwargs)

            # Cache the result
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, default=str) # Use default=str to handle datetime objects
                logging.info("Analysis result cached.")
            except Exception as e:
                logging.error(f"Failed to write to cache file: {e}")
            return result

        except Exception as e:
            logging.warning(f"Cache operation failed: {e}. Proceeding without cache.")
            return await func(**kwargs)  # Fallback to non-cached execution
    return wrapper

# --- Input Validation ---

def is_valid_file_data(file: Dict[str, Any]) -> bool:
    """Validates that the file data contains necessary fields."""
    if not isinstance(file, dict):
        logging.warning("Invalid file format: File data must be a dictionary.")
        return False
    if not all(key in file for key in ('path', 'type', 'size')):
        logging.warning(f"Invalid file format: File data must contain 'path', 'type', and 'size' keys. Found: {file.keys()}")
        return False
    if not isinstance(file.get('path'), str):
        logging.warning(f"Invalid file format: 'path' must be a string. Found type: {type(file.get('path'))}")
        return False
    if not isinstance(file.get('type'), str):
        logging.warning(f"Invalid file format: 'type' must be a string. Found type: {type(file.get('type'))}")
        return False

    size = file.get('size')
    if not isinstance(size, (int, float)):
        logging.warning(f"Invalid file format: 'size' must be a number. Found type: {type(size)}")
        return False
    if size < 0:
        logging.warning(f"Invalid file format: 'size' cannot be negative. Found size: {size}")
        return False

    path = file.get('path')
    if not os.path.exists(path):
        logging.warning(f"File does not exist at path: {path}")
        return False
    
    #Check if the file is accessible (read permission)
    if not os.access(path, os.R_OK):
        logging.warning(f"File not accessible (read permission denied): {path}")
        return False


    return True

# --- Recommendation Summarization ---

def summarize_recommendations(recommendations: List[Dict[str, Any]]) -> str:
    """Creates a concise summary of the recommendations."""
    if not recommendations:
        return "No specific recommendations at this time."

    summary_parts = []
    for recommendation in recommendations:
        issue = recommendation.get('issue', 'Unknown issue')
        count = recommendation.get('count', 0)
        summary_parts.append(f"{issue}: {count} instances found.")

    return "; ".join(summary_parts)

# --- Anomaly Detection ---

def detect_file_size_anomalies(file_sizes: List[int], threshold: float = 3.0) -> List[int]:
    """Detects file size anomalies using the z-score method."""

    if not file_sizes:
        return []

    file_sizes_np = np.array(file_sizes)
    mean = np.mean(file_sizes_np)
    std = np.std(file_sizes_np)

    if std == 0:  # Handle the case where all file sizes are the same.
        return []  # No anomalies if all sizes are identical

    z_scores = np.abs((file_sizes_np - mean) / std)
    anomalies_indices = np.where(z_scores > threshold)[0].tolist()  # Return indices as a list
    return anomalies_indices

# --- Data Enrichment ---

def enrich_file_data(file: Dict[str, Any]) -> Dict[str, Any]:
    """Enriches the file data with additional information."""
    file_path = file.get('path')

    try:
        file['extension'] = get_file_extension(file_path)
        file['mime_type'] = get_mime_type(file_path)
        file['creation_date'] = get_file_creation_date(file_path)
        file['last_accessed'] = get_file_last_accessed_date(file_path)
        file['entropy'] = calculate_file_entropy(file_path)
        file['owner'] = get_file_owner(file_path)
        file['is_binary'] = is_binary_file(file_path) #Determine if file is binary
        file['last_modified'] = os.path.getmtime(file_path) # Add last modified timestamp
        file['encoding'] = get_file_encoding(file_path) # Determine file encoding
        return file
    except Exception as e:
        logging.warning(f"Failed to enrich file data for {file_path}: {e}")
        return file  # Return the original file data even if enrichment fails

# --- Additional Analysis Functions ---

def analyze_mime_types(mime_types: Dict[str, str]) -> Dict[str, int]:
    """Analyzes the distribution of MIME types."""
    mime_type_counts: Dict[str, int] = {}
    for mime_type in mime_types.values():
        mime_type_counts[mime_type] = mime_type_counts.get(mime_type, 0) + 1
    return mime_type_counts

def analyze_file_age(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes the age of files and provides insights."""
    now = datetime.datetime.now()
    age_data = {
        "num_files": len(files),
        "average_age": None,
        "oldest_file_path": None,
        "oldest_file_age": None,
        "newest_file_path": None,
        "newest_file_age": None
    }

    if not files:
        return age_data

    ages = []
    oldest_file = None
    newest_file = None

    for file in files:
        creation_date = file.get("creation_date")
        if creation_date:
            age = now - creation_date
            ages.append(age.total_seconds())

            if oldest_file is None or creation_date < oldest_file["creation_date"]:
                oldest_file = file
                oldest_file["age"] = age.total_seconds()

            if newest_file is None or creation_date > newest_file["creation_date"]:
                newest_file = file
                newest_file["age"] = age.total_seconds()

    if ages:
        average_age_seconds = sum(ages) / len(ages)
        average_age = datetime.timedelta(seconds=average_age_seconds)
        age_data["average_age"] = str(average_age)

    if oldest_file:
        age_data["oldest_file_path"] = oldest_file["path"]
        age_data["oldest_file_age"] = str(datetime.timedelta(seconds=oldest_file["age"]))

    if newest_file:
        age_data["newest_file_path"] = newest_file["path"]
        age_data["newest_file_age"] = str(datetime.timedelta(seconds=newest_file["age"]))

    return age_data

def analyze_file_permissions(file_path: str) -> Dict[str, bool]:
    """Analyzes file permissions and returns a dictionary of read, write, and execute permissions."""
    try:
        permissions = {
            "read": os.access(file_path, os.R_OK),
            "write": os.access(file_path, os.W_OK),
            "execute": os.access(file_path, os.X_OK)
        }
        return permissions
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return {"read": False, "write": False, "execute": False}
    except Exception as e:
        logging.warning(f"Error analyzing file permissions for {file_path}: {e}")
        return {"read": False, "write": False, "execute": False}

def analyze_file_encoding_distribution(files: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyzes the distribution of file encodings."""
    encoding_counts: Dict[str, int] = {}
    for file in files:
        encoding = file.get("encoding")
        if encoding:
            encoding_counts[encoding] = encoding_counts.get(encoding, 0) + 1
    return encoding_counts

# --- Main Analysis Function ---

@cached_analysis  # Apply caching decorator
async def analyze_file_types(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of file types in the system.

    Returns:
        Dict[str, Any]: Analysis report containing statistics, insights,
                        and recommendations about file types.

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If analysis cannot be completed
    """
    try:
        # Initialize state manager and load current state
        state_manager = StateManager()
        try:
            state = state_manager.load_state()
        except Exception as e:
            logging.error(f"Failed to load state from StateManager: {e}")
            raise RuntimeError(f"Failed to load state: {e}")

        # Validate no unexpected parameters were provided
        if kwargs:
            raise ValueError("This tool doesn't accept any parameters")

        # Get file data from state (example implementation)
        files = state.get('files', [])

        if not files:
            return {
                "total_files": 0,
                "summary": "No files found for analysis",
                "recommendations": [],
                "recommendation_summary": "No specific recommendations at this time."  # Comprehensive output
            }

        # Perform analysis
        file_types = {}
        suspicious_files = []
        large_files = []
        invalid_files = []
        text_files = [] # store the paths of text files
        file_sizes_list = [] # store sizes for anomaly detection
        file_paths_list = [] # corresponding file paths
        entropy_values = {} # Store file entropy
        mime_types = {} # Store MIME types of files
        duplicate_hashes = {} # Store hashes of duplicate files
        file_owners = {} # Store file owners
        binary_files = [] # Store list of binary files for analysis
        file_permissions = {} # Store file permissions
        file_encodings = {} # Store the detected file encoding


        for file in files:
            try:  # Error handling within the loop for individual files
                if not is_valid_file_data(file):
                    invalid_files.append(file)
                    continue  # Skip to the next file

                # Enrich file data (add extension, MIME type, etc.)
                file = enrich_file_data(file)

                # Count file types
                file_type = file.get('type', 'unknown').lower() # Normalize to lowercase for consistency
                file_types[file_type] = file_types.get(file_type, 0) + 1

                # Identify potentially suspicious files
                if file_type in ('exe', 'dll', 'bat', 'sh', 'vbs', 'ps1'): # Add more suspicious types
                    suspicious_files.append(file.get('path'))

                # Identify large files (>100MB)
                size = file.get('size', 0)
                size_mb = calculate_file_size_in_mb(size)  # Convert to MB for reporting
                if size > 100 * 1024 * 1024:  # 100MB
                    large_files.append({
                        'path': file.get('path'),
                        'size': size,
                        'size_mb': size_mb,  # Add size in MB for easier reading
                        'type': file_type,
                        'creation_date': file.get('creation_date'), # Added file creation date,
                        'owner': file.get('owner'),
                        'last_accessed': file.get('last_accessed')
                    })
                #Identify text files
                file_path = file.get('path')
                if file_type in ('txt', 'log', 'csv') or is_text_file(file_path):
                    text_files.append(file_path)

                # Add to list for file size anomaly detection
                file_sizes_list.append(size)
                file_paths_list.append(file_path)

                # Store file entropy
                entropy = file.get('entropy')
                if entropy is not None:
                    entropy_values[file_path] = entropy

                # Store MIME type
                mime_type = file.get('mime_type')
                mime_types[file_path] = mime_type

                # Store file owner
                owner = file.get('owner')
                if owner is not None:
                    file_owners[file_path] = owner
                
                # Store binary file
                is_binary = file.get('is_binary')
                if is_binary:
                    binary_files.append(file_path)

                # Calculate file hash for duplicate detection
                try:
                    with open(file_path, "rb") as f:
                        file_content = f.read()
                        file_hash = hashlib.md5(file_content).hexdigest()

                        if file_hash in duplicate_hashes:
                            duplicate_hashes[file_hash].append(file_path)
                        else:
                            duplicate_hashes[file_hash] = [file_paths_list[-1]] # Use the last file path because it is certain to exist
                except Exception as e:
                    logging.warning(f"Error calculating hash for {file_path}: {e}")

                # Analyze file permissions
                permissions = analyze_file_permissions(file_path)
                file_permissions[file_path] = permissions
                
                # Store file encoding
                encoding = file.get('encoding')
                if encoding:
                    file_encodings[file_path] = encoding


            except Exception as e:
                logging.error(f"Error processing file {file.get('path', 'Unknown')}: {e}")
                # Consider continuing to the next file or re-raising the error based on the use case

        # Generate insights and recommendations
        total_files = len(files) - len(invalid_files)  # Exclude invalid files from total count
        most_common_type = max(file_types.items(), key=lambda x: x[1], default=('unknown', 0))[0]
        unique_types = len(file_types)

        recommendations = []
        if suspicious_files:
            recommendations.append({
                'issue': 'Potentially suspicious files detected',
                'count': len(suspicious_files),
                'action': 'Review these files for security risks',
                'items': suspicious_files[:10]  # Show first 10 as examples
            })

        if large_files:
            recommendations.append({
                'issue': 'Large files detected',
                'count': len(large_files),
                'action': 'Consider archiving or compressing these files',
                'items': large_files[:5]  # Show first 5 as examples
            })

        if invalid_files:
            recommendations.append({
                'issue': 'Invalid file data detected',
                'count': len(invalid_files),
                'action': 'Check the file data source for errors',
                'items': invalid_files[:5]
            })

         # Text files recommendation
        if text_files:
            recommendations.append({
                'issue': 'Text files detected',
                'count': len(text_files),
                'action': 'Consider analyzing the content of these text files for sensitive data',
                'items': text_files[:5]
            })

        # Add recommendation to investigate uncommon file types
        uncommon_threshold = 0.01  # 1% threshold - adjust as needed
        total_valid_files = len(files) - len(invalid_files)
        uncommon_file_types = {file_type: count for file_type, count in file_types.items() if count / total_valid_files < uncommon_threshold}

        if uncommon_file_types:
            recommendations.append({
                'issue': 'Uncommon file types detected',
                'count': len(uncommon_file_types),
                'action': 'Investigate the purpose and origin of these file types',
                'items': list(uncommon_file_types.keys())[:5]  # Show first 5 as examples
            })

        # File size anomaly detection
        anomaly_indices = detect_file_size_anomalies(file_sizes_list)
        anomalous_files = []
        for index in anomaly_indices:
            anomalous_files.append({
                'path': file_paths_list[index],
                'size': file_sizes_list[index],
                'size_formatted': format_file_size(file_sizes_list[index])
            })

        if anomalous_files:
            recommendations.append({
                'issue': 'File size anomalies detected',
                'count': len(anomalous_files),
                'action': 'Investigate these files as they deviate significantly in size compared to others',
                'items': anomalous_files[:5]
            })

        # High Entropy Files Detection
        high_entropy_threshold = 7.0 # Adjust based on needs
        high_entropy_files = {path: entropy for path, entropy in entropy_values.items() if entropy > high_entropy_threshold}
        if high_entropy_files:
             recommendations.append({
                'issue': 'High entropy files detected',
                'count': len(high_entropy_files),
                'action': 'Investigate these files. High entropy may indicate encryption, compression, or obfuscation',
                'items': list(high_entropy_files.keys())[:5]
             })

        # Duplicate files detection
        duplicate_files = {hash_value: files for hash_value, files in duplicate_hashes.items() if len(files) > 1}
        if duplicate_files:
            recommendations.append({
                'issue': 'Duplicate files detected',
                'count': sum(len(files) for files in duplicate_files.values()),
                'action': 'Consider removing duplicate files to save space',
                'items': {hash_value: files[:3] for hash_value, files in duplicate_files.items()} # Show first 3 files for each hash
            })

        # Inconsistent file owner detection
        if len(set(file_owners.values())) > 1:
            recommendations.append({
                'issue': 'Inconsistent file owners detected',
                'count': len(set(file_owners.values())),
                'action': 'Investigate the different file owners to ensure proper access control',
                'items': file_owners
            })
            
        # Binary files detection
        if binary_files:
            recommendations.append({
                'issue': 'Binary files detected',
                'count': len(binary_files),
                'action': 'Analyze these files for potential security risks',
                'items': binary_files[:5]
            })
        
        # Insecure File Permissions Detection
        insecure_permissions = {path: permissions for path, permissions in file_permissions.items() if permissions.get("write", False) and permissions.get("execute", False)}
        if insecure_permissions:
            recommendations.append({
                'issue': 'Insecure file permissions detected',
                'count': len(insecure_permissions),
                'action': 'Restrict write and execute permissions on these files',
                'items': insecure_permissions
            })

        # Analyze MIME Types
        mime_type_analysis = analyze_mime_types(mime_types)

        # Analyze File Age
        file_age_analysis = analyze_file_age(files)

        # Analyze File Encoding Distribution
        file_encoding_analysis = analyze_file_encoding_distribution(files)

        # Prepare final report
        report = {
            "total_files": total_files,
            "file_type_distribution": file_types,
            "most_common_type": most_common_type,
            "unique_file_types": unique_types,
            "suspicious_files_count": len(suspicious_files),
            "large_files_count": len(large_files),
            "invalid_files_count": len(invalid_files), # Additional information
            "text_files_count": len(text_files),
            "binary_files_count": len(binary_files),
            "summary": f"Analyzed {total_files} files with {unique_types} different types.  {len(invalid_files)} files were skipped due to invalid data.  Most common type: {most_common_type}.",
            "recommendations": recommendations,
            "recommendation_summary": summarize_recommendations(recommendations),  # Comprehensive output
            "entropy_values": entropy_values, # Include entropy data in the report
            "mime_types": mime_types, # Include MIME types in the report
            "mime_type_analysis": mime_type_analysis,
            "duplicate_files_count": sum(len(files) for files in duplicate_files.values()) if duplicate_files else 0, # Count of duplicate files
            "file_owners": file_owners, # Include file owners in the report
            "file_enrichment_metadata": ['extension', 'mime_type', 'creation_date', 'entropy', 'owner', 'last_modified', 'last_accessed', 'is_binary', 'encoding'],
            "file_age_analysis": file_age_analysis,
            "file_permissions": file_permissions,
            "insecure_permissions_count": len(insecure_permissions),
            "file_encodings": file_encodings, # Include detected file encodings in the report
            "file_encoding_analysis": file_encoding_analysis # Include file encoding analysis in the report
        }

        return report

    except ValueError as ve:
        logging.error(f"Parameter validation error: {str(ve)}")
        raise
    except RuntimeError as re:
        logging.error(f"Runtime error during analysis: {str(re)}")  # More specific logging
        raise
    except Exception as e:
        logging.exception(f"Unexpected error during file type analysis: {str(e)}") # Capture full stack trace
        raise RuntimeError(f"File type analysis failed: {str(e)}")