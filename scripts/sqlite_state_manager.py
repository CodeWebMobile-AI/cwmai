"""
SQLite State Manager

Drop-in replacement for Redis-based state management using SQLite.
Provides the same interface but with no connection limits and better reliability.
"""

import sqlite3
import json
import threading
import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import logging


class SQLiteStateManager:
    """SQLite-based state manager with Redis-compatible interface."""
    
    def __init__(self, db_path: str = "cwmai_state.db", enable_wal: bool = True):
        """Initialize SQLite state manager.
        
        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._local = threading.local()
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
        # Enable WAL mode for better concurrency
        if enable_wal:
            self._execute("PRAGMA journal_mode=WAL")
            self._execute("PRAGMA synchronous=NORMAL")
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_expiry = {}
        self._cache_ttl = 60  # 1 minute cache
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query with automatic retry."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise
    
    def _init_db(self):
        """Initialize database schema."""
        # Key-value store for state
        self._execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ttl INTEGER DEFAULT NULL
            )
        """)
        
        # Sets for Redis SET operations
        self._execute("""
            CREATE TABLE IF NOT EXISTS sets (
                set_key TEXT NOT NULL,
                member TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (set_key, member)
            )
        """)
        
        # Hashes for Redis HASH operations
        self._execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                hash_key TEXT NOT NULL,
                field TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hash_key, field)
            )
        """)
        
        # Lists for Redis LIST operations
        self._execute("""
            CREATE TABLE IF NOT EXISTS lists (
                list_key TEXT NOT NULL,
                position INTEGER NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (list_key, position)
            )
        """)
        
        # Create indexes for performance
        self._execute("CREATE INDEX IF NOT EXISTS idx_state_updated ON state(updated_at)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_sets_key ON sets(set_key)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_hashes_key ON hashes(hash_key)")
        self._execute("CREATE INDEX IF NOT EXISTS idx_lists_key ON lists(list_key)")
    
    # Redis-compatible String operations
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key (Redis GET)."""
        # Check cache first
        if key in self._cache:
            if self._cache_expiry.get(key, 0) > time.time():
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_expiry[key]
        
        cursor = self._execute(
            "SELECT value, ttl, created_at FROM state WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row:
            # Check TTL
            if row['ttl']:
                created = datetime.fromisoformat(row['created_at'])
                if (datetime.now(timezone.utc) - created).total_seconds() > row['ttl']:
                    self.delete(key)
                    return None
            
            # Update cache
            self._cache[key] = row['value']
            self._cache_expiry[key] = time.time() + self._cache_ttl
            
            return row['value']
        return None
    
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value pair (Redis SET)."""
        try:
            value_str = value if isinstance(value, str) else json.dumps(value)
            
            self._execute("""
                INSERT OR REPLACE INTO state (key, value, ttl, updated_at) 
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (key, value_str, ex))
            
            # Update cache
            self._cache[key] = value_str
            self._cache_expiry[key] = time.time() + self._cache_ttl
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key (Redis DEL)."""
        cursor = self._execute("DELETE FROM state WHERE key = ?", (key,))
        
        # Clear from cache
        self._cache.pop(key, None)
        self._cache_expiry.pop(key, None)
        
        return cursor.rowcount > 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists (Redis EXISTS)."""
        cursor = self._execute("SELECT 1 FROM state WHERE key = ? LIMIT 1", (key,))
        return cursor.fetchone() is not None
    
    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on key (Redis EXPIRE)."""
        cursor = self._execute(
            "UPDATE state SET ttl = ? WHERE key = ?",
            (seconds, key)
        )
        return cursor.rowcount > 0
    
    # Redis-compatible Hash operations
    
    def hset(self, name: str, key: str, value: str) -> int:
        """Set hash field (Redis HSET)."""
        try:
            value_str = value if isinstance(value, str) else json.dumps(value)
            
            self._execute("""
                INSERT OR REPLACE INTO hashes (hash_key, field, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (name, key, value_str))
            return 1
        except Exception as e:
            self.logger.error(f"Error in hset: {e}")
            return 0
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field (Redis HGET)."""
        cursor = self._execute(
            "SELECT value FROM hashes WHERE hash_key = ? AND field = ?",
            (name, key)
        )
        row = cursor.fetchone()
        return row['value'] if row else None
    
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields (Redis HGETALL)."""
        cursor = self._execute(
            "SELECT field, value FROM hashes WHERE hash_key = ?",
            (name,)
        )
        return {row['field']: row['value'] for row in cursor.fetchall()}
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields (Redis HDEL)."""
        placeholders = ','.join(['?'] * len(keys))
        cursor = self._execute(
            f"DELETE FROM hashes WHERE hash_key = ? AND field IN ({placeholders})",
            (name,) + keys
        )
        return cursor.rowcount
    
    # Redis-compatible Set operations
    
    def sadd(self, name: str, *values: str) -> int:
        """Add members to set (Redis SADD)."""
        added = 0
        for value in values:
            try:
                self._execute(
                    "INSERT OR IGNORE INTO sets (set_key, member) VALUES (?, ?)",
                    (name, value)
                )
                added += 1
            except sqlite3.IntegrityError:
                pass
        return added
    
    def srem(self, name: str, *values: str) -> int:
        """Remove members from set (Redis SREM)."""
        removed = 0
        for value in values:
            cursor = self._execute(
                "DELETE FROM sets WHERE set_key = ? AND member = ?",
                (name, value)
            )
            removed += cursor.rowcount
        return removed
    
    def smembers(self, name: str) -> Set[str]:
        """Get all set members (Redis SMEMBERS)."""
        cursor = self._execute(
            "SELECT member FROM sets WHERE set_key = ?",
            (name,)
        )
        return {row['member'] for row in cursor.fetchall()}
    
    def sismember(self, name: str, value: str) -> bool:
        """Check if value is in set (Redis SISMEMBER)."""
        cursor = self._execute(
            "SELECT 1 FROM sets WHERE set_key = ? AND member = ? LIMIT 1",
            (name, value)
        )
        return cursor.fetchone() is not None
    
    # Redis-compatible List operations
    
    def lpush(self, name: str, *values: str) -> int:
        """Push values to list head (Redis LPUSH)."""
        with self._lock:
            # Get current max position
            cursor = self._execute(
                "SELECT MIN(position) as min_pos FROM lists WHERE list_key = ?",
                (name,)
            )
            row = cursor.fetchone()
            position = (row['min_pos'] - 1) if row['min_pos'] is not None else 0
            
            # Insert values
            for value in reversed(values):  # Reverse to maintain order
                self._execute(
                    "INSERT INTO lists (list_key, position, value) VALUES (?, ?, ?)",
                    (name, position, value)
                )
                position -= 1
            
            return len(values)
    
    def rpush(self, name: str, *values: str) -> int:
        """Push values to list tail (Redis RPUSH)."""
        with self._lock:
            # Get current max position
            cursor = self._execute(
                "SELECT MAX(position) as max_pos FROM lists WHERE list_key = ?",
                (name,)
            )
            row = cursor.fetchone()
            position = (row['max_pos'] + 1) if row['max_pos'] is not None else 0
            
            # Insert values
            for value in values:
                self._execute(
                    "INSERT INTO lists (list_key, position, value) VALUES (?, ?, ?)",
                    (name, position, value)
                )
                position += 1
            
            return len(values)
    
    def lpop(self, name: str) -> Optional[str]:
        """Pop from list head (Redis LPOP)."""
        with self._lock:
            cursor = self._execute("""
                SELECT position, value FROM lists 
                WHERE list_key = ? 
                ORDER BY position ASC 
                LIMIT 1
            """, (name,))
            
            row = cursor.fetchone()
            if row:
                self._execute(
                    "DELETE FROM lists WHERE list_key = ? AND position = ?",
                    (name, row['position'])
                )
                return row['value']
            return None
    
    def llen(self, name: str) -> int:
        """Get list length (Redis LLEN)."""
        cursor = self._execute(
            "SELECT COUNT(*) as count FROM lists WHERE list_key = ?",
            (name,)
        )
        row = cursor.fetchone()
        return row['count'] if row else 0
    
    # Additional utility methods
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (Redis KEYS)."""
        if pattern == "*":
            cursor = self._execute("SELECT key FROM state")
        else:
            # Convert Redis pattern to SQL LIKE
            sql_pattern = pattern.replace("*", "%").replace("?", "_")
            cursor = self._execute("SELECT key FROM state WHERE key LIKE ?", (sql_pattern,))
        
        return [row['key'] for row in cursor.fetchall()]
    
    def flushdb(self) -> bool:
        """Delete all keys (Redis FLUSHDB)."""
        try:
            self._execute("DELETE FROM state")
            self._execute("DELETE FROM sets")
            self._execute("DELETE FROM hashes")
            self._execute("DELETE FROM lists")
            self._cache.clear()
            self._cache_expiry.clear()
            return True
        except Exception as e:
            self.logger.error(f"Error flushing database: {e}")
            return False
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
    
    # Convenience methods for JSON data
    
    def get_json(self, key: str) -> Optional[Any]:
        """Get and parse JSON value."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None
    
    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set value as JSON."""
        try:
            return self.set(key, json.dumps(value), ex)
        except Exception as e:
            self.logger.error(f"Error setting JSON for key {key}: {e}")
            return False