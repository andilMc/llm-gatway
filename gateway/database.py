"""SQLite database management with AES encryption for API keys."""

import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("GATEWAY_DB_PATH", "/app/data/gateway.db")


def get_encryption_key() -> bytes:
    """Generate or retrieve encryption key from environment."""
    key_env = os.environ.get("GATEWAY_ENCRYPTION_KEY")
    if key_env:
        return base64.urlsafe_b64encode(key_env.encode()[:32].ljust(32, b"0"))

    # Derive key from a default secret (should be changed in production)
    secret = os.environ.get(
        "GATEWAY_SECRET", "llm-gateway-secret-key-change-in-production"
    )
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"llm-gateway-salt",
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
    return key


class DatabaseManager:
    """Manages SQLite database with encryption for API keys."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_data_dir()
        self._cipher = Fernet(get_encryption_key())
        self._init_db()

    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            # Providers table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS providers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    strategy TEXT DEFAULT 'sequential',
                    timeout REAL DEFAULT 120.0,
                    max_retries INTEGER DEFAULT 3,
                    config_json TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # API Keys table with encryption
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id INTEGER NOT NULL,
                    key_encrypted BLOB NOT NULL,
                    key_preview TEXT NOT NULL,
                    models_allowed TEXT,
                    status TEXT DEFAULT 'available',
                    error_count INTEGER DEFAULT 0,
                    session_start TIMESTAMP,
                    cooldown_until TIMESTAMP,
                    total_requests INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
                )
            """)

            # Models table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id INTEGER,
                    alias TEXT NOT NULL,
                    technical_name TEXT NOT NULL,
                    description TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE SET NULL
                )
            """)

            # Model aliases mapping
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alias TEXT UNIQUE NOT NULL,
                    provider_ids TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Requests log table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id INTEGER,
                    key_id INTEGER,
                    model_used TEXT,
                    status_code INTEGER,
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    latency_ms REAL,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE SET NULL,
                    FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE SET NULL
                )
            """)

            # Health checks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS health_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id INTEGER NOT NULL,
                    is_healthy BOOLEAN NOT NULL,
                    response_time_ms REAL,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
                )
            """)

            # Key rotation log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS key_rotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id INTEGER NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
                )
            """)

            # System config
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Circuit breaker states
            conn.execute("""
                CREATE TABLE IF NOT EXISTS circuit_breakers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider_id INTEGER UNIQUE NOT NULL,
                    state TEXT DEFAULT 'closed',
                    failure_count INTEGER DEFAULT 0,
                    last_failure_time TIMESTAMP,
                    last_success_time TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE
                )
            """)

            # Logs table for real-time log streaming
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    logger_name TEXT,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_provider ON api_keys(provider_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_provider ON requests(provider_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_health_checks_provider ON health_checks(provider_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)")

            conn.commit()
            logger.info("Database initialized successfully")

    def encrypt_key(self, api_key: str) -> bytes:
        """Encrypt an API key."""
        return self._cipher.encrypt(api_key.encode())

    def decrypt_key(self, encrypted_key: bytes) -> str:
        """Decrypt an API key."""
        return self._cipher.decrypt(encrypted_key).decode()

    # Provider operations
    def create_provider(
        self,
        name: str,
        provider_type: str,
        base_url: str,
        strategy: str = "sequential",
        timeout: float = 120.0,
        max_retries: int = 3,
        config: Dict = None,
    ) -> int:
        """Create a new provider. Returns the provider ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO providers (name, type, base_url, strategy, timeout, max_retries, config_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    provider_type,
                    base_url,
                    strategy,
                    timeout,
                    max_retries,
                    json.dumps(config) if config else None,
                ),
            )
            conn.commit()
            logger.info(f"Created provider: {name} (ID: {cursor.lastrowid})")
            return cursor.lastrowid

    def get_provider(self, provider_id: int) -> Optional[Dict]:
        """Get provider by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM providers WHERE id = ? AND is_active = 1", (provider_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_provider_by_name(self, name: str) -> Optional[Dict]:
        """Get provider by name."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM providers WHERE name = ? AND is_active = 1", (name,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_providers(self) -> List[Dict]:
        """Get all active providers."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM providers WHERE is_active = 1 ORDER BY name"
            ).fetchall()
            return [dict(row) for row in rows]

    def update_provider(self, provider_id: int, **kwargs) -> bool:
        """Update provider fields."""
        allowed_fields = [
            "name",
            "type",
            "base_url",
            "strategy",
            "timeout",
            "max_retries",
            "config_json",
            "is_active",
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return False

        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [provider_id]

        with self._get_connection() as conn:
            conn.execute(f"UPDATE providers SET {set_clause} WHERE id = ?", values)
            conn.commit()
            return True

    def delete_provider(self, provider_id: int) -> bool:
        """Soft delete a provider."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE providers SET is_active = 0 WHERE id = ?", (provider_id,)
            )
            conn.commit()
            return True

    # API Key operations
    def create_api_key(
        self, provider_id: int, api_key: str, models: List[str] = None
    ) -> int:
        """Create a new API key for a provider. Returns key ID."""
        encrypted = self.encrypt_key(api_key)
        preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO api_keys (provider_id, key_encrypted, key_preview, models_allowed)
                   VALUES (?, ?, ?, ?)""",
                (
                    provider_id,
                    encrypted,
                    preview,
                    json.dumps(models) if models else None,
                ),
            )
            conn.commit()
            logger.info(
                f"Created API key for provider {provider_id} (ID: {cursor.lastrowid})"
            )
            return cursor.lastrowid

    def get_api_key(
        self, key_id: int, include_decrypted: bool = False
    ) -> Optional[Dict]:
        """Get API key by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()

            if not row:
                return None

            result = dict(row)
            if include_decrypted:
                result["key_decrypted"] = self.decrypt_key(row["key_encrypted"])

            if result["models_allowed"]:
                result["models_allowed"] = json.loads(result["models_allowed"])

            return result

    def get_api_keys_by_provider(self, provider_id: int) -> List[Dict]:
        """Get all API keys for a provider."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE provider_id = ? ORDER BY id",
                (provider_id,),
            ).fetchall()

            results = []
            for row in rows:
                result = dict(row)
                if result["models_allowed"]:
                    result["models_allowed"] = json.loads(result["models_allowed"])
                results.append(result)

            return results

    def get_all_api_keys(self) -> List[Dict]:
        """Get all API keys."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT k.*, p.name as provider_name 
                   FROM api_keys k
                   JOIN providers p ON k.provider_id = p.id
                   ORDER BY k.id"""
            ).fetchall()

            results = []
            for row in rows:
                result = dict(row)
                if result["models_allowed"]:
                    result["models_allowed"] = json.loads(result["models_allowed"])
                results.append(result)

            return results

    def update_api_key(self, key_id: int, **kwargs) -> bool:
        """Update API key fields."""
        allowed_fields = [
            "key_encrypted",
            "key_preview",
            "models_allowed",
            "status",
            "error_count",
            "session_start",
            "cooldown_until",
            "total_requests",
            "last_used",
        ]
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not updates:
            return False

        updates["updated_at"] = datetime.now().isoformat()

        # Handle special fields
        if "models_allowed" in updates and isinstance(updates["models_allowed"], list):
            updates["models_allowed"] = json.dumps(updates["models_allowed"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [key_id]

        with self._get_connection() as conn:
            conn.execute(f"UPDATE api_keys SET {set_clause} WHERE id = ?", values)
            conn.commit()
            return True

    def delete_api_key(self, key_id: int) -> bool:
        """Delete an API key."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            conn.commit()
            return True

    def rotate_api_key(self, key_id: int, new_status: str, reason: str = None):
        """Log key rotation and update status."""
        key = self.get_api_key(key_id)
        if not key:
            return

        old_status = key["status"]

        with self._get_connection() as conn:
            # Log rotation
            conn.execute(
                """INSERT INTO key_rotations (key_id, from_status, to_status, reason)
                   VALUES (?, ?, ?, ?)""",
                (key_id, old_status, new_status, reason),
            )

            # Update key status
            self.update_api_key(key_id, status=new_status)
            conn.commit()

    # Model operations
    def create_model(
        self,
        alias: str,
        technical_name: str,
        provider_id: int = None,
        description: str = None,
    ) -> int:
        """Create a model alias."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO models (provider_id, alias, technical_name, description)
                   VALUES (?, ?, ?, ?)""",
                (provider_id, alias, technical_name, description),
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_models(self) -> List[Dict]:
        """Get all models."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT m.*, p.name as provider_name 
                   FROM models m
                   LEFT JOIN providers p ON m.provider_id = p.id
                   WHERE m.is_active = 1
                   ORDER BY m.alias"""
            ).fetchall()
            return [dict(row) for row in rows]

    def create_model_alias_mapping(self, alias: str, provider_ids: List[int]) -> int:
        """Create/update model alias to providers mapping."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT OR REPLACE INTO model_aliases (alias, provider_ids)
                   VALUES (?, ?)""",
                (alias, json.dumps(provider_ids)),
            )
            conn.commit()
            return cursor.lastrowid

    def get_model_alias_mapping(self, alias: str) -> Optional[List[int]]:
        """Get providers for a model alias."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT provider_ids FROM model_aliases WHERE alias = ?", (alias,)
            ).fetchone()

            if row:
                return json.loads(row["provider_ids"])
            return None

    # Request logging
    def log_request(
        self,
        provider_id: int = None,
        key_id: int = None,
        model_used: str = None,
        status_code: int = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        latency_ms: float = None,
        error_message: str = None,
    ):
        """Log a request."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO requests 
                   (provider_id, key_id, model_used, status_code, tokens_input, 
                    tokens_output, latency_ms, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    provider_id,
                    key_id,
                    model_used,
                    status_code,
                    tokens_input,
                    tokens_output,
                    latency_ms,
                    error_message,
                ),
            )
            conn.commit()

    def get_requests_stats(self, hours: int = 24) -> Dict:
        """Get request statistics."""
        with self._get_connection() as conn:
            # Total requests
            total = conn.execute(
                """SELECT COUNT(*) as count FROM requests 
                   WHERE timestamp > datetime('now', '-{} hours')""".format(hours)
            ).fetchone()["count"]

            # Success vs failure
            success = conn.execute(
                """SELECT COUNT(*) as count FROM requests 
                   WHERE status_code = 200 
                   AND timestamp > datetime('now', '-{} hours')""".format(hours)
            ).fetchone()["count"]

            # Average latency
            latency = conn.execute(
                """SELECT AVG(latency_ms) as avg FROM requests 
                   WHERE timestamp > datetime('now', '-{} hours')""".format(hours)
            ).fetchone()["avg"]

            # Total tokens
            tokens = (
                conn.execute(
                    """SELECT SUM(tokens_input + tokens_output) as total FROM requests
                   WHERE timestamp > datetime('now', '-{} hours')""".format(hours)
                ).fetchone()["total"]
                or 0
            )

            return {
                "total_requests": total,
                "successful_requests": success,
                "failed_requests": total - success,
                "average_latency_ms": round(latency, 2) if latency else 0,
                "total_tokens": tokens,
            }

    def get_recent_requests(self, limit: int = 100) -> List[Dict]:
        """Get recent requests."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT r.*, p.name as provider_name, k.key_preview
                   FROM requests r
                   LEFT JOIN providers p ON r.provider_id = p.id
                   LEFT JOIN api_keys k ON r.key_id = k.id
                   ORDER BY r.timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    # Health check operations
    def log_health_check(
        self,
        provider_id: int,
        is_healthy: bool,
        response_time_ms: float = None,
        error_message: str = None,
    ):
        """Log a health check."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO health_checks 
                   (provider_id, is_healthy, response_time_ms, error_message)
                   VALUES (?, ?, ?, ?)""",
                (provider_id, is_healthy, response_time_ms, error_message),
            )

            # Update circuit breaker
            state = "closed" if is_healthy else "open"
            conn.execute(
                """INSERT OR REPLACE INTO circuit_breakers 
                   (provider_id, state, last_success_time, last_failure_time)
                   VALUES (?, ?, 
                   CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END,
                   CASE WHEN ? THEN NULL ELSE CURRENT_TIMESTAMP END)""",
                (provider_id, state, is_healthy, is_healthy),
            )
            conn.commit()

    def get_health_status(self) -> List[Dict]:
        """Get current health status for all providers."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """SELECT p.*, c.state as circuit_state, c.failure_count,
                   (SELECT COUNT(*) FROM api_keys k WHERE k.provider_id = p.id AND k.status = 'available') as available_keys,
                   (SELECT COUNT(*) FROM api_keys k WHERE k.provider_id = p.id) as total_keys
                   FROM providers p
                   LEFT JOIN circuit_breakers c ON p.id = c.provider_id
                   WHERE p.is_active = 1"""
            ).fetchall()
            return [dict(row) for row in rows]

    # Log operations
    def add_log(self, level: str, logger_name: str, message: str):
        """Add a log entry."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO logs (level, logger_name, message) VALUES (?, ?, ?)",
                (level, logger_name, message),
            )
            conn.commit()

    def get_recent_logs(self, level: str = None, limit: int = 100) -> List[Dict]:
        """Get recent logs."""
        with self._get_connection() as conn:
            if level:
                rows = conn.execute(
                    """SELECT * FROM logs 
                       WHERE level = ? 
                       ORDER BY timestamp DESC 
                       LIMIT ?""",
                    (level, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM logs 
                       ORDER BY timestamp DESC 
                       LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]

    def clear_old_logs(self, days: int = 7):
        """Clear logs older than specified days."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM logs WHERE timestamp < datetime('now', '-{} days')".format(
                    days
                )
            )
            conn.commit()

    # System config
    def set_config(self, key: str, value: str):
        """Set a system configuration value."""
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO system_config (key, value, updated_at)
                   VALUES (?, ?, CURRENT_TIMESTAMP)""",
                (key, value),
            )
            conn.commit()

    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """Get a system configuration value."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default


# Global database instance
_db_instance: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    """Get or create the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
