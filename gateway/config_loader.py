"""Configuration loader for LLM Gateway using SQLite database."""

import os
import yaml
from typing import Dict, Any, Optional, List
import logging
from .database import get_db, DatabaseManager

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and validates gateway configuration from SQLite database.

    Falls back to YAML for initial migration if database is empty.
    """

    DEFAULT_CONFIG_PATH = "config/providers.yml"
    DEFAULT_DB_PATH = "data/gateway.db"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.db = get_db()
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from SQLite database.
        Falls back to YAML if database is empty.

        Returns:
            Parsed configuration dictionary
        """
        # Check if database has providers
        providers = self.db.get_all_providers()

        if not providers and os.path.exists(self.config_path):
            # Database is empty, try to migrate from YAML
            logger.info("Database empty, attempting migration from YAML...")
            try:
                self._migrate_from_yaml()
                providers = self.db.get_all_providers()
            except Exception as e:
                logger.error(f"Failed to migrate from YAML: {e}")

        if not providers:
            # Still no providers, create empty config
            logger.warning("No providers configured in database")
            self._config = {
                "providers": {},
                "models": {},
                "gateway": self._build_gateway_config(),
                "circuit_breaker": self._build_circuit_breaker_config(),
            }
            return self._config

        # Build config from database
        self._config = {
            "providers": self._build_providers_config(),
            "models": self._build_models_config(),
            "gateway": self._build_gateway_config(),
            "circuit_breaker": self._build_circuit_breaker_config(),
        }

        logger.info("Loaded configuration from SQLite database")
        return self._config

    def _migrate_from_yaml(self):
        """Migrate configuration from YAML to database."""
        if not os.path.exists(self.config_path):
            return

        with open(self.config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Migrate providers
        for provider_name, provider_data in yaml_config.get("providers", {}).items():
            provider_id = self.db.create_provider(
                name=provider_name,
                provider_type=provider_data.get("type", "ollama"),
                base_url=provider_data.get("base_url", "https://ollama.com/v1"),
                strategy=provider_data.get("strategy", "sequential"),
                timeout=float(provider_data.get("timeout", 120.0)),
                max_retries=int(provider_data.get("max_retries", 3)),
                config={
                    k: v
                    for k, v in provider_data.items()
                    if k
                    not in [
                        "keys",
                        "models",
                        "type",
                        "base_url",
                        "strategy",
                        "timeout",
                        "max_retries",
                    ]
                },
            )

            # Migrate keys
            for key_data in provider_data.get("keys", []):
                if isinstance(key_data, dict):
                    api_key = key_data.get("key", "")
                    models = key_data.get("models", [])
                else:
                    api_key = key_data
                    models = []

                if api_key:
                    self.db.create_api_key(provider_id, api_key, models)

            # Migrate models mapping
            for alias, technical_name in provider_data.get("models", {}).items():
                self.db.create_model(alias, technical_name, provider_id)

        # Migrate model aliases
        for alias, alias_data in yaml_config.get("models", {}).items():
            provider_names = alias_data.get("providers", [])
            provider_ids = []
            for name in provider_names:
                provider = self.db.get_provider_by_name(name)
                if provider:
                    provider_ids.append(provider["id"])
            if provider_ids:
                self.db.create_model_alias_mapping(alias, provider_ids)

        # Migrate system settings
        gateway = yaml_config.get("gateway", {})
        self.db.set_config("gateway_host", str(gateway.get("host", "0.0.0.0")))
        self.db.set_config("gateway_port", str(gateway.get("port", "8000")))
        self.db.set_config("gateway_log_level", gateway.get("log_level", "INFO"))
        self.db.set_config("gateway_max_workers", str(gateway.get("max_workers", "10")))

        circuit = yaml_config.get("circuit_breaker", {})
        self.db.set_config(
            "circuit_failure_threshold", str(circuit.get("failure_threshold", "3"))
        )
        self.db.set_config(
            "circuit_recovery_timeout", str(circuit.get("recovery_timeout", "30.0"))
        )

        logger.info("Migration from YAML completed successfully")

    def _build_providers_config(self) -> Dict[str, Any]:
        """Build providers config from database."""
        providers_config = {}

        for provider in self.db.get_all_providers():
            provider_name = provider["name"]

            # Get API keys for this provider
            keys_config = []
            for key in self.db.get_api_keys_by_provider(provider["id"]):
                key_data = {"key": self.db.decrypt_key(key["key_encrypted"])}
                if key.get("models_allowed"):
                    key_data["models"] = key["models_allowed"]
                keys_config.append(key_data)

            # Get models mapping
            models_mapping = {}
            for model in self.db.get_all_models():
                if model.get("provider_id") == provider["id"]:
                    models_mapping[model["alias"]] = model["technical_name"]

            providers_config[provider_name] = {
                "type": provider["type"],
                "base_url": provider["base_url"],
                "strategy": provider["strategy"],
                "timeout": provider["timeout"],
                "max_retries": provider["max_retries"],
                "keys": keys_config,
                "models": models_mapping,
            }

            # Add extra config from JSON
            if provider.get("config_json"):
                try:
                    import json

                    extra_config = json.loads(provider["config_json"])
                    providers_config[provider_name].update(extra_config)
                except:
                    pass

        return providers_config

    def _build_models_config(self) -> Dict[str, Any]:
        """Build models config from database."""
        models_config = {}

        # Get all unique aliases
        for model in self.db.get_all_models():
            alias = model["alias"]
            if alias not in models_config:
                # Get providers for this alias
                provider_ids = self.db.get_model_alias_mapping(alias) or []
                provider_names = []
                for pid in provider_ids:
                    provider = self.db.get_provider(pid)
                    if provider:
                        provider_names.append(provider["name"])

                models_config[alias] = {"providers": provider_names}

        return models_config

    def _build_gateway_config(self) -> Dict[str, Any]:
        """Build gateway config from database."""
        return {
            "host": self.db.get_config("gateway_host", "0.0.0.0"),
            "port": int(self.db.get_config("gateway_port", "8000")),
            "log_level": self.db.get_config("gateway_log_level", "INFO"),
            "max_workers": int(self.db.get_config("gateway_max_workers", "10")),
        }

    def _build_circuit_breaker_config(self) -> Dict[str, Any]:
        """Build circuit breaker config from database."""
        return {
            "failure_threshold": int(
                self.db.get_config("circuit_failure_threshold", "3")
            ),
            "recovery_timeout": float(
                self.db.get_config("circuit_recovery_timeout", "30.0")
            ),
        }

    def get_providers_config(self) -> Dict[str, Any]:
        """Get provider configurations."""
        if self._config is None:
            self.load()
        return self._config.get("providers", {})

    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider."""
        providers = self.get_providers_config()
        return providers.get(provider_name)

    def get_models_config(self) -> Dict[str, Any]:
        """Get model alias configurations."""
        if self._config is None:
            self.load()
        return self._config.get("models", {})

    def get_model_config(self, model_alias: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model alias."""
        models = self.get_models_config()
        return models.get(model_alias)

    def get_gateway_config(self) -> Dict[str, Any]:
        """Get gateway server configuration."""
        if self._config is None:
            self.load()
        return self._config.get(
            "gateway",
            {
                "host": "0.0.0.0",
                "port": 8000,
                "log_level": "INFO",
                "max_workers": 10,
            },
        )

    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration."""
        if self._config is None:
            self.load()
        return self._config.get(
            "circuit_breaker",
            {
                "failure_threshold": 3,
                "recovery_timeout": 30.0,
            },
        )

    def validate(self) -> bool:
        """Validate the loaded configuration."""
        if self._config is None:
            self.load()

        providers = self.get_providers_config()
        if not providers:
            raise ValueError("No providers configured")

        for name, config in providers.items():
            keys = config.get("keys", [])
            if not keys:
                logger.warning(f"Provider {name} has no keys configured")

            strategy = config.get("strategy", "sequential")
            if strategy not in ["sequential", "round_robin", "random", "fallback"]:
                raise ValueError(f"Invalid strategy '{strategy}' for provider {name}")

        logger.info("Configuration validated successfully")
        return True

    def reload(self) -> Dict[str, Any]:
        """Reload configuration from database."""
        self._config = None
        return self.load()


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Convenience function to create and load a ConfigLoader."""
    loader = ConfigLoader(config_path)
    loader.load()
    return loader
