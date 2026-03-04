"""Configuration loader for LLM Gateway."""

import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and validates gateway configuration from YAML file.

    Configuration includes:
    - Provider settings (API keys, base URLs, strategies)
    - Model aliases
    - Gateway settings (host, port, logging)
    """

    DEFAULT_CONFIG_PATH = "config/providers.yml"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Parsed configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

        if not self._config:
            self._config = {}

        logger.info(f"Loaded configuration from {self.config_path}")
        return self._config

    def get_providers_config(self) -> Dict[str, Any]:
        """
        Get provider configurations.

        Returns:
            Dictionary of provider configurations
        """
        if self._config is None:
            self.load()
        return self._config.get("providers", {})

    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider configuration or None if not found
        """
        providers = self.get_providers_config()
        return providers.get(provider_name)

    def get_models_config(self) -> Dict[str, Any]:
        """
        Get model alias configurations.

        Returns:
            Dictionary of model aliases to provider mappings
        """
        if self._config is None:
            self.load()
        return self._config.get("models", {})

    def get_model_config(self, model_alias: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model alias.

        Args:
            model_alias: Model alias (e.g., "smart", "fast")

        Returns:
            Model configuration or None if not found
        """
        models = self.get_models_config()
        return models.get(model_alias)

    def get_gateway_config(self) -> Dict[str, Any]:
        """
        Get gateway server configuration.

        Returns:
            Gateway configuration dictionary
        """
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
        """
        Get circuit breaker configuration.

        Returns:
            Circuit breaker configuration dictionary
        """
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
        """
        Validate the loaded configuration.

        Returns:
            True if valid, raises ValueError otherwise

        Raises:
            ValueError: If configuration is invalid
        """
        if self._config is None:
            self.load()

        providers = self.get_providers_config()
        if not providers:
            raise ValueError("No providers configured")

        for name, config in providers.items():
            # Validate provider has keys
            keys = config.get("keys", [])
            if not keys:
                logger.warning(f"Provider {name} has no keys configured")

            # Validate strategy
            strategy = config.get("strategy", "sequential")
            if strategy not in ["sequential", "round_robin", "random", "fallback"]:
                raise ValueError(f"Invalid strategy '{strategy}' for provider {name}")

        logger.info("Configuration validated successfully")
        return True

    def reload(self) -> Dict[str, Any]:
        """
        Reload configuration from file.

        Returns:
            Updated configuration dictionary
        """
        self._config = None
        return self.load()


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Convenience function to create and load a ConfigLoader.

    Args:
        config_path: Optional path to configuration file

    Returns:
        ConfigLoader instance with loaded configuration
    """
    loader = ConfigLoader(config_path)
    loader.load()
    return loader
