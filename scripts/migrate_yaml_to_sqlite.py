#!/usr/bin/env python3
"""Migrate configuration from YAML to SQLite database."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import logging
from gateway.database import DatabaseManager, get_db
from gateway.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_config(yaml_path: str = None, db_path: str = None):
    """Migrate YAML configuration to SQLite database."""

    # Load YAML config
    config_loader = ConfigLoader(yaml_path)
    config_loader.load()

    # Initialize database
    db = DatabaseManager(db_path) if db_path else get_db()

    logger.info("Starting migration from YAML to SQLite...")

    # Migrate providers
    providers_config = config_loader.get_providers_config()
    for provider_name, provider_data in providers_config.items():
        logger.info(f"Migrating provider: {provider_name}")

        # Create provider
        provider_id = db.create_provider(
            name=provider_name,
            provider_type=provider_data.get("type", "ollama"),
            base_url=provider_data.get("base_url", "https://ollama.com/v1"),
            strategy=provider_data.get("strategy", "sequential"),
            timeout=provider_data.get("timeout", 120.0),
            max_retries=provider_data.get("max_retries", 3),
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

        # Migrate API keys
        keys = provider_data.get("keys", [])
        for key_data in keys:
            if isinstance(key_data, dict):
                api_key = key_data.get("key", "")
                models = key_data.get("models", [])
            else:
                api_key = key_data
                models = []

            if api_key:
                db.create_api_key(
                    provider_id=provider_id, api_key=api_key, models=models
                )
                logger.info(f"  Migrated API key for provider {provider_name}")

        # Migrate models
        models_mapping = provider_data.get("models", {})
        for alias, technical_name in models_mapping.items():
            db.create_model(
                alias=alias, technical_name=technical_name, provider_id=provider_id
            )
            logger.info(f"  Migrated model: {alias} -> {technical_name}")

    # Migrate model aliases
    models_config = config_loader.get_models_config()
    for alias, alias_data in models_config.items():
        provider_names = alias_data.get("providers", [])
        provider_ids = []

        for provider_name in provider_names:
            provider = db.get_provider_by_name(provider_name)
            if provider:
                provider_ids.append(provider["id"])

        if provider_ids:
            db.create_model_alias_mapping(alias, provider_ids)
            logger.info(f"Migrated model alias: {alias} -> {provider_names}")

    # Migrate system config
    gateway_config = config_loader.config.get("gateway", {})
    if gateway_config:
        db.set_config("gateway_host", str(gateway_config.get("host", "0.0.0.0")))
        db.set_config("gateway_port", str(gateway_config.get("port", 8000)))
        db.set_config("gateway_log_level", gateway_config.get("log_level", "INFO"))
        db.set_config("gateway_max_workers", str(gateway_config.get("max_workers", 10)))
        logger.info("Migrated gateway settings")

    circuit_breaker_config = config_loader.config.get("circuit_breaker", {})
    if circuit_breaker_config:
        db.set_config(
            "circuit_failure_threshold",
            str(circuit_breaker_config.get("failure_threshold", 3)),
        )
        db.set_config(
            "circuit_recovery_timeout",
            str(circuit_breaker_config.get("recovery_timeout", 30.0)),
        )
        logger.info("Migrated circuit breaker settings")

    logger.info("Migration completed successfully!")
    logger.info(f"Database location: {db.db_path}")

    # Show summary
    providers = db.get_all_providers()
    total_keys = len(db.get_all_api_keys())
    total_models = len(db.get_all_models())

    logger.info("\nMigration Summary:")
    logger.info(f"  Providers: {len(providers)}")
    logger.info(f"  API Keys: {total_keys}")
    logger.info(f"  Models: {total_models}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate YAML config to SQLite")
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML config file",
        default=os.environ.get("GATEWAY_CONFIG", "config/providers.yml"),
    )
    parser.add_argument(
        "--db",
        "-d",
        help="Path to SQLite database",
        default=os.environ.get("GATEWAY_DB_PATH", "data/gateway.db"),
    )

    args = parser.parse_args()

    migrate_config(args.config, args.db)
