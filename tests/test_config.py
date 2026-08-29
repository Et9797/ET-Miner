"""Tests for the configuration system."""

import os
import pytest
import tempfile

from et_miner.config import (
    Config,
    AprioriConfig,
    LoggingConfig,
    load_config,
    get_default_config,
)
from et_miner.exceptions import (
    InvalidConfigurationError,
    ConfigFileNotFoundError,
)


class TestAprioriConfig:
    """Tests for AprioriConfig validation."""

    def test_valid_min_support(self):
        """Test valid min_support values."""
        config = AprioriConfig(min_support=0.5)
        config.validate()  # Should not raise

        config = AprioriConfig(min_support=0.0)
        config.validate()

        config = AprioriConfig(min_support=1.0)
        config.validate()

    def test_invalid_min_support_negative(self):
        """Test that negative min_support raises error."""
        config = AprioriConfig(min_support=-0.1)
        with pytest.raises(InvalidConfigurationError) as exc_info:
            config.validate()
        assert "min_support" in str(exc_info.value)

    def test_invalid_min_support_too_high(self):
        """Test that min_support > 1 raises error."""
        config = AprioriConfig(min_support=1.5)
        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_invalid_max_length(self):
        """Test that max_length < 1 raises error."""
        config = AprioriConfig(max_length=0)
        with pytest.raises(InvalidConfigurationError):
            config.validate()


class TestConfig:
    """Tests for main Config class."""

    def test_default_values(self):
        """Test default config values."""
        config = Config()
        assert config.apriori.min_support == 0.01
        assert config.streaming.chunk_size == 100_000

    def test_validate_all(self):
        """Test that validate() checks all sections."""
        config = Config()
        config.apriori.min_support = -1  # Invalid

        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            "apriori": {"min_support": 0.05},
            "streaming": {"memory_budget_mb": 2048},
        }
        config = Config.from_dict(data)

        assert config.apriori.min_support == 0.05
        assert config.streaming.memory_budget_mb == 2048
        # Defaults preserved
        assert config.streaming.chunk_size == 100_000

    def test_to_dict(self):
        """Test converting config to dict."""
        config = Config()
        config.apriori.min_support = 0.02

        data = config.to_dict()

        assert data["apriori"]["min_support"] == 0.02
        assert "streaming" in data


class TestConfigFromFile:
    """Tests for loading config from TOML files."""

    def test_from_file_not_found(self):
        """Test that missing file raises ConfigFileNotFoundError."""
        with pytest.raises(ConfigFileNotFoundError):
            Config.from_file("/nonexistent/path/config.toml")

    def test_from_file_valid_toml(self):
        """Test loading valid TOML config."""
        toml_content = """
[apriori]
min_support = 0.03
max_length = 5

[streaming]
chunk_size = 50000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.apriori.min_support == 0.03
            assert config.apriori.max_length == 5
            assert config.streaming.chunk_size == 50000
        finally:
            os.unlink(temp_path)

    def test_from_file_partial_config(self):
        """Test that partial config uses defaults for missing values."""
        toml_content = """
[apriori]
min_support = 0.05
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = Config.from_file(temp_path)
            assert config.apriori.min_support == 0.05
            # Defaults
            assert config.apriori.sparse is True
            assert config.streaming.chunk_size == 100_000
        finally:
            os.unlink(temp_path)


class TestConfigFromEnv:
    """Tests for loading config from environment variables."""

    def test_env_override_float(self):
        """Test float env override."""
        try:
            os.environ["ET_MINER_APRIORI_MIN_SUPPORT"] = "0.07"
            config = Config.from_env()
            assert config.apriori.min_support == 0.07
        finally:
            del os.environ["ET_MINER_APRIORI_MIN_SUPPORT"]

    def test_env_override_int(self):
        """Test int env override."""
        try:
            os.environ["ET_MINER_STREAMING_CHUNK_SIZE"] = "50000"
            config = Config.from_env()
            assert config.streaming.chunk_size == 50000
        finally:
            del os.environ["ET_MINER_STREAMING_CHUNK_SIZE"]

    def test_env_override_bool(self):
        """Test bool env override."""
        try:
            os.environ["ET_MINER_CACHE_ENABLED"] = "false"
            config = Config.from_env()
            assert config.cache.enabled is False
        finally:
            del os.environ["ET_MINER_CACHE_ENABLED"]

    def test_env_override_string(self):
        """Test string env override."""
        try:
            os.environ["ET_MINER_LOGGING_LEVEL"] = "DEBUG"
            config = Config.from_env()
            assert config.logging.level == "DEBUG"
        finally:
            del os.environ["ET_MINER_LOGGING_LEVEL"]

    def test_env_invalid_value(self):
        """Test that invalid env value raises error."""
        try:
            os.environ["ET_MINER_APRIORI_MIN_SUPPORT"] = "not_a_number"
            with pytest.raises(InvalidConfigurationError):
                Config.from_env()
        finally:
            del os.environ["ET_MINER_APRIORI_MIN_SUPPORT"]


class TestLoadConfig:
    """Tests for the load_config convenience function."""

    def test_load_config_defaults(self):
        """Test loading config with no file."""
        config = load_config(path=None, use_env=False)
        assert config.apriori.min_support == 0.01

    def test_load_config_with_file(self):
        """Test loading config from explicit path."""
        toml_content = """
[apriori]
min_support = 0.08
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            config = load_config(path=temp_path)
            assert config.apriori.min_support == 0.08
        finally:
            os.unlink(temp_path)

    def test_load_config_env_overrides_file(self):
        """Test that env vars override file values."""
        toml_content = """
[apriori]
min_support = 0.02
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            temp_path = f.name

        try:
            os.environ["ET_MINER_APRIORI_MIN_SUPPORT"] = "0.09"
            config = load_config(path=temp_path, use_env=True)
            assert config.apriori.min_support == 0.09  # Env wins
        finally:
            os.unlink(temp_path)
            del os.environ["ET_MINER_APRIORI_MIN_SUPPORT"]


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_defaults(self):
        """Test that get_default_config returns default values."""
        config = get_default_config()
        assert config.apriori.min_support == 0.01
        assert config.apriori.sparse is True
        assert config.streaming.chunk_size == 100_000


class TestLoggingConfig:
    """Tests for LoggingConfig validation."""

    def test_valid_levels(self):
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LoggingConfig(level=level)
            config.validate()

    def test_invalid_level(self):
        """Test invalid log level."""
        config = LoggingConfig(level="INVALID")
        with pytest.raises(InvalidConfigurationError):
            config.validate()

    def test_valid_formats(self):
        """Test valid log formats."""
        for fmt in ["default", "compact", "json"]:
            config = LoggingConfig(format=fmt)
            config.validate()

    def test_invalid_format(self):
        """Test invalid log format."""
        config = LoggingConfig(format="invalid_format")
        with pytest.raises(InvalidConfigurationError):
            config.validate()
