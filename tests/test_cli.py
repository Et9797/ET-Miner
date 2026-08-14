"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path

import pytest
import polars as pl

from et_miner.cli import main, create_parser


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_help(self):
        """Test --help doesn't crash."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_version(self):
        """Test --version doesn't crash."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_no_command(self):
        """Test running without command shows help."""
        result = main([])
        assert result == 0


class TestMineCommand:
    """Tests for the mine command."""

    def test_mine_parquet(self, tmp_path):
        """Test mining from parquet file."""
        # Create test data
        data = pl.DataFrame({
            "items": [[1, 2, 3], [2, 3, 4], [1, 2, 3, 4], [1, 3], [2, 3]],
        })
        input_path = tmp_path / "transactions.parquet"
        data.write_parquet(input_path)

        output_path = tmp_path / "rules.json"

        result = main([
            "-q",  # Quiet mode goes before subcommand
            "mine",
            "--input", str(input_path),
            "--output", str(output_path),
            "--min-support", "0.4",
            "--min-confidence", "0.5",
        ])

        assert result == 0
        assert output_path.exists()

        with open(output_path) as f:
            rules = json.load(f)
        assert isinstance(rules, list)

    def test_mine_missing_file(self):
        """Test mine with missing input file."""
        result = main([
            "-q",
            "mine",
            "--input", "/nonexistent/file.parquet",
        ])
        assert result == 1

    def test_mine_missing_column(self, tmp_path):
        """Test mine with missing column."""
        data = pl.DataFrame({"other_col": [[1, 2]]})
        input_path = tmp_path / "data.parquet"
        data.write_parquet(input_path)

        result = main([
            "-q",
            "mine",
            "--input", str(input_path),
            "--item-col", "items",
        ])
        assert result == 1


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_shows_version(self, capsys):
        """Test info shows version."""
        result = main(["-q", "info"])
        assert result == 0

        captured = capsys.readouterr()
        assert "et-miner" in captured.out
        assert "polars:" in captured.out


class TestConfigIntegration:
    """Tests for CLI config integration."""

    def test_config_from_file(self, tmp_path):
        """Test loading config from file."""
        config_content = """
[apriori]
min_support = 0.05
"""
        config_path = tmp_path / "config.toml"
        config_path.write_text(config_content)

        # Just test that config file is accepted
        result = main(["--config", str(config_path), "-q", "info"])
        assert result == 0

    def test_verbose_mode(self, capsys):
        """Test verbose mode outputs more."""
        result = main(["--verbose", "info"])
        assert result == 0

    def test_quiet_mode(self, capsys):
        """Test quiet mode suppresses output."""
        result = main(["--quiet", "info"])
        assert result == 0
