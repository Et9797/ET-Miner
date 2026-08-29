#!/usr/bin/env python3
"""Command-line interface for et-miner.

Usage:
    et-miner mine --input data.parquet --min-support 0.01
    et-miner --help

Or via Python module:
    python -m et_miner --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import polars as pl

from et_miner import (
    __version__,
    apriori,
    apriori_streaming,
    generate_rules,
    Config,
    load_config,
    configure_logging,
    logger,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="et-miner",
        description="ET-Miner: Fast, Memory-Efficient & GPU-Accelerated Frequent Itemset Mining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mine rules from transactions
  et-miner mine --input transactions.parquet --min-support 0.01

  # Mine with streaming for large datasets
  et-miner mine --input large.parquet --streaming --chunk-size 100000

  # Show installation info
  et-miner info
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"et-miner {__version__}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === MINE command ===
    mine_parser = subparsers.add_parser(
        "mine",
        help="Mine association rules from transaction data",
    )
    mine_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file (parquet, csv) with transaction data",
    )
    mine_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file for rules (JSON). Defaults to stdout.",
    )
    mine_parser.add_argument(
        "--item-col",
        default="items",
        help="Column name containing item lists (default: items)",
    )
    mine_parser.add_argument(
        "--min-support",
        type=float,
        default=None,
        help="Minimum support threshold (default: from config)",
    )
    mine_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for rule generation (default: 0.5)",
    )
    mine_parser.add_argument(
        "--min-lift",
        type=float,
        default=1.0,
        help="Minimum lift for rule filtering (default: 1.0)",
    )
    mine_parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum itemset length",
    )
    mine_parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets",
    )
    mine_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for streaming mode",
    )
    mine_parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format (default: json)",
    )

    # === INFO command ===
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about the installation",
    )

    return parser


def cmd_mine(args: argparse.Namespace, config: Config) -> int:
    """Execute the mine command."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Load data
    logger.info(f"Loading data from {input_path}")
    if input_path.suffix == ".parquet":
        df = pl.scan_parquet(input_path).collect(engine="streaming")
    elif input_path.suffix == ".csv":
        df = pl.read_csv(input_path)
    else:
        logger.error(f"Unsupported file format: {input_path.suffix}")
        return 1

    if args.item_col not in df.columns:
        logger.error(f"Column '{args.item_col}' not found in data")
        return 1

    logger.info(f"Loaded {len(df)} transactions")

    # Get parameters
    min_support = args.min_support or config.apriori.min_support
    max_length = args.max_length or config.apriori.max_length

    # Mine itemsets
    logger.info(f"Mining itemsets (min_support={min_support})")
    start = time.perf_counter()

    if args.streaming:
        chunk_size = args.chunk_size or config.streaming.chunk_size
        logger.info(f"Using streaming mode (chunk_size={chunk_size})")
        itemsets = apriori_streaming(
            df.lazy(),
            min_support=min_support,
            max_length=max_length,
            chunk_size=chunk_size,
            item_col=args.item_col,
        )
    else:
        itemsets = apriori(
            df,
            min_support=min_support,
            max_length=max_length,
            item_col=args.item_col,
        )

    mining_time = time.perf_counter() - start
    logger.success(f"Found {len(itemsets)} itemsets in {mining_time:.2f}s")

    # Generate rules
    logger.info(f"Generating rules (min_confidence={args.min_confidence})")
    rules = generate_rules(itemsets, min_confidence=args.min_confidence)

    # Filter by lift
    if args.min_lift > 1.0:
        rules = [r for r in rules if r.lift >= args.min_lift]

    logger.success(f"Generated {len(rules)} rules")

    # Convert to serializable format
    rules_data = [
        {
            "antecedent": list(r.lhs),
            "consequent": list(r.rhs),
            "support": r.support,
            "confidence": r.confidence,
            "lift": r.lift,
        }
        for r in rules
    ]

    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            if args.format == "jsonl":
                for rule in rules_data:
                    f.write(json.dumps(rule) + "\n")
            else:
                json.dump(rules_data, f, indent=2)
        logger.info(f"Rules saved to {output_path}")
    else:
        # Print to stdout
        if args.format == "jsonl":
            for rule in rules_data:
                print(json.dumps(rule))
        else:
            print(json.dumps(rules_data, indent=2))

    return 0



def cmd_info(args: argparse.Namespace, config: Config) -> int:
    """Execute the info command."""
    from et_miner._compat import HAS_TQDM
    from et_miner.backends import (
        CUPY_INSTALLED,
        get_cupy_version,
        get_gpu_count,
        get_rust_version,
        has_rust_extension,
    )

    # Print to stdout so `info` stays visible regardless of the --quiet log level.
    print(f"et-miner {__version__}")
    print("Dependencies:")
    print(f"  polars: {pl.__version__}")
    print(f"  tqdm: {'installed' if HAS_TQDM else 'not installed'}")
    print("Backends:")
    print(f"  rust: {get_rust_version() if has_rust_extension() else 'not built'}")
    print(f"  cupy: {get_cupy_version() if CUPY_INSTALLED else 'not installed'} ({get_gpu_count()} GPUs)")
    print("Configuration:")
    print(f"  min_support: {config.apriori.min_support}")
    print(f"  chunk_size: {config.streaming.chunk_size}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Load configuration
    config = load_config(path=args.config)

    # Configure logging
    if args.quiet:
        configure_logging(level="ERROR")
    elif args.verbose:
        configure_logging(level="DEBUG")
    else:
        configure_logging(level=config.logging.level)

    # No command specified
    if args.command is None:
        parser.print_help()
        return 0

    # Execute command
    commands = {
        "mine": cmd_mine,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler is None:
        logger.error(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args, config)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
