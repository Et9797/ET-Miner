"""Enable running et-miner as a module.

Usage:
    python -m et_miner --help
    python -m et_miner mine --input data.parquet --min-support 0.01
"""
import sys

from et_miner.cli import main

if __name__ == "__main__":
    sys.exit(main())
