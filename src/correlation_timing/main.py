"""Main script to run correlation timing analysis.

This script orchestrates the entire analysis pipeline:
1. Load and prepare data
2. Analyze Hypothesis 1 (time differences to all correlation factors)
3. Analyze Hypothesis 2 (time differences for same correlation types)
"""
import os
from pathlib import Path
import argparse

from dotenv import load_dotenv
from loguru import logger

from iconfig.iconfig import iConfig

from correlation_timing.analysis import run_analysis

def parse_args(config: iConfig) -> argparse.Namespace: 

    parser = argparse.ArgumentParser(
        description='Run correlation timing analysis'
    )
    parser.add_argument(
        'srcdir',
        type=Path,
        help='Directory containing JSON data files'
    )
    parser.add_argument(
        '--max-lookback',
        type=int,
        default = config('max_lookback_length', default=4),
        help='Maximum lookback window in hours'
    )
    parser.add_argument(
        '--no-category',
        action='store_true',
        default=not config('process_by_category', default=True),
        help='Do not process by category'
    )
    parser.add_argument(
        '--skip-h1',
        action='store_true',
        default= not config('run_hypothesis_1', default=True),
        help='Skip Hypothesis 1 analysis'
    )
    parser.add_argument(
        '--skip-h2',
        action='store_true',
        default=not config('run_hypothesis_2', default=True),
        help='Skip Hypothesis 2 analysis'
    )
    parser.add_argument(
        '--no-fit',
        action='store_true',
        default=not config('fit_distributions', default=True),
        help='Do not fit distributions'
    )
    parser.add_argument(
        '--cumulative',
        action='store_true',
        default=config('cumulative', default=False),
        help='Show cumulative plots'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Example usage

    load_dotenv()

    logdir = Path(os.getenv('LOG_DIR', '.'))
    logger.add(logdir / "run.log", rotation="10 MB")
    
    config = iConfig()

    args = parse_args(config)

    run_analysis(
        config=config,
        srcdir=args.srcdir,
        max_lookback_length=args.max_lookback,
        process_by_category=not args.no_category,
        run_hypothesis_1=not args.skip_h1,
        run_hypothesis_2=not args.skip_h2,
        fit_distributions=not args.no_fit,
        cumulative=args.cumulative,
    )
