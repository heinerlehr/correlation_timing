# Correlation Timing Analysis

An advanced statistical analysis tool for optimizing temporal detection windows in livestock farming operations. This project analyzes historical anomaly-correlation data to determine data-driven, optimal lookback windows for identifying correlations between water consumption anomalies and potential causative factors.

**📖 For detailed documentation, see [README_DETAILED.md](README_DETAILED.md)**

## Overview

### Problem Statement

In agricultural monitoring systems like Marvin, when water consumption anomalies occur, the system needs to identify potential causative factors. Currently, fixed time windows are used:
- **Poultry**: 1-hour lookback window
- **Pigs**: 2-hour lookback window

These fixed windows are not necessarily optimal for all correlation types. Different factors may have different temporal patterns—some correlate within minutes, others take hours to manifest.

### Objective

**Determine the optimal, data-driven time window to look back for each type of correlation when a water anomaly occurs.**

This analysis uses mixture distribution fitting (exponential, Weibull, log-normal) to model temporal patterns and identify when past events are most likely to be causally related to detected anomalies.

### Key Hypotheses

1. **Hypothesis 1**: Time difference between anomalies and ALL possible correlation factors
   - Analyzes how frequently each correlation type appears in the past
   - Identifies typical time ranges for different correlation types

2. **Hypothesis 2**: Time difference for specific correlating factors within the same anomaly
   - Focuses on recurrence patterns of specific correlations
   - Reveals how frequently specific factors repeat in the same shed

## Installation

### Prerequisites

- **Python**: 3.13 or higher
- **Package Manager**: [UV](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/heinerlehr/correlation_timing.git
cd correlation_timing

# Install with UV (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

## How to obtain a list of correlation timings

**NOTE**: Chickens and pigs should be run separately as the production systems are vastly different. The same goes very likely for broilers, breeders, and layers amongst
poultry production and sows, piglets/weaners, and finishers in pig production.

## Step 1: clean from old data
```bash
rm outputs/*
rm inputs/type*
```

## Step 2: run the ct tool
```bash
# Run analysis on a data directory
python -m ct --skip-h1 /path/to/data/directory
```
See below

This creates a number of graphics in the outputs folder

## Step 3: collect data from tool

Each graphic details the correlation factor, the category if requested, the number of correlations of this type and the fits. Visual inspection of the fits is relevant. Most of the time the major component should be selected but sometimes the fit is better to the minority component. The correct parameter to collect is the "scale".


## Usage

### Command Line

The primary way to run the analysis is via command line:

```bash
# Run analysis on a data directory
python -m ct /path/to/data/directory

# Show available options
python -m ct -h
```

**Required argument:**
- `srcdir` - Directory containing JSON anomaly/correlation data files

**Options:**
- `-h, --help` - Show help message and all available options
- `--max-lookback HOURS` - Maximum lookback window in hours (default: 4)
- `--no-category` - Do not process by category (default: process by category)
- `--skip-h1` - Skip Hypothesis 1 analysis (default: run H1)
- `--skip-h2` - Skip Hypothesis 2 analysis (default: run H2)
- `--no-fit` - Do not fit distributions (default: fit distributions)
- `--cumulative` - Show cumulative plots (default: off)

Configuration can also be customized via `config/config.yaml`. See [README_DETAILED.md](README_DETAILED.md#configuration) for details.

### Jupyter Notebooks

Two example notebooks are included:

1. **Farm-specific-correlations.ipynb** - Comprehensive correlation analysis
   - DBSCAN clustering
   - Exponential distribution fitting
   - Statistical significance testing
   - Production visualizations

```bash
jupyter notebook notebooks/Farm-specific-correlations.ipynb
```

## Core Features

### Data Processing
- ✅ Loads JSON anomaly-correlation data
- ✅ Filters by category and time window
- ✅ Automatic datetime parsing
- ✅ Vectorized delay calculations

### Analysis
- ✅ Hypothesis 1: Time to all correlation types
- ✅ Hypothesis 2: Recurrence of specific correlations
- ✅ Category-based segmentation
- ✅ Configurable lookback windows

### Statistical Modeling
- ✅ Mixture distribution fitting
- ✅ Three distribution types (Exponential, Weibull, Log-normal)
- ✅ AIC/BIC model selection
- ✅ Parallel processing (10 workers default)

### Visualization
- ✅ Frequency histograms
- ✅ Cumulative distribution plots
- ✅ Mixture component curves
- ✅ Publication-quality output

## Output

Analysis generates:
- **PNG figures**: Delay distribution plots per correlation
- **JSON results**: Fitted parameters and statistics
- **DataFrames**: Raw and aggregated results for further analysis

All outputs use configurable file paths via `config.yaml`.

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=src/ct tests/

# Specific test
pytest tests/test_utils.py::test_fit_mixture_simple_expon
```

Tests cover:
- Distribution fitting (exponential, Weibull, log-normal)
- NaN handling
- Parameter validation
- Mixture component fitting

---

**For more information**: See [README_DETAILED.md](README_DETAILED.md)  
**Python**: 3.13+ required  
**Status**: Active Development

