# Correlation Timing Analysis

An app for analyzing optimal time windows for identifying correlations between water consumption anomalies and potential causative factors in livestock farming operations.

## Overview

This package analyzes historical anomaly-correlation data to determine optimal lookback windows for different types of correlations. Currently, Marvin (the monitoring system) uses fixed time windows (1h for poultry, 2h for pigs) when identifying factors correlated with water consumption anomalies. This analysis helps determine individual, data-driven lookback windows per correlation type.

## Objective

The primary goal is to answer: **What is the optimal time window to look back for each type of correlation when a water anomaly occurs?**

### Key Questions

1. **Hypothesis 1**: If we measure the time difference between an anomaly and all possible correlation factors, how frequently does another anomaly with that correlating factor occur in the past?

2. **Hypothesis 2**: If we measure the time difference for all correlating factors of an anomaly, how far back does the same correlating factor occur in other anomalies?

Both hypotheses use mixture distribution fitting (exponential, Weibull, log-normal) to model the temporal patterns of correlations.

## Installation

### Prerequisites

- Python 3.10 or higher
- UV package manager (recommended) or pip

### Install with UV (Recommended)

```bash
# Clone the repository
git clone https://github.com/heinerlehr/correlation_timing.git
cd correlation_timing

# Install in development mode
uv pip install -e .
```

### Install with pip

```bash
pip install -e .
```

### Dependencies

Core dependencies include:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scipy` - Statistical distributions and optimization
- `matplotlib` - Visualization
- `seaborn` - Enhanced plotting
- `pydantic` - Data validation
- `orjson` - Fast JSON parsing
- `loguru` - Logging
- `iconfig` - Configuration management

## Usage

### Command Line Interface

```bash
# Basic usage
python -m correlation_timing.main /path/to/data/directory

# With custom options
python -m correlation_timing.main /path/to/data/directory \
    --max-lookback 6 \
    --no-category \
    --skip-h1 \
    --cumulative

# See all options
python -m correlation_timing.main --help
```

### As a Python Module

```python
from pathlib import Path
from correlation_timing.analysis import run_analysis

# Run complete analysis
results = run_analysis(
    srcdir=Path("/path/to/data"),
    max_lookback_length=4,  # hours
    process_by_category=True,
    run_hypothesis_1=True,
    run_hypothesis_2=True,
    fit_distributions=True,
    cumulative=False
)

# Access results
h1_types = results['hypothesis1']['types']
h2_merged = results['hypothesis2']['merged']
```

### Using Individual Components

```python
from correlation_timing.data_preparation import load_data, prepare_anomalies, create_interval_labels, get_dataset_info
from correlation_timing.hypothesis1 import analyze_hypothesis1
from correlation_timing.hypothesis2 import analyze_hypothesis2

# Load and prepare data
df = load_data(Path("/path/to/data"))
anomalies, earliest_time = prepare_anomalies(df, max_lookback_length=4)

interval_labels = create_interval_labels(4)
info = get_dataset_info(df)
correlations = info['correlations']
number_of_plots = 20

# Run specific hypothesis
result, types = analyze_hypothesis1(
    config=None,
    df=df,
    anomalies=anomalies,
    max_lookback_length=4,
    interval_labels=interval_labels,
    correlations_ordered=correlations,
    number_of_plots=number_of_plots,
    dataset_info=info,
    fit_distributions=True
)
```

## Methodology

### Data Structure

The analysis expects JSON files with the following structure per record:
```json
{
  "AnomalyId": "unique-id",
  "LocalTime": "2024-01-01T12:00:00",
  "FarmId": "farm-123",
  "FarmName": "Farm Name",
  "ShedId": "shed-456",
  "ShedName": "Shed Name",
  "Correlation": "Temperature Increased",
  "Category": "Pigs"
}
```

### Analysis Pipeline

1. **Data Loading**: Reads JSON files and normalizes into pandas DataFrame
2. **Data Preparation**: 
   - Filters anomalies based on lookback window
   - Orders correlations by type or category
   - Creates time interval labels
3. **Hypothesis Testing**:
   - **H1**: Calculates delays from all possible correlations to anomalies
   - **H2**: Calculates delays from same correlation types across anomalies
4. **Distribution Fitting**: Fits mixture distributions (exponential + Weibull/log-normal) to delay data
5. **Visualization**: Generates comprehensive plots showing delay distributions

### Mixture Distribution Fitting

The package fits two-component mixture distributions to model bimodal temporal patterns:

- **Component 1**: Short-term effects (e.g., immediate responses)
- **Component 2**: Long-term effects (e.g., delayed responses)

Models tested:
- Exponential + Weibull
- Exponential + Log-normal
- Weibull + Log-normal

Best model selected using Bayesian Information Criterion (BIC).

### Parallelization

Distribution fitting runs in parallel using `ProcessPoolExecutor`:
- Default: 10 worker processes
- Configurable via `max_workers` parameter
- Automatically handles pickling of data and results
- Each mixture fit uses multiple random restarts for robustness

## Configuration

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `srcdir` | Directory containing JSON data files | Required |
| `--max-lookback` | Maximum lookback window (hours) | 4 |
| `--no-category` | Don't process by category | False |
| `--skip-h1` | Skip Hypothesis 1 analysis | False |
| `--skip-h2` | Skip Hypothesis 2 analysis | False |
| `--no-fit` | Don't fit distributions | False |
| `--cumulative` | Show cumulative plots | False |

### Python API Parameters

```python
run_analysis(
    srcdir: Path,              # Data directory
    max_lookback_length: int,  # Hours to look back
    process_by_category: bool, # Separate by category
    run_hypothesis_1: bool,    # Run H1 analysis
    run_hypothesis_2: bool,    # Run H2 analysis
    fit_distributions: bool,   # Fit mixture models
    cumulative: bool           # Show cumulative %
)
```

## Output

### Plots

The analysis generates comprehensive visualizations showing:
- Delay distribution histograms
- Fitted mixture distribution curves
- Individual component distributions
- Cumulative percentage lines (optional)
- 90% threshold markers

Plots are organized by correlation type and category (if enabled).

### Results Dictionary

```python
{
    'hypothesis1': {
        'merged': DataFrame,  # Raw delay data
        'result': DataFrame,  # Aggregated counts
        'types': TypeList         # Fitted distributions
    },
    'hypothesis2': {
        'merged': DataFrame,
        'result': DataFrame,
        'types': TypeList
    }
}
```


## Authors

- Heiner Lehr


## Version History

- **v0.1.0** (2024-11): Initial release
  - Basic hypothesis testing framework
  - Mixture distribution fitting
  - Parallel processing support
  - Comprehensive visualization
