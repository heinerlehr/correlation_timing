# Correlation Timing Analysis

An advanced statistical analysis tool for optimizing temporal detection windows in livestock farming operations. This project analyzes historical anomaly-correlation data to determine data-driven, optimal lookback windows for identifying correlations between water consumption anomalies and potential causative factors in poultry and pig farming systems.

## Overview

### Problem Statement

In agricultural monitoring systems (like Marvin), when water consumption anomalies occur, the system needs to identify potential causative factors. Currently, fixed time windows are used to look back and find correlations:
- **Poultry**: 1-hour lookback window
- **Pigs**: 2-hour lookback window

However, these fixed windows are not necessarily optimal for all types of correlations. Different correlation types may have different temporal patterns—some factors consistently correlate within minutes, while others take hours to manifest.

### Objective

**Determine the optimal, data-driven time window to look back for each type of correlation when a water anomaly occurs.**

This analysis uses mixture distribution fitting (exponential, Weibull, log-normal) to model temporal patterns and identify when past events are most likely to be causally related to detected anomalies.

## Key Questions

### Hypothesis 1
**"If we measure the time difference between an anomaly and ALL possible correlation factors, how frequently does another anomaly with that correlating factor occur in the past?"**

- Approach: For each water anomaly, look at all historical occurrences of each correlation type in that shed
- Analyzes the distribution of time delays between anomalies and their correlations
- Helps identify which correlations typically manifest in what timeframe

### Hypothesis 2
**"If we measure the time difference for all correlating factors of an anomaly, how far back does the SAME correlating factor occur in other anomalies?"**

- Approach: For each water anomaly, identify its correlations, then find previous occurrences of those same correlations
- Focuses on recurrence patterns of specific correlations
- Reveals how frequently specific factors repeat in the same shed

## Core Components

### 1. Data Preparation (`data_preparation.py`)

**Key Functions:**
- `load_data(srcdir)`: Loads JSON data files containing anomalies and correlations
- `get_dataset_info(df)`: Extracts statistics about the dataset
- `initial_cleaning(df, config)`: Filters data by acceptable categories
- `order_correlations_by_pairs()`: Orders correlations for visualization
- `prepare_anomalies()`: Prepares anomaly data for analysis
- `create_interval_labels()`: Creates time interval bins

**Data Format:**
- Expects JSON files with fields: `AnomalyId`, `LocalTime`, `FarmId`, `FarmName`, `ShedId`, `ShedName`, `Correlation`, `Category`
- Automatically converts `LocalTime` to datetime

### 2. Analysis Module (`analysis.py`)

**Main Function:** `run_analysis()`

Orchestrates the complete workflow:
1. Load data from JSON files
2. Initial data cleaning and validation
3. Extract dataset statistics
4. Order correlations
5. Prepare anomalies
6. Run Hypothesis 1 and/or Hypothesis 2
7. Optionally fit mixture distributions

### 3. Hypothesis 1 (`hypothesis1.py`)

**Analysis Approach:**
- For each water anomaly, examine ALL correlation types in that shed
- Calculate time delay from each correlation occurrence to the anomaly
- Filter to only past events within the lookback window
- Categorize delays into time intervals
- Optionally fit mixture distributions to identify temporal patterns

**Output:**
- Time distribution histograms
- Cumulative probability plots
- Fitted mixture parameters (if enabled)

### 4. Hypothesis 2 (`hypothesis2.py`)

**Analysis Approach:**
- For each water anomaly, identify its associated correlations
- Find previous occurrences of those SAME correlation types in the same shed
- Calculate recurrence intervals
- Categorize into time bins

**Key Functions:**
- `get_timedifferences()`: Calculate delays between anomalies and same-correlation recurrences
- `get_timedifferences_per_category()`: Category-specific analysis

### 5. Distribution Fitting (`utils.py`)

**Mixture Distribution Approach:**
- Fits two-component mixture distributions to time delay data
- Supports three distribution types:
  - **Exponential**: Models rapid, random correlations
  - **Weibull**: Captures both increasing and decreasing hazard rates
  - **Log-normal**: Models correlations with characteristic timescales

**Key Function:** `fit_mixture_simple(data, dist1, dist2)`

**Returns:** `Result` object containing:
- `lambda_`: Mixture weight (0-1)
- `params1`, `params2`: Distribution parameters
- `dist1`, `dist2`: Distribution types
- `nll`, `aic`, `bic`: Goodness-of-fit metrics
- `n`: Sample size

**Parallel Processing:**
- Uses `ProcessPoolExecutor` for fitting multiple correlations simultaneously
- Configurable via `max_workers` parameter

### 6. Visualization (`plotting.py`)

**Functions:**
- `plot_mixture()`: Plots histogram with mixture overlay
- `plot()`: Main plotting orchestrator
- Supports both frequency and cumulative distributions
- Generates publication-quality figures

## Configuration

Configuration is managed via `config/config.yaml`. Key parameters:

```yaml
max_lookback_length: 4              # Maximum lookback window in hours
process_by_category: true           # Process separately by category (Increased/Decreased Water)
run_hypothesis_1: false             # Enable Hypothesis 1 analysis
run_hypothesis_2: true              # Enable Hypothesis 2 analysis
fit_distributions: true             # Fit mixture distributions
cumulative: false                   # Show cumulative plots
max_workers: 10                     # Parallel workers for distribution fitting
save_types: true                    # Save fitted types to JSON

categories:
  - 'Increased Water'               # Water consumption increased
  - 'Decreased Water'               # Water consumption decreased

hypothesis_1:
  fn: "${OUTPUTS}/hypothesis_1.png" # Output file for H1 plots

hypothesis_2:
  fn: "${OUTPUTS}/hypothesis_2.png" # Output file for H2 plots
```

## Usage

### Jupyter Notebooks

The repository includes two analysis notebooks:

#### 1. **Farm-specific-correlations.ipynb**
Advanced, in-depth analysis of farm-specific correlation patterns:
- Data loading and initial cleaning
- DBSCAN clustering of correlated events
- Exponential distribution fitting
- Comparison with random reference data
- Statistical significance testing (log-rank test, permutation KS test)
- Production-ready visualizations

Key workflow sections:
- Load and clean data
- Filter for top farms/sheds
- Cluster adjacent events
- Generate random reference data
- Fit exponential distributions to tail data
- Compare real vs. random distributions

## Output

### Generated Files

#### Hypothesis 1 Results
- `hypothesis_1.png`: Multi-panel figure showing time distributions
- Each panel: One correlation type
- X-axis: Time delay from anomaly to correlation (minutes)
- Y-axis: Frequency or cumulative probability

#### Hypothesis 2 Results
- `hypothesis_2.png`: Recurrence patterns
- Shows how frequently same correlations repeat in same shed
- Time intervals between repeated correlations

#### Fitted Types (JSON)
If `save_types: true` in config:
- `types_by_category.json`: Fitted mixture parameters per correlation/category
- Contains: lambda, distribution types, parameters, AIC/BIC scores

### Result DataFrame Structures

**Hypothesis 1 Result:**
```python
merged = pd.DataFrame({
    'AnomalyId': [...],
    'ShedId': [...],
    'Correlation': [...],
    'Category': [...],
    'Delay': [...],              # Time in minutes
    'DelayInterval': [...],      # Binned time window
    'LocalTime_anomaly': [...],
    'LocalTime_corr': [...]
})
```

**Hypothesis 2 Result:**
```python
result = pd.DataFrame({
    'AnomalyId': [...],
    'ShedId': [...],
    'Correlation': [...],
    'Delay': [...]               # Recurrence interval in minutes
})
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.3.3 | Data manipulation and analysis |
| numpy | (via scipy) | Numerical operations |
| scipy | >=1.16.3 | Statistical distributions, optimization |
| scipy-stubs | >=1.18.1.1 | Type hints for scipy |
| scikit-learn | >=1.8.0 | DBSCAN clustering |
| matplotlib | >=3.10.7 | Visualization |
| seaborn | >=0.13.2 | Statistical plots |
| lifelines | >=0.30.0 | Survival analysis (log-rank test) |

### Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=9.0.1 | Unit testing |
| debugpy | >=1.8.17 | Debugging support |
| ipykernel | >=7.1.0 | Jupyter kernel |
| loguru | >=0.7.3 | Logging |
| pydantic | >=2.12.4 | Data validation |
| iconfig-py | >=0.1.7 | YAML configuration |
| orjson | >=3.11.4 | Fast JSON parsing |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ct tests/

# Run specific test
pytest tests/test_utils.py::test_fit_mixture_simple_expon
```

### Test Coverage

- `test_utils.py`: Tests for mixture distribution fitting
  - Exponential distribution fitting
  - Weibull distribution fitting
  - Log-normal distribution fitting
  - NaN handling
  - Parameter validation

## Advanced Topics

### Mixture Distribution Fitting

The package uses maximum likelihood estimation to fit two-component mixture distributions:

$$f(x) = \lambda \cdot f_1(x; \theta_1) + (1-\lambda) \cdot f_2(x; \theta_2)$$

Where:
- $\lambda$ is the mixture weight (0 < λ < 1)
- $f_1, f_2$ are probability density functions
- $\theta_1, \theta_2$ are distribution parameters

**Optimization:** Negative log-likelihood minimization using `scipy.optimize.minimize`

**Goodness-of-Fit Metrics:**
- **NLL** (Negative Log-Likelihood): Lower is better
- **AIC** (Akaike Information Criterion): Balances fit quality with model complexity
- **BIC** (Bayesian Information Criterion): Stronger penalty for complexity than AIC

### Parallel Processing

For large datasets with many correlations, distribution fitting is parallelized:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=10) as executor:
    results = executor.map(fit_mixture_simple, data_chunks)
```

### Time Interval Binning

Time delays are categorized into uniform bins (e.g., 5-minute intervals):

```python
bins = pd.interval_range(start=0, end=12*60, freq=5, closed='left')
# Creates: [0, 5), [5, 10), [10, 15), ..., [715, 720)
```

## Examples and Workflows

### Example 1: Complete Analysis Pipeline

```python
from pathlib import Path
from ct.analysis import run_analysis
from iconfig.iconfig import iConfig

config = iConfig()
srcdir = Path("/mnt/data/chicken_anomalies")

# Run both hypotheses with distribution fitting
run_analysis(
    config=config,
    srcdir=srcdir,
    max_lookback_length=4,
    process_by_category=True,
    run_hypothesis_1=True,
    run_hypothesis_2=True,
    fit_distributions=True
)
```

### Example 2: Hypothesis 2 Only with Custom Parameters

```python
run_analysis(
    config=config,
    srcdir=srcdir,
    max_lookback_length=6,
    process_by_category=True,
    run_hypothesis_1=False,
    run_hypothesis_2=True,
    fit_distributions=False  # Skip distribution fitting
)
```

### Example 3: Manual Hypothesis Analysis

```python
import pandas as pd
from pathlib import Path
from ct.data_preparation import load_data, prepare_anomalies, create_interval_labels
from ct.hypothesis1 import analyze_hypothesis1

# Load and prepare data
df = load_data(Path("/data/anomalies"))
anomalies, earliest_time = prepare_anomalies(df, max_lookback_length=4)
intervals, labels = create_interval_labels(4)

# Run Hypothesis 1
merged, aggregated, types = analyze_hypothesis1(
    config=config,
    df=df,
    anomalies=anomalies,
    max_lookback_length=4,
    intervals=intervals,
    interval_labels=labels,
    # ... additional parameters
)

# Access results
print(f"Found {len(merged)} correlation occurrences")
```

## Interpretation Guide

### Reading Mixture Distribution Results

When a correlation shows a fitted mixture distribution:

1. **Lambda (λ)**: Proportion of events following first distribution
   - λ ≈ 0.7, 0.3: Well-separated two modes
   - λ ≈ 0.5: Equally mixed components
   - λ ≈ 0.95: One dominant component

2. **Distribution Types**:
   - Exponential + Exponential: Random events at different rates
   - Exponential + Weibull: Mix of random and patterned events
   - Weibull + Log-normal: Structured temporal patterns

3. **AIC/BIC**:
   - Lower values indicate better fit
   - Use for model comparison

### Optimizing Time Windows

Based on results:

1. **If exponential distribution dominates**: Events are random, fixed window OK
2. **If bimodal with sharp peaks**: Optimize window to align with peaks
3. **If long tail**: Need longer window to capture late-occurring correlations
4. **If strong mode < 30 minutes**: Can use shorter window for faster response

## Performance Considerations

- **Memory**: Loads full dataset into memory; suitable for data < 1GB
- **CPU**: Parallelized distribution fitting scales with core count
- **Time**: Typical analysis of 50k anomalies across 100+ farms: 5-15 minutes

### Optimization Tips

1. Enable `process_by_category=True` for parallel processing per category
2. Adjust `max_workers` based on system cores
3. Use smaller `max_lookback_length` for faster initial exploration
4. Set `fit_distributions=False` for quick preliminary analysis

## References

### Statistical Methods

- **Mixture Models**: Reynolds, D. A. (2009). Gaussian mixture models.
- **Weibull Distribution**: Rinne, H. (2009). The Weibull Distribution.
- **Log-Normal Distribution**: Crow, E. L., & Shimizu, K. (1988).

### Related Work

- DBSCAN Clustering (used in notebooks): Ester et al., 1996
- Log-rank Test (statistical comparison): Mantel, N., 1966
- Permutation Tests: Phipson & Smyth, 2010

---

**Last Updated**: 2024  
**Python Support**: 3.13+  
**Status**: Active Development
