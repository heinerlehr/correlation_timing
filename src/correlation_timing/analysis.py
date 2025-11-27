
from pathlib import Path
import warnings

from iconfig.iconfig import iConfig
from loguru import logger

from correlation_timing.data_preparation import (
    load_data,
    get_dataset_info,
    order_correlations_by_pairs,
    order_correlations_by_category,
    prepare_anomalies,
    create_interval_labels,
)
from correlation_timing.hypothesis1 import analyze_hypothesis1
from correlation_timing.hypothesis2 import analyze_hypothesis2

def run_analysis(
        config: iConfig,
        srcdir: Path,
        max_lookback_length: int = 4,
        process_by_category: bool = True,
        run_hypothesis_1: bool = True,
        run_hypothesis_2: bool = True,
        fit_distributions: bool = True,
        cumulative: bool = False,
):
    """Run the complete correlation timing analysis.
    
    Parameters
    ----------
    srcdir : Path
        Directory containing JSON data files
    max_lookback_length : int
        Maximum lookback window in hours (default: 4)
    process_by_category : bool
        Whether to process by category (default: True)
    run_hypothesis_1 : bool
        Whether to run Hypothesis 1 analysis (default: True)
    run_hypothesis_2 : bool
        Whether to run Hypothesis 2 analysis (default: True)
    fit_distributions : bool
        Whether to fit mixture distributions (default: True)
    cumulative : bool
        Whether to show cumulative plots (default: False)
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    logger.info("CORRELATION TIMING ANALYSIS")
    logger.info(f"Source directory: {srcdir}")
    logger.info(f"Max lookback length: {max_lookback_length} hours")
    logger.info(f"Process by category: {process_by_category}")
    logger.info(f"Run Hypothesis 1: {run_hypothesis_1}")
    logger.info(f"Run Hypothesis 2: {run_hypothesis_2}")
    logger.info(f"Fit distributions: {fit_distributions}")
    logger.info(f"Cumulative plots: {cumulative}")
    # 1. Load data
    logger.info("Loading data...")
    df = load_data(srcdir)
    logger.info(f"Loaded {len(df)} records\n")
    
    # 2. Get dataset info
    dataset_info = get_dataset_info(df)
    logger.info("Dataset contains:")
    logger.info(f"  - {dataset_info['n_correlations']} unique correlations")
    logger.info(f"  - {dataset_info['n_categories']} unique categories")
    logger.info(f"  - {dataset_info['n_farms']} farms")
    logger.info(f"  - {dataset_info['n_sheds']} sheds")
    logger.info(f"  - {dataset_info['n_anomalies']} unique anomalies\n")
    
    # 3. Order correlations
    if process_by_category:
        number_of_plots, correlations_ordered = order_correlations_by_category(
            correlations=dataset_info['correlations'],
            df=df
        )
    else:
        number_of_plots, correlations_ordered = order_correlations_by_pairs(
            correlations=dataset_info['correlations']
        )
    
    logger.info(f"Ordered {len(correlations_ordered)} correlations ({number_of_plots} plots)\n")
    
    # 4. Prepare anomalies
    anomalies, earliest_time = prepare_anomalies(df, max_lookback_length)
    logger.info(f"Found {len(anomalies)} unique anomalies after {earliest_time}\n")
    
    # 5. Create interval labels
    intervals, interval_labels = create_interval_labels(max_lookback_length)
    
    # 6. Run Hypothesis 1
    if run_hypothesis_1:
        merged1, types1 = analyze_hypothesis1(
            config=config,
            df=df,
            anomalies=anomalies,
            max_lookback_length=max_lookback_length,
            intervals=intervals,
            interval_labels=interval_labels,
            correlations_ordered=correlations_ordered,
            number_of_plots=number_of_plots,
            dataset_info=dataset_info,
            process_by_category=process_by_category,
            fit_distributions=fit_distributions,
            cumulative=cumulative,
        )
    
    # 7. Run Hypothesis 2
    if run_hypothesis_2:
        merged2, types2 = analyze_hypothesis2(
            config=config,
            df=df,
            anomalies=anomalies,
            max_lookback_length=max_lookback_length,
            intervals = intervals,
            interval_labels= interval_labels,
            correlations_ordered=correlations_ordered,
            number_of_plots=number_of_plots,
            dataset_info=dataset_info,
            process_by_category=process_by_category,
            fit_distributions=fit_distributions,
            cumulative=cumulative,
        )
    
    logger.info("ANALYSIS COMPLETE")
    
    # Return results for further analysis if needed
    results = {}
    if run_hypothesis_1:
        results['hypothesis1'] = {
            'merged': merged1,
            'types': types1,
        }
    if run_hypothesis_2:
        results['hypothesis2'] = {
            'merged': merged2,
            'types': types2,
        }
