"""Hypothesis 1: Time difference between anomaly and all possible correlation factors.

If we measure the time difference between an anomaly and all possible correlation 
factors, we should see how frequently another anomaly with that correlating factor 
occurs in the past.
"""

import pandas as pd
from typing import Tuple

from iconfig.iconfig import iConfig

from loguru import logger

from ct.utils import determine_type, determine_type_by_category
from ct.plotting import plot


def analyze_hypothesis1(
    config: iConfig,
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    max_lookback_length: int,
    intervals: list[int],
    interval_labels: list[str],
    correlations_ordered: list,
    number_of_plots: int,
    dataset_info: dict,
    process_by_category: bool = True,
    fit_distributions: bool = True,
    cumulative: bool = False
) -> Tuple[pd.DataFrame, dict | None]:
    """Analyze Hypothesis 1: Time differences to all correlation factors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main dataset with correlations
    anomalies : pd.DataFrame
        Unique anomalies DataFrame
    max_lookback_length : int
        Maximum lookback window in hours
    interval_labels : list[str]
        Time interval labels for categorization
    correlations_ordered : list
        Ordered list of correlations to analyze
    number_of_plots : int
        Number of plots needed
    dataset_info : dict
        Dataset statistics
    process_by_category : bool
        Whether to process by category
    fit_distributions : bool
        Whether to fit mixture distributions
    cumulative : bool
        Whether to show cumulative plots
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict | None]
        merged data, aggregated results, and fitted types (if requested)
    """

    logger.info("HYPOTHESIS 1: Time difference to all correlation factors")
    
    if not process_by_category:
        # Self-join approach
        merged = anomalies.merge(df, on='ShedId', suffixes=('_anomaly', '_corr'))

        # Filter: correlation must be before anomaly
        merged = merged[merged['LocalTime_corr'] < merged['LocalTime_anomaly']]

        # Calculate time difference
        merged['Delay'] = (
            merged['LocalTime_anomaly'] - merged['LocalTime_corr']
        ).dt.total_seconds() / 60

        # Filter: only within lookback window
        merged = merged[merged['Delay'] <= max_lookback_length * 60]

        # Categorize and count
        merged['DelayInterval'] = pd.cut(
            merged['Delay'], 
            bins=intervals,
            labels=interval_labels, 
            right=False
        )
    else:
        # Process each category separately
        mergeds = []
        categories = dataset_info['categories']
        
        for category in categories:
            anomalies_cat = anomalies[anomalies['Category'] == category]
            df_cat = df[df['Category'] == category]

            merged = anomalies_cat.merge(
                df_cat, 
                on=['ShedId', 'Category'], 
                suffixes=('_anomaly', '_corr')
            )

            # Filter: correlation must be before anomaly
            merged = merged[merged['LocalTime_corr'] < merged['LocalTime_anomaly']]

            # Calculate time difference
            merged['Delay'] = (
                merged['LocalTime_anomaly'] - merged['LocalTime_corr']
            ).dt.total_seconds() / 60

            # Filter: only within lookback window
            merged = merged[merged['Delay'] <= max_lookback_length * 60]

            # Categorize and count
            merged['DelayInterval'] = pd.cut(
                merged['Delay'],
                bins=intervals,
                labels=interval_labels,
                right=False
            )
            mergeds.append(merged)
        
        merged = pd.concat(mergeds, ignore_index=True)
    
    logger.info(f"Found {len(merged)} correlation occurrences")
    
    # Fit distributions if requested
    max_workers = config('max_workers', default=10)
    types = None
    if fit_distributions:
        if process_by_category:
            types = determine_type_by_category(
                correlations=correlations_ordered, 
                corr_data=merged,
                max_workers=max_workers
            )
        else:
            types = determine_type(
                correlations=correlations_ordered, 
                corr_data=merged,
                max_workers=max_workers
            )
    
    # Plot results
    fn = config('hypothesis_1.fn', default='hypothesis_1.png')
    max_lookback_length = config('max_lookback_length', default=4)
    plot(
        number_of_plots=number_of_plots,
        result=merged,
        df=df,
        correlations=correlations_ordered,
        dataset_info=dataset_info,
        process_by_category=process_by_category,
        types=types,
        cumulative=cumulative,
        fn = fn,
        max_lookback_length=max_lookback_length,
        intervals = intervals,
        interval_labels=interval_labels,
    )
    
    return merged, types
