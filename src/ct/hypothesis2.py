"""Hypothesis 2: Time difference for correlating factors within same anomaly.

If we measure the time difference for all correlating factors of an anomaly, 
we would see how far back the same correlating factor occurs in other anomalies.
"""

import pandas as pd
from typing import Tuple

from iconfig.iconfig import iConfig

from loguru import logger

from ct.utils import determine_type, determine_type_by_category
from ct.plotting import plot


def get_timedifferences(
    df: pd.DataFrame, 
    anomalies: pd.DataFrame, 
    max_lookback_length: int
) -> pd.DataFrame:
    """Calculate time differences for same correlation types.
    
    For each anomaly-correlation pair, find previous occurrences of that 
    SAME correlation type in the same shed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Main dataset with correlations
    anomalies : pd.DataFrame
        Unique anomalies DataFrame
    max_lookback_length : int
        Maximum lookback window in hours
        
    Returns
    -------
    pd.DataFrame
        DataFrame with AnomalyId, ShedId, Correlation, and Delay columns
    """

    # Make sure LocalTime is datetime
    df["LocalTime"] = pd.to_datetime(df["LocalTime"])
    anomalies["LocalTime"] = pd.to_datetime(anomalies["LocalTime"])


    logger.info("Computing time differences for same correlation types...")
    # Get unique correlations tied to anomalies (vectorized explode)
    anomaly_corr = (
        df.merge(anomalies[["ShedId", "AnomalyId", "LocalTime"]], 
                on=["ShedId", "AnomalyId"], 
                how="inner")
        .loc[:, ["ShedId", "AnomalyId", "Correlation", "LocalTime_y"]]
        .rename(columns={"LocalTime_y": "anomaly_time"})
        .drop_duplicates()
    )
    # Self-join to find previous SAME correlation in same shed
    prev_corr = (
        df.loc[:, ["ShedId", "Correlation", "LocalTime"]]
        .merge(anomaly_corr, 
            on=["ShedId", "Correlation"], 
            how="inner", 
            suffixes=("_prev", "_anom"))
    )
    # Vectorised delay computation
    prev_corr["Delay"] = (
        prev_corr["anomaly_time"] - prev_corr["LocalTime_prev"]
    ).dt.total_seconds() / 60

    # Filter:
    # 1. Only past events
    # 2. Only within lookback window in minutes
    lookback_minutes = max_lookback_length * 60
    result = prev_corr.query("Delay > 0 and Delay <= @lookback_minutes")

    # Select final columns
    result = result.loc[:, ["AnomalyId", "ShedId", "Correlation", "Delay"]]

    # Optional safeguard if empty
    return result



def get_timedifferences_per_category(
    df: pd.DataFrame, 
    anomalies: pd.DataFrame, 
    max_lookback_length: int
) -> pd.DataFrame:
    # Ensure datetime dtype
    df["LocalTime"] = pd.to_datetime(df["LocalTime"])
    anomalies["LocalTime"] = pd.to_datetime(anomalies["LocalTime"])

    # Sanity check (as in original)
    categories = anomalies["Category"].unique()
    assert len(categories) == 2

    logger.info("Computing time differences per category...")
    # 1. Create anomaly/correlation mappings in one pass
    # We only need correlations tied to anomalies, so we inner-join on anomaly keys
    anomaly_corr = (
        df.merge(
            anomalies[["ShedId", "AnomalyId", "Category", "LocalTime"]],
            on=["ShedId", "AnomalyId", "Category"],
            how="inner",
            suffixes=("", "_anom")
        )
        .loc[:, ["ShedId", "AnomalyId", "Category", "Correlation", "LocalTime_anom"]]
        .rename(columns={"LocalTime_anom": "anomaly_time"})
        .drop_duplicates()
    )

    # 2. Self-join df with anomaly_corr on correlation + shed + category
    prev = (
        df.loc[:, ["ShedId", "Category", "Correlation", "LocalTime"]]
        .rename(columns={"LocalTime": "prev_time"})
        .merge(anomaly_corr, on=["ShedId", "Category", "Correlation"], how="inner")
    )

    # 3. Vectorised delay computation (minutes)
    prev["Delay"] = (prev["anomaly_time"] - prev["prev_time"]).dt.total_seconds() / 60

    # 4. Filter only past events inside lookback window
    lookback_minutes = max_lookback_length * 60
    result = prev.query("Delay > 0 and Delay <= @lookback_minutes")

    # 5. Select final columns
    return result.loc[:, ["AnomalyId", "ShedId", "Correlation", "Category", "Delay"]]



def analyze_hypothesis2(
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
) -> tuple:
    """Analyze Hypothesis 2: Time differences for same correlation types.
    
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
        result data, aggregated results, and fitted types (if requested)
    """
    logger.info("HYPOTHESIS 2: Time difference for same correlation types")
    
    if process_by_category:
        merged = get_timedifferences_per_category(df, anomalies, max_lookback_length)
        # Categorize and count
        merged['DelayInterval'] = pd.cut(
            merged['Delay'],
            bins=range(0, max_lookback_length*60+1, 15),
            labels=interval_labels,
            right=False
        )

    else:
        merged = get_timedifferences(df, anomalies, max_lookback_length)
        # Categorize and count
        merged['DelayInterval'] = pd.cut(
            merged['Delay'],
            bins=range(0, max_lookback_length*60+1, 15),
            labels=interval_labels,
            right=False
        )
    
    logger.info(f"Found {len(merged)} correlation occurrences")
    
    # Fit distributions if requested
    types = None
    if fit_distributions:
        logger.info("Fitting mixture distributions...")
        if process_by_category:
            types = determine_type_by_category(
                correlations=correlations_ordered,
                corr_data=merged
            )
        else:
            types = determine_type(
                correlations=correlations_ordered,
                corr_data=merged
            )
    
    # Plot results
    logger.info("Generating plots...")
    fn = config('hypothesis_2.fn', default='hypothesis_2.png')
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
