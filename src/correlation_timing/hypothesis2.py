"""Hypothesis 2: Time difference for correlating factors within same anomaly.

If we measure the time difference for all correlating factors of an anomaly, 
we would see how far back the same correlating factor occurs in other anomalies.
"""

import pandas as pd
from typing import Tuple

from iconfig.iconfig import iConfig

from loguru import logger

from correlation_timing.utils import determine_type, determine_type_by_category, plot


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
    result_list = []

    for _, anomaly_row in anomalies.iterrows():
        shed_id = anomaly_row['ShedId']
        anomaly_id = anomaly_row['AnomalyId']
        anomaly_time = anomaly_row['LocalTime']
        
        # Get all correlations associated with this specific anomaly
        anomaly_correlations = df[
            (df['ShedId'] == shed_id) & 
            (df['AnomalyId'] == anomaly_id)
        ]['Correlation'].unique()
        
        # For each correlation type associated with this anomaly
        for corr_type in anomaly_correlations:
            # Find previous occurrences of this SAME correlation type in the same shed
            previous_same_corr = df[
                (df['ShedId'] == shed_id) & 
                (df['Correlation'] == corr_type) & 
                (df['LocalTime'] < anomaly_time)
            ]
            
            # Calculate time differences
            for _, prev_row in previous_same_corr.iterrows():
                delay_minutes = (anomaly_time - prev_row['LocalTime']).total_seconds() / 60
                
                # Only include if within lookback window
                if delay_minutes <= max_lookback_length * 60:
                    result_list.append({
                        'AnomalyId': anomaly_id,
                        'ShedId': shed_id,
                        'Correlation': corr_type,
                        'Delay': delay_minutes
                    })

    return pd.DataFrame(result_list)


def get_timedifferences_per_category(
    df: pd.DataFrame, 
    anomalies: pd.DataFrame, 
    max_lookback_length: int
) -> pd.DataFrame:
    """Calculate time differences for same correlation types, separated by category.
    
    For each anomaly-correlation pair, find previous occurrences of that 
    SAME correlation type in the same shed and category.
    
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
        DataFrame with AnomalyId, ShedId, Correlation, Category, and Delay columns
    """
    result_list = []
    categories = anomalies['Category'].unique()
    assert len(categories) == 2

    for _, anomaly_row in anomalies.iterrows():
        shed_id = anomaly_row['ShedId']
        anomaly_id = anomaly_row['AnomalyId']
        anomaly_time = anomaly_row['LocalTime']
        category = anomaly_row['Category']
        
        # Get all correlations associated with this specific anomaly
        anomaly_correlations = df[
            (df['ShedId'] == shed_id) & 
            (df['AnomalyId'] == anomaly_id) & 
            (df['Category'] == category)
        ]['Correlation'].unique()
        
        # For each correlation type associated with this anomaly
        for corr_type in anomaly_correlations:
            # Find previous occurrences of this SAME correlation type in the same shed
            previous_same_corr = df[
                (df['ShedId'] == shed_id) & 
                (df['Correlation'] == corr_type) & 
                (df['Category'] == category) &
                (df['LocalTime'] < anomaly_time)
            ]
            
            # Calculate time differences
            for _, prev_row in previous_same_corr.iterrows():
                delay_minutes = (anomaly_time - prev_row['LocalTime']).total_seconds() / 60
                
                # Only include if within lookback window
                if delay_minutes <= max_lookback_length * 60:
                    result_list.append({
                        'AnomalyId': anomaly_id,
                        'ShedId': shed_id,
                        'Correlation': corr_type,
                        'Category': category,
                        'Delay': delay_minutes
                    })

    return pd.DataFrame(result_list)


def analyze_hypothesis2(
    config: iConfig,
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    max_lookback_length: int,
    interval_labels: list[str],
    correlations_ordered: list,
    number_of_plots: int,
    dataset_info: dict,
    process_by_category: bool = True,
    fit_distributions: bool = True,
    cumulative: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, dict | None]:
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
        result2 = get_timedifferences_per_category(df, anomalies, max_lookback_length)
        # Categorize and count
        result2['DelayInterval'] = pd.cut(
            result2['Delay'],
            bins=range(0, max_lookback_length*60+1, 15),
            labels=interval_labels,
            right=False
        )
        merged2 = result2.groupby(
            ['Correlation', 'Category', 'DelayInterval']
        ).size().reset_index(name='Count')
    else:
        result2 = get_timedifferences(df, anomalies, max_lookback_length)
        # Categorize and count
        result2['DelayInterval'] = pd.cut(
            result2['Delay'],
            bins=range(0, max_lookback_length*60+1, 15),
            labels=interval_labels,
            right=False
        )
        merged2 = result2.groupby(
            ['Correlation', 'DelayInterval']
        ).size().reset_index(name='Count')
    
    logger.info(f"Found {len(result2)} correlation occurrences")
    
    # Fit distributions if requested
    types = None
    if fit_distributions:
        if process_by_category:
            types = determine_type_by_category(
                correlations=correlations_ordered,
                corr_data=result2
            )
        else:
            types = determine_type(
                correlations=correlations_ordered,
                corr_data=result2
            )
    
    # Plot results
    fn = config('hypothesis_2.fn', default='hypothesis_2.png')
    max_lookback_length = config('max_lookback_length', default=4)
    save = config.get('save_types', default=False)
    plot(
        config=config,
        number_of_plots=number_of_plots,
        result=result2,
        df=df,
        correlations=correlations_ordered,
        dataset_info=dataset_info,
        process_by_category=process_by_category,
        types=types,
        cumulative=cumulative,
        fn = fn,
        max_lookback_length=max_lookback_length,
        save = save,
    )
    
    return result2, merged2, types
