"""Data loading and preparation for correlation timing analysis."""

from pathlib import Path
from typing import Tuple

import orjson
import pandas as pd
import numpy as np

def load_data(srcdir: Path) -> pd.DataFrame:
    """Load JSON data files and combine into a single DataFrame.
    
    Parameters
    ----------
    srcdir : Path
        Directory containing JSON files with anomaly and correlation data
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: AnomalyId, LocalTime, FarmId, FarmName,
        ShedId, ShedName, Correlation, Category
    """
    files = [f for f in srcdir.iterdir() if f.suffix == '.json']
    df = pd.DataFrame()
    
    for f in files:
        with open(f, 'r') as infile:
            data = orjson.loads(infile.read())
            t_df = pd.json_normalize(data)
            t_df['LocalTime'] = pd.to_datetime(t_df['LocalTime'])  # type:ignore
            t_df = t_df[['AnomalyId', 'LocalTime', 'FarmId', 'FarmName', 
                         'ShedId', 'ShedName', 'Correlation', 'Category']]
            df = pd.concat([df, t_df], ignore_index=True)
    
    df = df.sort_values(by='LocalTime')
    df = df.reset_index(drop=True)
    
    # Clean correlation strings
    df['Correlation'] = [corr.strip() for corr in df['Correlation']]
    
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Extract basic information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset
        
    Returns
    -------
    dict
        Dictionary containing counts and lists of unique values
    """
    correlations = df['Correlation'].sort_values().unique()
    categories = df['Category'].unique()
    farmids = df['FarmId'].unique()
    shedids = df['ShedId'].unique()
    anomalyids = df['AnomalyId'].unique()
    
    return {
        'correlations': correlations,
        'categories': categories,
        'farmids': farmids,
        'shedids': shedids,
        'anomalyids': anomalyids,
        'n_correlations': len(correlations),
        'n_categories': len(categories),
        'n_farms': len(farmids),
        'n_sheds': len(shedids),
        'n_anomalies': len(anomalyids),
    }


def order_correlations_by_pairs(correlations: list | np.ndarray) -> Tuple[int, list]:
    """Order correlations by pairing Increased/Decreased variants.
    
    Parameters
    ----------
    correlations : list or np.ndarray
        List of correlation names
        
    Returns
    -------
    Tuple[int, list]
        Number of plots needed and ordered list of correlations
    """
    pairs = []
    singles = []

    special_pairs = [('Sunset', 'Sunrise'), ('Lights On', 'Lights Off')]

    for correlation in correlations:
        if correlation.endswith('Increased'):
            cat_name = correlation.split('Increased')[0]
            if f"{cat_name}Decreased" in correlations:
                pairs.append(correlation)
        elif correlation.endswith('Decreased'):
            cat_name = correlation.split('Decreased')[0]
            if f"{cat_name}Increased" in correlations:
                pairs.append(correlation)
        elif any([correlation in pair for pair in special_pairs]):
            for pair in special_pairs:
                if correlation in pair:
                    other = pair[1] if correlation == pair[0] else pair[0]
                    break
            if other in correlations:
                pairs.append(correlation)
            else:
                singles.append(correlation)
        else:
            singles.append(correlation)
    
    pairs.sort()  # automatic ordering
    singles.sort()

    return len(pairs + singles), pairs + singles


def order_correlations_by_category(
    correlations: list | np.ndarray, 
    df: pd.DataFrame
) -> Tuple[int, list]:
    """Order correlations by category presence.
    
    Parameters
    ----------
    correlations : list or np.ndarray
        List of correlation names
    df : pd.DataFrame
        The main dataset
        
    Returns
    -------
    Tuple[int, list]
        Number of plots needed and ordered list of correlations
    """
    pairs = []
    singles = []
    categories = df['Category'].unique()
    assert len(categories) == 2

    for correlation in correlations:
        if len(df[df['Correlation'] == correlation]['Category'].unique()) == 2:
            pairs.append(correlation)
        else:
            singles.append(correlation)
    
    pairs.sort()  # automatic ordering
    singles.sort()

    return len(pairs) * 2 + len(singles), pairs + singles


def prepare_anomalies(
    df: pd.DataFrame, 
    max_lookback_length: int
) -> Tuple[pd.DataFrame, pd.Timedelta]:
    """Prepare unique anomalies with time filtering.
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset
    max_lookback_length : int
        Maximum lookback window in hours
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Timedelta]
        Filtered anomalies DataFrame and earliest valid time
    """
    start_of_dataset = df['LocalTime'].min()
    earliest_time = start_of_dataset + pd.Timedelta(hours=max_lookback_length)
    
    # Find all unique anomalies for each shed at least max_lookback_length after start
    anomalies = df.groupby(
        by=['ShedId', 'AnomalyId', 'Category']
    ).agg({'LocalTime': 'min'}).reset_index()
    anomalies = anomalies[anomalies['LocalTime'] > earliest_time]
    
    return anomalies, earliest_time


def create_interval_labels(max_lookback_length: int) -> list[str]:
    """Create time interval labels for categorization.
    
    Parameters
    ----------
    max_lookback_length : int
        Maximum lookback window in hours
        
    Returns
    -------
    list[str]
        List of interval labels like "0-15min", "15-30min", etc.
    """
    return [f"{i*15}-{(i+1)*15}min" 
            for i in range(int(max_lookback_length * 60 / 15))]
