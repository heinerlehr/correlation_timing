

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from scipy.stats import expon, weibull_min, lognorm


from loguru import logger

from correlation_timing.utils import Result, TypeList


def plot_mixture(data:pd.DataFrame, result: Result, intervals: list[int], interval_labels: list[str], 
                 cumulative:bool=False, ax: Axes | None = None) -> Axes | Figure:
    """Plot data histogram with fitted mixture overlay.
    
    Parameters
    ----------
    data : array
        The data that was fitted
    result : Result
        Fitted mixture model result
    bins : int
        Number of histogram bins
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
        
    Returns
    -------
    Axes or Figure
        The axes if provided, otherwise the figure
    """
    D1 = {'expon': expon, 'weibull_min': weibull_min, 'lognorm': lognorm}[result.dist1]
    D2 = {'expon': expon, 'weibull_min': weibull_min, 'lognorm': lognorm}[result.dist2]
    
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 5))

    # Calculate percentage and cumulative percentage
    grouped_results = data.groupby('DelayInterval').size().reset_index(name='Count')
    grouped_results['Percent'] = 100 * grouped_results['Count'] / grouped_results['Count'].sum()
    grouped_results = grouped_results.sort_values('DelayInterval')
    grouped_results['CumPercent'] = grouped_results['Percent'].cumsum()
    
    # Assign colors: blue for first 90%, orange for rest
    colors = ['#1f77b4' if cum <= 90 else '#ff7f0e' for cum in grouped_results['CumPercent']]

    # Histogram
    # sns.histplot(data=data, x='Delay', stat='percent', bins=intervals, palette=colors, ax=ax)
    counts, bins = np.histogram(data['Delay'], bins=intervals)
    percent = 100 * counts / counts.sum()
    ax.bar(bins[:-1], percent, width=np.diff(bins), color=colors, align='edge', edgecolor='black')

    # Overlay percent-per-bin for the fitted distributions
    w1 = result.lambda_
    w2 = 1 - result.lambda_
    n_bins = len(intervals) - 1
    percent1 = []
    percent2 = []
    percent_mix = []
    bin_centers = []
    for i in range(n_bins):
        lower = intervals[i]
        upper = intervals[i+1]
        center = (lower + upper) / 2
        bin_centers.append(center)
        p1 = D1.cdf(upper, *result.params1) - D1.cdf(lower, *result.params1)
        p2 = D2.cdf(upper, *result.params2) - D2.cdf(lower, *result.params2)
        percent1.append(w1 * p1 * 100)
        percent2.append(w2 * p2 * 100)
        percent_mix.append((w1 * p1 + w2 * p2) * 100)

    ax.plot(bin_centers, percent1, 'r-', lw=2.5, marker='o', label=f'{result.dist1} ({w1:.1%})')
    ax.plot(bin_centers, percent2, 'b-', lw=2.5, marker='s', label=f'{result.dist2} ({w2:.1%})')
    ax.plot(bin_centers, percent_mix, 'k--', lw=3, marker='^', label='Mixture')
    ax.legend(loc='upper right')

    if cumulative:
        # Add cumulative line on secondary axis
        ax2 = ax.twinx()
        midpoints = [(intervals[i] + intervals[i+1]) / 2 for i in range(len(intervals) - 1)]
        ax2.plot(midpoints, grouped_results['CumPercent'].values, 
                 color='darkred', linestyle='-', linewidth=2, 
                 label='Cumulative %')  # type:ignore
        ax2.axhline(y=90, color='darkred', linestyle='--', alpha=0.5, 
                    label='90% threshold')
        ax2.set_ylabel('Cumulative %', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.set_ylim(0, 105)

    ax.set_xlabel('Time Delay')
    ax.set_ylabel('Percent')

    # Prepare table data
    param_labels = {
        'expon': ['loc', 'scale'],
        'weibull_min': ['c', 'loc', 'scale'],
        'lognorm': ['s', 'loc', 'scale']
    }
    table_data = []
    table_data.append(["λ", f"{result.lambda_:.3f}"])
    for i, val in enumerate(result.params1):
        label = param_labels[result.dist1][i] if i < len(param_labels[result.dist1]) else f"param{i+1}"
        table_data.append([f"{result.dist1} {label}", f"{val:.3f}"])
    for i, val in enumerate(result.params2):
        label = param_labels[result.dist2][i] if i < len(param_labels[result.dist2]) else f"param{i+1}"
        table_data.append([f"{result.dist2} {label}", f"{val:.3f}"])

    # Add table to plot (bottom right)
    left, bottom, width, height = 0.66, 0.40, 0.33, 0.38
    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value"],
        cellLoc='left',
        loc='lower right',
        bbox=[left, bottom, width, height]  # [left, bottom, width, height]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Make 'Value' column narrower and right-aligned
    nrows = len(table_data) + 1  # +1 for header
    param_col_width = 0.70 * width  # 75% of 0.3
    value_col_width = 0.30 * width  # 25% of 0.3
    for row in range(nrows):
        cell_param = table[(row, 0)]
        cell_value = table[(row, 1)]
        cell_param.set_width(param_col_width)
        cell_value.set_width(value_col_width)
        cell_value.set_text_props(ha='right')

    bic = f"{result.bic:.1f}" if result.bic is not None else "N/A"
    ax.set_title(f'{w1:.1%} {result.dist1} + {w2:.1%} {result.dist2} (BIC={bic})')
    ax.grid(alpha=0.3)

    if not ax:
        return fig
    else:
        return ax

def plot_subset(
    ax: Axes, 
    subset: pd.DataFrame, 
    n_sample: int, 
    corr: str,
    intervals: list[int],
    interval_labels: list[str],
    types: TypeList | None = None, 
    category: str | None = None, 
    max_lookback_length:int=4,
    cumulative: bool = True
) -> None:
    """Plot a single correlation's delay distribution.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    subset : pd.DataFrame
        Subset of data for this correlation
    n_sample : int
        Total number of samples
    corr : str
        Correlation name
    types : dict, optional
        Dictionary of fitted distribution types
    category : str, optional
        Category name for filtering
    cumulative : bool
        Whether to show cumulative percentage
    """
    if subset.empty:
        ax.set_title(f'Correlation: {corr} {category} (No Data)')
        ax.axis('off')  # Remove empty plot
        return

    # Plot distribution fit if available
    if not types:
        sns.barplot(data=subset, x='DelayInterval', y='Percent', ax=ax)
    else:
        if category:
            result = types.get(correlation=corr, category=category)
        else:
            result = types.get(correlation=corr)
        plot_mixture(data=subset, result=result, cumulative=cumulative, 
                     intervals=intervals, interval_labels=interval_labels, ax=ax)


    title = f'{corr}'
    if category is not None:
        title += f', {category}'

    ax.set_title(f'{title} N={n_sample:,}')
    ax.set_xlabel('Delay Interval')
    ax.set_ylabel('Percent')
    ax.tick_params(axis='x', rotation=45)
    if isinstance(types, TypeList) and types.types:
        ax.legend()


def plot(
    number_of_plots: int, 
    result: pd.DataFrame, 
    df: pd.DataFrame,
    correlations: list,
    dataset_info: dict,
    intervals: list[int],
    interval_labels: list[str],
    process_by_category: bool = True,
    types: TypeList | None = None,
    cumulative: bool = True,
    fn: str|Path|None = None,
    max_lookback_length:int=4,
) -> None:
    """Plot all correlation delay distributions.
    
    Parameters
    ----------
    number_of_plots : int
        Total number of subplots needed
    result : pd.DataFrame
        Aggregated results with DelayInterval and Count
    df : pd.DataFrame
        Original dataset
    correlations : list
        Ordered list of correlations to plot
    dataset_info : dict
        Dataset statistics for the title
    process_by_category : bool
        Whether to process by category
    types : TypeList, optional
        Dictionary of fitted distribution types
    cumulative : bool
        Whether to show cumulative percentage
    """

    categories = dataset_info['categories']

    correlation_batches = get_correlation_batches(correlations, process_by_category)
    batch_nr = 0

    for correlations in correlation_batches:
        j=0
        rows = len(correlations) + 1
        if not process_by_category:
            rows = math.ceil(rows / 2)
        columns = 2
        fig, axs = plt.subplots(rows, columns, figsize=(20, rows * 6))

        # Add overall figure title
        fig.suptitle(
            f'Correlation Delay Analysis\n'
            f'{dataset_info["n_anomalies"]:,} anomalies from '
            f'{dataset_info["n_farms"]:,} farms and '
            f'{dataset_info["n_sheds"]:,} sheds',
            fontsize=16, fontweight='bold'
        )

        axs = axs.flatten() if rows > 1 else axs

        j=0
        for corr in correlations:
            if process_by_category:
                subset = result[
                    (result['Correlation'] == corr) & 
                    (result['Category'] == categories[0])
                ].copy()
                n_sample = len(df[
                    (df['Correlation'] == corr) & 
                    (df['Category'] == categories[0])
                ])
                plot_subset(ax=axs[j], subset=subset, n_sample=n_sample, corr=corr, 
                            types=types, category=categories[0], max_lookback_length=max_lookback_length,
                        cumulative=cumulative, intervals=intervals, interval_labels=interval_labels)
                j += 1
                
                subset = result[
                    (result['Correlation'] == corr) & 
                    (result['Category'] == categories[1])
                ].copy()
                n_sample = len(df[
                    (df['Correlation'] == corr) & 
                    (df['Category'] == categories[1])
                ])
                plot_subset(ax=axs[j], subset=subset, n_sample=n_sample, corr=corr, types=types, category=categories[1], 
                            cumulative=cumulative, intervals=intervals, interval_labels=interval_labels)
                j += 1
            else:
                n_sample = len(df[df['Correlation'] == corr])
                subset = result[result['Correlation'] == corr].copy()
                plot_subset(ax=axs[j], subset=subset, n_sample=n_sample, corr=corr, types=types, cumulative=cumulative,
                        intervals=intervals, interval_labels=interval_labels)
                j += 1
        
        # Remove any remaining empty subplots
        for k in range(j, len(axs)):
            axs[k].axis('off')

        if fn:
            filename = Path(fn)
            filename = filename.parent / f"{filename.stem}_part{batch_nr+1}{filename.suffix}"
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            batch_nr += 1

def get_correlation_batches(
    correlations: list,
    process_by_category: bool,
    max_correlations_per_batch: int = 10
) -> list[list]:
    """Split correlations into batches for plotting.
    
    Parameters
    ----------
    correlations : list or np.ndarrayList of correlation names
    process_by_category : bool Whether processing is by category
    max_correlations_per_batch : int Maximum number of correlations per batch (default: 10)
    """
    def split_after_n(sentence:str, n:int=1) -> tuple[str, str]:
        """Split list into chunks of size n."""
        words = sentence.split()
        first_two = " ".join(words[:n])
        rest = " ".join(words[n:])
        return first_two, rest


    df = pd.DataFrame({'correlations': correlations}, index=pd.RangeIndex(len(correlations)))
    df['first_words'] = [split_after_n(corr)[0] for corr in correlations]
    grouped = df.groupby('first_words').size()
    batches = []
    for group in grouped.index:
        batch = df[df['first_words'] == group]['correlations'].tolist()
        batches.append(batch)
    return batches

def print_results(result: Result):
    """Print results nicely.
    
    Parameters
    ----------
    result : Result
        Fitted mixture model result
    """

    logger.info(f"MIXTURE: {result.lambda_:.1%} {result.dist1} + {1-result.lambda_:.1%} {result.dist2}")
    logger.info("="*60)
    logger.info(f"λ (mixture weight):     {result.lambda_:.6f}")
    logger.info(f"params1 ({result.dist1}):        {result.params1}")
    logger.info(f"params2 ({result.dist2}):        {result.params2}")
    logger.info(f"Negative LL:            {result.nll:.2f}")
    logger.info(f"AIC:                    {result.aic:.2f}")
    logger.info(f"BIC:                    {result.bic:.2f}")
    logger.info(f"N samples:              {result.n}")