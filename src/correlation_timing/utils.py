"""Utility functions for mixture distribution fitting and plotting."""

import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from scipy.stats import expon, weibull_min, lognorm
from scipy.optimize import minimize
from pydantic import BaseModel, Field, ConfigDict

from iconfig.iconfig import iConfig

from loguru import logger

from concurrent.futures import ProcessPoolExecutor

class Result(BaseModel):
    """Results from mixture distribution fitting."""
    
    lambda_: float = Field(..., description="Mixture weight parameter")
    params1: list[float] = Field(..., description="Parameters for first distribution")
    params2: list[float] = Field(..., description="Parameters for second distribution")
    dist1: str = Field(..., description="Type of first distribution")
    dist2: str = Field(..., description="Type of second distribution")
    nll: float = Field(..., description="Negative log-likelihood of the fit")
    aic: float = Field(..., description="Akaike Information Criterion")
    bic: float = Field(..., description="Bayesian Information Criterion")
    n: int = Field(..., description="Number of data points")

    model_config = ConfigDict(arbitrary_types_allowed=True)

class TypesCategory(BaseModel):
    correlation: str = Field(..., description="Correlation name")
    category: str = Field(..., description="Category name")
    result: Result | None = Field(None, description="Fitted result or None if no data")

class Types(BaseModel):
    correlation: str = Field(..., description="Correlation name")
    results: dict[str, Result | None] = Field(..., description="Mapping of category to Result or None if no data")

class TypeList(BaseModel):
    types: list[TypesCategory|Types] = Field(..., description="List of TypesCategory instances")

    def get(self, correlation:str, category:str|None=None) -> Result | None:
        """Retrieve the Result for a given correlation and optional category."""
        for item in self.types:
            if isinstance(item, Types):
                if item.correlation == correlation and category is None:
                    return item.result
            elif isinstance(item, TypesCategory):
                if item.correlation == correlation and item.category == category:
                    return item.result
        return None


def fit_mixture_simple(data, dist1='expon', dist2='weibull_min') -> Result:
    """Fit a two-component mixture distribution.
    
    Parameters
    ----------
    data : array
        Your observations (e.g., time delays)
    dist1 : str
        Component 1: 'expon', 'weibull_min', or 'lognorm'
    dist2 : str
        Component 2: 'expon', 'weibull_min', or 'lognorm'
    
    Returns
    -------
    Result
        Fitted parameters and statistics
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    dists = {
        'expon': expon,
        'weibull_min': weibull_min,
        'lognorm': lognorm,
    }
    
    D1 = dists[dist1]
    D2 = dists[dist2]
    
    # Fit individual distributions for starting guesses
    p1 = D1.fit(data)
    p2 = D2.fit(data)
    
    # Objective function: negative log-likelihood
    def objective(params):
        lam = params[0]
        
        # Ensure lambda is valid
        if lam <= 0 or lam >= 1:
            return 1e10
        
        # Extract parameters for each distribution
        p1_est = params[1:len(p1)+1]
        p2_est = params[len(p1)+1:]
        
        try:
            # Evaluate mixture density
            pdf1 = D1.pdf(data, *p1_est)
            pdf2 = D2.pdf(data, *p2_est)
            
            # Clamp to avoid log(0)
            mix = np.clip(lam * pdf1 + (1-lam) * pdf2, 1e-10, None)
            
            return -np.sum(np.log(mix))
        except Exception:
            return 1e10
    
    # Initial guess
    x0 = [0.5] + list(p1) + list(p2)
    
    # Optimize
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 5000})
    
    # Extract results
    lam = np.clip(res.x[0], 0.01, 0.99)
    p1_fit = res.x[1:len(p1)+1]
    p2_fit = res.x[len(p1)+1:]
    
    # Calculate statistics
    n_params = len(res.x)
    nll = res.fun
    aic = 2*nll + 2*n_params
    bic = 2*nll + n_params*np.log(len(data))
    
    return Result(
        lambda_=lam,
        params1=p1_fit,
        params2=p2_fit,
        dist1=dist1,
        dist2=dist2,
        nll=nll,
        aic=aic,
        bic=bic,
        n=len(data),
    )


def plot_mixture(data, result: Result, colors, max_lookback_length:int=4, ax: Axes | None = None) -> Axes | Figure:
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
    
    # Histogram
    sns.barplot(data=data, x='DelayInterval', y='Percent', ax=ax, palette=colors)
    
    
    # Fitted curves
    x = np.linspace(0, max_lookback_length*60, 300)
    
    pdf1 = D1.pdf(x, *result.params1)
    pdf2 = D2.pdf(x, *result.params2)
    
    w1 = result.lambda_
    w2 = 1 - result.lambda_
    
    ax.plot(x, w1*pdf1, 'r-', lw=2.5, label=f'{result.dist1} ({w1:.1%})')
    ax.plot(x, w2*pdf2, 'b-', lw=2.5, label=f'{result.dist2} ({w2:.1%})')
    ax.plot(x, w1*pdf1 + w2*pdf2, 'k--', lw=3, label='Mixture')
    
    ax.set_xlabel('Time Delay')
    ax.set_ylabel('Density')
    ax.set_title(f'{w1:.1%} {result.dist1} + {w2:.1%} {result.dist2} (BIC={result.bic:.1f})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if not ax:
        return fig
    else:
        return ax


def print_results(result: Result):
    """Print results nicely.
    
    Parameters
    ----------
    result : Result
        Fitted mixture model result
    """
    logger.info("\n" + "="*60)
    logger.info(f"MIXTURE: {result.lambda_:.1%} {result.dist1} + {1-result.lambda_:.1%} {result.dist2}")
    logger.info("="*60)
    logger.info(f"λ (mixture weight):     {result.lambda_:.6f}")
    logger.info(f"params1 ({result.dist1}):        {result.params1}")
    logger.info(f"params2 ({result.dist2}):        {result.params2}")
    logger.info(f"Negative LL:            {result.nll:.2f}")
    logger.info(f"AIC:                    {result.aic:.2f}")
    logger.info(f"BIC:                    {result.bic:.2f}")
    logger.info(f"N samples:              {result.n}")
    logger.info("="*60 + "\n")


def find_best_fit(delays: pd.Series) -> Result:
    """Find the best mixture model fit by comparing BIC across models.
    
    Parameters
    ----------
    delays : pd.Series
        Time delay data to fit
        
    Returns
    -------
    Result
        Best fitting mixture model result
    """
    best_bic = float('inf')
    best_result = None
    models = [
        ('expon', 'weibull_min'),
        ('expon', 'lognorm'),
        ('weibull_min', 'lognorm'),
    ]
    for dist1, dist2 in models:
        result = fit_mixture_simple(delays, dist1=dist1, dist2=dist2)
        if result.bic < best_bic:
            best_bic = result.bic
            best_result = result
    return best_result

def load_types() -> TypeList|None:
    """Load previously saved types from disk.
    
    Parameters
    ----------
    config : iConfig
        Configuration object
    correlations : list
        List of correlation names
        
    Returns
    -------
    dict[str, Result]
        Dictionary mapping correlation name to Result
    """

    if not (types_dir := Path(os.getenv('INPUTS'))).exists():
        return None
    
    with open(types_dir / "types.json") as f:
        types = TypeList.model_validate_json(f.read())
    return types

def save_types(types: TypeList) -> bool:
    """Save fitted types to disk.
    
    Parameters
    ----------
    config : iConfig
        Configuration object
    correlations : list
        List of correlation names
        
    Returns
    -------
    dict[str, Result]
        Dictionary mapping correlation name to Result
    """
    if not (types_dir := Path(os.getenv('INPUTS'))).exists():
        types_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        types.dump_json((types_dir / "types.json"))
        return True
    except Exception:
        logger.error("Failed to save types:")
        return False

def _process_correlation_wrapper(args):
    """Wrapper for unpacking args in map."""
    return process_single_correlation(*args)


def _process_correlation_category_wrapper(args):
    """Wrapper for unpacking args in map."""
    return process_single_correlation_category(*args)


def process_single_correlation(correlation, corr_data):
    """Process a single correlation."""
    delays = corr_data[corr_data['Correlation'] == correlation]['Delay']
    result = find_best_fit(delays)
    return correlation, result


def determine_type(
    correlations: list, 
    corr_data: pd.DataFrame,
    max_workers: int = 10,
    save: bool = True
) -> dict[str, Result]:
    """Determine best distribution type for each correlation.
    
    Parameters
    ----------
    correlations : list
        List of correlation names
    corr_data : pd.DataFrame
        DataFrame with Correlation and Delay columns
    max_workers : int, optional
        Maximum number of worker processes. If None, uses CPU count.
        
    Returns
    -------
    dict[str, Result]
        Dictionary mapping correlation name to best fit result
    """
    ret = None
    if save:
        ret = load_types()
    
    if ret:
        return ret
    
    tasks = [(corr, corr_data) for corr in correlations]
    ret = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_correlation_wrapper, tasks)
    
    for correlation, result in results:
        ret.append(Types(correlation=correlation, result=result))
    
    return ret


def process_single_correlation_category(correlation, corr_data, category):
    """Process a single correlation-category combination."""
    delays = corr_data[
        (corr_data['Correlation'] == correlation) & 
        (corr_data['Category'] == category)
    ]['Delay']
    
    if delays.empty:
        return correlation, category, None
    else:
        result = find_best_fit(delays)
        return correlation, category, result

def determine_type_by_category(
    correlations: list, 
    corr_data: pd.DataFrame,
    max_workers: int = 10,
    save: bool = True
) -> TypeList|None:
    """Parallel version using ProcessPoolExecutor.
    
    Parameters
    ----------
    correlations : list
        List of correlation names
    corr_data : pd.DataFrame
        DataFrame with Correlation, Category, and Delay columns
    max_workers : int, optional
        Maximum number of worker processes. If None, uses CPU count.
        
    Returns
    -------
    dict[str, dict[str, Result]]
        Nested dictionary: correlation -> category -> Result
    """
    ret = None
    if save:
        ret = load_types()
    
    if ret:
        return ret

    categories = corr_data['Category'].unique()
    assert len(categories) == 2
    
    # Create all tasks
    tasks = [
        (corr, corr_data, cat) 
        for corr in correlations 
        for cat in categories
    ]
    
    ret = {corr: {} for corr in correlations}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_correlation_category_wrapper, tasks)
    
    # Organize results
    for correlation, category, result in results:
        ret.append(TypesCategory(correlation=correlation, category=category, result=result))
    
    tpl = TypeList(types=ret)

    if save:
        save_types(tpl)

    return tpl

def plot_subset(
    ax: Axes, 
    subset: pd.DataFrame, 
    n_sample: int, 
    corr: str,
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
    
    # Calculate percentage and cumulative percentage
    subset['Percent'] = 100 * subset['Count'] / subset['Count'].sum()
    subset = subset.sort_values('DelayInterval')
    subset['CumPercent'] = subset['Percent'].cumsum()
    
    # Assign colors: blue for first 90%, orange for rest
    colors = ['#1f77b4' if cum <= 90 else '#ff7f0e' for cum in subset['CumPercent']]

    # Plot distribution fit if available
    if not types:
        sns.barplot(data=subset, x='DelayInterval', y='Percent', ax=ax, palette=colors)
    else:
        if category:
            result = types.get(correlation=corr, category=category)
        else:
            result = types.get(correlation=corr)
        plot_mixture(data=subset, result=result, max_lookback_length=max_lookback_length, colors=colors, ax=ax)

    if cumulative:
        # Add cumulative line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(range(len(subset)), subset['CumPercent'].values, 
                 color='darkred', linestyle='-', linewidth=2, 
                 label='Cumulative %')  # type:ignore
        ax2.axhline(y=90, color='darkred', linestyle='--', alpha=0.5, 
                    label='90% threshold')
        ax2.set_ylabel('Cumulative %', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.set_ylim(0, 105)

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
    config: iConfig,
    number_of_plots: int, 
    result: pd.DataFrame, 
    df: pd.DataFrame,
    correlations: list,
    dataset_info: dict,
    process_by_category: bool = True,
    types: TypeList | None = None,
    cumulative: bool = True,
    fn: str|Path|None = None,
    max_lookback_length:int=4
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
    rows = number_of_plots + 1
    rows = math.ceil(rows / 2)
    columns = 2
    fig, axs = plt.subplots(rows, columns, figsize=(20, rows * 4 + 20))

    # Add overall figure title
    fig.suptitle(
        f'Correlation Delay Analysis\n'
        f'{dataset_info["n_anomalies"]:,} anomalies from '
        f'{dataset_info["n_farms"]:,} farms and '
        f'{dataset_info["n_sheds"]:,} sheds',
        fontsize=16, fontweight='bold', y=1.01
    )

    axs = axs.flatten() if rows > 1 else [axs]
    categories = dataset_info['categories']
    
    j = 0
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
            plot_subset(ax=axs[j], subset=subset, n_sample=n_sample, 
                       corr=corr, types=types, category=categories[0],
                       max_lookback_length=max_lookback_length,
                       cumulative=cumulative)
            j += 1
            
            subset = result[
                (result['Correlation'] == corr) & 
                (result['Category'] == categories[1])
            ].copy()
            n_sample = len(df[
                (df['Correlation'] == corr) & 
                (df['Category'] == categories[1])
            ])
            plot_subset(ax=axs[j], subset=subset, n_sample=n_sample, 
                       corr=corr, types=types, category=categories[1], 
                        max_lookback_length=max_lookback_length,
                       cumulative=cumulative)
            j += 1
        else:
            n_sample = len(df[df['Correlation'] == corr])
            subset = result[result['Correlation'] == corr].copy()
            plot_subset(ax=axs[j], subset=subset, n_sample=n_sample, 
                       corr=corr, types=types, cumulative=cumulative)
            j += 1
    
    # Remove any remaining empty subplots
    for k in range(j, len(axs)):
        axs[k].axis('off')

    if fn:
        fn = Path(fn)
        if not fn.parent.exists():
            fn.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fn, bbox_inches='tight', dpi=300)
    else:
        plt.tight_layout()
        plt.show()