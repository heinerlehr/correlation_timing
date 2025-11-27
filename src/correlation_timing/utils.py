"""Utility functions for mixture distribution fitting and plotting."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import expon, weibull_min, lognorm
from scipy.optimize import minimize
from pydantic import BaseModel, Field, ConfigDict

from loguru import logger

from concurrent.futures import ProcessPoolExecutor

class Result(BaseModel):
    """Results from mixture distribution fitting."""
    
    lambda_: float = Field(..., description="Mixture weight parameter")
    params1: list[float] = Field(..., description="Parameters for first distribution")
    params2: list[float] = Field(..., description="Parameters for second distribution")
    dist1: str = Field(..., description="Type of first distribution")
    dist2: str = Field(..., description="Type of second distribution")
    nll: float | None = Field(None, description="Negative log-likelihood of the fit")
    aic: float | None = Field(None, description="Akaike Information Criterion")
    bic: float | None = Field(None, description="Bayesian Information Criterion")
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
    if dist1 == 'expon':
        p1 = D1.fit(data, floc=0)
    else:
        p1 = D1.fit(data)
    if dist2 == 'expon':
        p2 = D2.fit(data, floc=0)
    else:
        p2 = D2.fit(data)

    # Parameter bounds
    bounds = []
    # lambda bounds
    bounds.append((1e-3, 1-1e-3))
    # dist1 parameter bounds
    for i, val in enumerate(p1):
        # location parameter (usually 2nd param) can be any, scale/shape > 0
        if i == 0 and dist1 == 'expon':
            bounds.append((0, 0))  # force loc=0 for expon
        elif i == 1 and dist1 == 'expon':
            bounds.append((1e-8, None))  # scale > 0
        elif i == 0 and dist1 in ('weibull_min', 'lognorm'):
            bounds.append((1e-8, None))  # shape > 0
        elif i == 1:
            bounds.append((None, None))  # loc
        elif i == 2:
            bounds.append((1e-8, None))  # scale > 0
        else:
            bounds.append((None, None))
    # dist2 parameter bounds
    for i, val in enumerate(p2):
        if i == 0 and dist2 == 'expon':
            bounds.append((0, 0))
        elif i == 1 and dist2 == 'expon':
            bounds.append((1e-8, None))
        elif i == 0 and dist2 in ('weibull_min', 'lognorm'):
            bounds.append((1e-8, None))
        elif i == 1:
            bounds.append((None, None))
        elif i == 2:
            bounds.append((1e-8, None))
        else:
            bounds.append((None, None))

    # Objective function: negative log-likelihood
    def objective(params):
        lam = params[0]
        p1_est = params[1:len(p1)+1]
        p2_est = params[len(p1)+1:]
        try:
            pdf1 = D1.pdf(data, *p1_est)
            pdf2 = D2.pdf(data, *p2_est)
            mix = np.clip(lam * pdf1 + (1-lam) * pdf2, 1e-10, None)
            return -np.sum(np.log(mix))
        except Exception:
            return 1e10

    # Multiple random restarts
    best_res = None
    best_nll = np.inf
    n_restarts = 8
    rng = np.random.default_rng()
    all_nlls = []
    for _ in range(n_restarts):
        # Randomize lambda
        lam0 = rng.uniform(0.05, 0.95)
        # Randomize parameters around fitted values
        p1_0 = []
        for i, val in enumerate(p1):
            if isinstance(val, (float, int)):
                spread = abs(val) if abs(val) > 1e-6 else 1.0
                p1_0.append(val + rng.normal(0, 0.2*spread))
            else:
                p1_0.append(val)
        p2_0 = []
        for i, val in enumerate(p2):
            if isinstance(val, (float, int)):
                spread = abs(val) if abs(val) > 1e-6 else 1.0
                p2_0.append(val + rng.normal(0, 0.2*spread))
            else:
                p2_0.append(val)
        x0 = [lam0] + list(p1_0) + list(p2_0)
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
            nll = res.fun
            all_nlls.append(nll)
            if nll < best_nll:
                best_nll = nll
                best_res = res
        except Exception:
            continue

    if best_res is None:
        raise RuntimeError("All mixture fits failed.")

    lam = np.clip(best_res.x[0], 0.01, 0.99)
    p1_fit = best_res.x[1:len(p1)+1]
    p2_fit = best_res.x[len(p1)+1:]

    n_params = len(best_res.x)
    nll = best_res.fun
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

def load_types(correlations: list, fn:str) -> TypeList|None:
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
    
    if not (types_dir / fn).exists():
        return None
    
    with open(types_dir / fn) as f:
        types = TypeList.model_validate_json(f.read())

    if all(
        any(
            (isinstance(item, Types) and item.correlation == corr) or
            (isinstance(item, TypesCategory) and item.correlation == corr)
            for item in types.types
        )
        for corr in correlations
    ):
        return types
    return None

def save_types(types: TypeList, fn:str="types.json") -> bool:
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
        with open(types_dir / fn, 'w') as f:
            f.write(types.model_dump_json(indent=2))
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
        ret = load_types(correlations=correlations, fn="types.json")
    
    if ret:
        logger.info("Loaded previously saved fits")
        return ret
    
    tasks = [(corr, corr_data) for corr in correlations]
    ret = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_correlation_wrapper, tasks)
    
    for correlation, result in results:
        ret.append(Types(correlation=correlation, result=result))
   
    tpl = TypeList(types=ret)

    if save:
        save_types(tpl, fn="types.json")
        logger.info("Saved fits")

    return tpl


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
        ret = load_types(correlations=correlations, fn="types_by_category.json")
    
    if ret:
        logger.info("Loaded previously saved fits")
        return ret

    categories = corr_data['Category'].unique()
    assert len(categories) == 2
    
    # Create all tasks
    tasks = [
        (corr, corr_data, cat) 
        for corr in correlations 
        for cat in categories
    ]
    
    ret = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_process_correlation_category_wrapper, tasks)
    
    # Organize results
    for correlation, category, result in results:
        ret.append(TypesCategory(correlation=correlation, category=category, result=result))
    
    tpl = TypeList(types=ret)

    if save:
        save_types(tpl, fn="types_by_category.json")
        logger.info("Saved fits")

    return tpl


