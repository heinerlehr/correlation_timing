import numpy as np
import pytest
from scipy.stats import expon, weibull_min, lognorm
from src.correlation_timing.utils import fit_mixture_simple, Result


def test_fit_mixture_simple_expon():
    # Generate exponential data
    np.random.seed(42)
    data = np.random.exponential(scale=10, size=1000)
    result = fit_mixture_simple(data, dist1='expon', dist2='expon')
    assert isinstance(result, Result)
    # Check lambda is between 0 and 1
    assert 0 < result.lambda_ < 1
    # Check params1 and params2 are reasonable
    assert result.params1[1] > 0
    assert result.params2[1] > 0
    # Check nll, aic, bic are not None
    assert result.nll is not None
    assert result.aic is not None
    assert result.bic is not None
    # Check n matches data
    assert result.n == len(data)


def test_fit_mixture_simple_weibull():
    np.random.seed(0)
    data = weibull_min.rvs(1.5, loc=0, scale=5, size=1000)
    result = fit_mixture_simple(data, dist1='weibull_min', dist2='expon')
    assert isinstance(result, Result)
    assert 0 < result.lambda_ < 1
    assert result.params1[1] > 0
    assert result.params2[1] > 0
    assert result.n == len(data)


def test_fit_mixture_simple_lognorm():
    np.random.seed(1)
    data = lognorm.rvs(0.5, loc=0, scale=2, size=1000)
    result = fit_mixture_simple(data, dist1='lognorm', dist2='weibull_min')
    assert isinstance(result, Result)
    assert 0 < result.lambda_ < 1
    assert result.params1[1] > 0
    assert result.params2[1] > 0
    assert result.n == len(data)


def test_fit_mixture_simple_nan_handling():
    np.random.seed(2)
    data = np.random.exponential(scale=5, size=1000)
    data[::10] = np.nan  # introduce NaNs
    result = fit_mixture_simple(data, dist1='expon', dist2='weibull_min')
    assert isinstance(result, Result)
    assert result.n == np.sum(~np.isnan(data))


def test_fit_mixture_simple_invalid():
    # All NaN input
    data = np.full(100, np.nan)
    with pytest.raises(Exception):
        fit_mixture_simple(data, dist1='expon', dist2='weibull_min')


def test_fit_mixture_simple_small_sample():
    # Very small sample
    data = np.array([1.0, 2.0, 3.0])
    result = fit_mixture_simple(data, dist1='expon', dist2='weibull_min')
    assert isinstance(result, Result)
    assert result.n == 3


def test_fit_mixture_simple_dist_choices():
    # Try all combinations
    data = np.random.exponential(scale=3, size=200)
    for d1 in ['expon', 'weibull_min', 'lognorm']:
        for d2 in ['expon', 'weibull_min', 'lognorm']:
            result = fit_mixture_simple(data, dist1=d1, dist2=d2)
            assert isinstance(result, Result)
            assert result.n == len(data)


def test_fit_mixture_simple_exp_weibull_recovery():
    # Parameters for the mixture
    np.random.seed(123)
    lam = 0.7
    expon_scale = 5.0
    weib_shape = 2.0
    weib_scale = 10.0
    n = 2000
    # Generate mixture data
    n_exp = int(n * lam)
    n_weib = n - n_exp
    data_exp = np.random.exponential(scale=expon_scale, size=n_exp)
    data_weib = weibull_min.rvs(weib_shape, loc=0, scale=weib_scale, size=n_weib)
    data = np.concatenate([data_exp, data_weib])
    np.random.shuffle(data)
    # Fit mixture
    result = fit_mixture_simple(data, dist1='expon', dist2='weibull_min')
    assert isinstance(result, Result)
    # Check lambda is close to true value
    assert abs(result.lambda_ - lam) < 0.25  # allow some tolerance
    # Check that one component is close to expon, the other to weibull
    # Exponential scale is params1[1] or params2[1] depending on which is expon
    if result.dist1 == 'expon':
        exp_scale_fit = result.params1[1]
        weib_shape_fit = result.params2[0]
        weib_scale_fit = result.params2[2]
    else:
        exp_scale_fit = result.params2[1]
        weib_shape_fit = result.params1[0]
        weib_scale_fit = result.params1[2]
    assert abs(exp_scale_fit - expon_scale) < 2.0
    assert abs(weib_shape_fit - weib_shape) < 1.0
    assert abs(weib_scale_fit - weib_scale) < 3.0
