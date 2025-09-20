# Module Used to Compute Tolerance Intervals based on 4 methods
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import binom


def compute_normal_ti(data, proportion, confidence):

    # Normal Tolerance Interval estimated with Gunthers Approximation

    x_bar = np.mean(data)
    n = len(data)
    s = np.std(data, ddof=1) # Sample standard deviation
    z = norm.ppf((1+proportion)/2) # Normal CDF
    k_num = (n-1)*(1+1/n)
    k_denom = chi2.ppf(confidence, n-1) # CHI Square CDF
    k_orig = (k_num/k_denom)**0.5
    k_orig = z * k_orig

    # Gunther Correction

    w = 1 + (n-3-chi2.ppf(confidence,n-1))/(2*(n+1)**2)
    w = w ** 0.5


    final_k = k_orig * w

    lower_bound = x_bar - (s*final_k)
    upper_bound = x_bar + (s*final_k)

    return (lower_bound, upper_bound)


def compute_nonparametric_ti(data, proportion, confidence):

    x_bar = np.mean(data)
    n = len(data)
    s = np.std(data, ddof=1) # Sample standard deviation

    v = n - binom.ppf(n,proportion, confidence)
    L = int(np.floor(v/2))
    U = int(np.ceil(n+1-v/2))

    upper_bound = data[U-1]
    lower_bound = data[L-1]

    return (lower_bound, upper_bound)


def compute_bootstrap_ti(data, proportion, confidence):

    np.random.seed(15)

    lower_bounds = []
    upper_bounds = []

    for x in range(10000):

        sample = np.random.choice(data,size=len(data), replace = True)
        lower_bounds.append(compute_nonparametric_ti(sample,proportion,confidence)[0])
        upper_bounds.append(compute_nonparametric_ti(sample,proportion,confidence)[1])

    lower = np.percentile(lower_bounds, (1-confidence)/2)
    upper = np.percentile(upper_bounds, confidence/2)

    return (lower,upper)








