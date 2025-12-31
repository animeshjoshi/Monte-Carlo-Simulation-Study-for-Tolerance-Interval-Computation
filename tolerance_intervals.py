# Module Used to Compute Tolerance Intervals based on 4 methods
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import binom
import openturns as ot

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

  
    v = n - binom.ppf(confidence, n, proportion)

    L = int(np.floor(v/2))
    U = int(np.ceil(n+1-v/2))
    data.sort()

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

    lower = np.percentile(lower_bounds, 100*((1-confidence)/2))
    upper = np.percentile(upper_bounds, 100*((1+confidence)/2))

    return (lower,upper)


def compute_bootstrap_kde_ti(data, proportion, confidence):

    np.random.seed(15)

    lower_bounds = []
    upper_bounds = []

    for x in range(10000):

        sample = np.random.choice(data,size=len(data), replace = True)
        sample = np.asarray(sample).reshape(-1, 1)
        sample = ot.Sample(sample)
       

        factory = ot.KernelSmoothing(ot.Normal())

        # Optional: shrink plugin bandwidth
        h = factory.computePluginBandwidth(sample)
        #factory.setBandwidth(h)  # v = shrinkage factor from your code

        # Build KDE distribution
        distribution = factory.build(sample)



        lower_bounds.append(distribution.computeQuantile(0.025)[0])
        upper_bounds.append(distribution.computeQuantile(0.975)[0])

    lower = np.percentile(lower_bounds, 100*((1-confidence)/2))
    upper = np.percentile(upper_bounds, 100*((1+confidence)/2))

    return (lower,upper)


def find_sheather_bandwith(sample):
    
    sample = np.asarray(sample).reshape(-1, 1)
    sample = ot.Sample(sample)
    factory = ot.KernelSmoothing(ot.Normal())
    h = factory.computePluginBandwidth(sample)[0]
    return h

def compute_bootstrap_kde_shrunk_smooth(data, proportion, confidence):

    np.random.seed(15)

    lower_bounds = []
    upper_bounds = []

    ybar = np.mean(data)

    for x in range(10000):

        sample = np.random.choice(data,size=len(data), replace = True)
        h = find_sheather_bandwith(sample)
        ystarbar = np.mean(sample)
        n = len(data)

        
        second_moment = 1

        denom = np.sum((data-ybar)**2)
        v = (1 + (n * h**2 * second_moment) / denom) ** (-0.5)

        delta = np.random.normal(size=n) 
        
        new_sample = (1 - v) * ystarbar + v * sample + h * v * delta





        sample = np.asarray(sample).reshape(-1, 1)
        sample = ot.Sample(sample)
        factory = ot.KernelSmoothing(ot.Normal())

        # Optional: shrink plugin bandwidth
        h = factory.computePluginBandwidth(sample)
      #  factory.setBandwidth(h)  # v = shrinkage factor from your code

        # Build KDE distribution
        distribution = factory.build(sample)



        lower_bounds.append(distribution.computeQuantile(0.025)[0])
        upper_bounds.append(distribution.computeQuantile(0.975)[0])

    lower = np.percentile(lower_bounds, 100*((1-confidence)/2))
    upper = np.percentile(upper_bounds, 100*((1+confidence)/2))

    return (lower,upper)













