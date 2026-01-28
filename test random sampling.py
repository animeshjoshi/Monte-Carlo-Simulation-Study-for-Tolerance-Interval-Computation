import pandas as pd
import numpy as np
import tolerance_intervals as Tolerance_Intervals

results_dict = {}


def test_normal_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.normal(mean, std, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
 

    for x in range(trials):

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)

    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)

    print("Smooth KDE Coverage")
    print(smooth_bootstrap_kde_counter/trials)


test_normal_distribution_coverage(0,1, 100, 100, 10000, 0.95,0.95)










