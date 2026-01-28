import pandas as pd
import numpy as np
import tolerance_intervals as Tolerance_Intervals

results_dict = {}
df = pd.DataFrame()



def test_normal_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.normal(mean, std, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    print('Normal Distribution')
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []

    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []
    approx_normal_widths = []
    approx_normal_counter = 0


    for x in range(trials):

  
        print(x)
        sample = rng.choice(population, sample_size, replace=True)
       
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])
       



        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_widths.append(normal_ti[1] - normal_ti[0])
        normal_coverages.append(empirical_coverage)

        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])



        from scipy.stats import shapiro
        print(shapiro(sample))

        print(empirical_coverage)

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])


        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])
        print('Normal Trial')

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Normal Distribution Normal TI Coverage'] = normal_counter/trials
    df['Normal Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Normal Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['Normal Distribution Bootstrap TI Coverages'] = bootstrap_coverages

    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Normal Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Normal Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Normal Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['Normal Distribution KDE Bootstrap TI Coverages'] = kde_coverages

    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Normal Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['Normal Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages

    df['Normal Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Normal Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths


    df['Normal Distribution Normal TI Widths'] = normal_widths
    df['Normal Distribution KDE TI Widths'] = kde_widths
    df['Normal Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Normal Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Normal Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths



test_normal_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)






def test_chi2_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.chisquare(10, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []

    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []

    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0
 

    for x in range(trials):

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])
        print(x)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        print('chi 2 trial')
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])


        empirical_coverage = np.sum(within_bounds)/len(population)

        normal_coverages.append(empirical_coverage)
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)


        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Chi-Square Distribution Normal TI Coverage'] = normal_counter/trials
    df['Chi-Square Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Chi-Square Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials    
    df['Chi-Square Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Chi-Square Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Chi-Square Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Chi-Square Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['Chi-Square Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Chi-Square Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['Chi-Square Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Chi-Square Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths

    df['Chi-Square Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages
    df['Chi-Square Distribution Normal TI Widths'] = normal_widths
    df['Chi-Square Distribution KDE TI Widths'] = kde_widths
    df['Chi-Square Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Chi-Square Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Chi-Square Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths
test_chi2_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)

def test_f_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.f(4,30, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0

    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []
 

    for x in range(trials):

  
        print(x)
        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])
        print('F Trial')

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['F Distribution Normal TI Coverage'] = normal_counter/trials
    df['F Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['F Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['F Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['F Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['F Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['F Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['F Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['F Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials

    df['F Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['F Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths
    df['F Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages
    df['F Distribution Normal TI Widths'] = normal_widths
    
    df['F Distribution KDE TI Widths'] = kde_widths
    df['F Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['F Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['F Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths

test_f_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)


def test_lognormal_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):

    rng = np.random.default_rng()
    population = rng.lognormal(mean, std, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0

    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []

 

    for x in range(trials):

  
        print(x)
        print('Log Normal Trial')
        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Log Normal Distribution Normal TI Coverage'] = normal_counter/trials
    df['Log Normal Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Log Normal Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['Log Normal Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Log Normal Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Log Normal Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Log Normal Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials  
    df['Log Normal Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Log Normal Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['Log Normal Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Approx Normal Distribution Normal TI Widths'] = approx_normal_widths
    df['Log Normal Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages
    df['Log Normal Distribution Normal TI Widths'] = normal_widths
    df['Log Normal Distribution KDE TI Widths'] = kde_widths
    df['Log Normal Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Log Normal Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Log Normal Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths
test_lognormal_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)













def test_t_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.standard_t(10, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    nonparametric_coverages = []
    kde_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0
    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []

    for x in range(trials):
        print('T Trial')

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])
        print(x)

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['T Distribution Normal TI Coverage'] = normal_counter/trials
    df['T Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['T Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['T Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['T Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['T Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['T Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['T Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['T Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['T Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['T Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths
    df['T Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages

    df['T Distribution Normal TI Widths'] = normal_widths
    df['T Distribution KDE TI Widths'] = kde_widths
    df['T Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['T Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['T Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths

test_t_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)








def test_gamma_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.gamma(shape = 0.5,scale=2, size = pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []
 

    for x in range(trials):

        print(x)
        print('Gamma Trial')

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Gamma Distribution Normal TI Coverage'] = normal_counter/trials
    df['Gamma Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Gamma Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['Gamma Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Gamma Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Gamma Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Gamma Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['Gamma Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Gamma Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['Gamma Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Gamma Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths
    df['Gamma Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages

    df['Gamma Distribution Normal TI Widths'] = normal_widths
    df['Gamma Distribution KDE TI Widths'] = kde_widths
    df['Gamma Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Gamma Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Gamma Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths
test_gamma_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)


def test_beta_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.beta(4,1, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    nonparametric_coverages = []
    kde_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0
    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []

    for x in range(trials):

        print(x)

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        print('Exponential Trial')
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Beta Distribution Normal TI Coverage'] = normal_counter/trials
    df['Beta Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Beta Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials 
    df['Beta Distribution Bootstrap TI Coverages'] = bootstrap_coverages     
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Beta Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Beta Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Beta Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['Beta Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Beta Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['Beta Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages
    df['Beta Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Beta Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths

    df['Beta Distribution Normal TI Widths'] = normal_widths
    df['Beta Distribution KDE TI Widths'] = kde_widths
    df['Beta Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Beta Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Beta Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths

test_beta_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)



def test_exponential_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.exponential(3, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0
    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []
 

    for x in range(trials):

        print(x)

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1

        print('Exponential Trial')


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Exponential Distribution Normal TI Coverage'] = normal_counter/trials
    df['Exponential Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Exponential Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['Exponential Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Exponential Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Exponential Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Exponential Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['Exponential Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Exponential Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials
    df['Exponential Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Exponential Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths
    df['Exponential Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages


    df['Exponential Distribution Normal TI Widths'] = normal_widths
    df['Exponential Distribution KDE TI Widths'] = kde_widths
    df['Exponential Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Exponential Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Exponential Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths
test_exponential_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)












def test_pareto_distribution_coverage(mean, std, sample_size, trials, pop_size, proportion, confidence):
    rng = np.random.default_rng()
    population = rng.pareto(2, pop_size)
    normal_counter = 0
    nonparametric_counter = 0
    kde_counter = 0
    bootstrap_counter = 0
    smooth_bootstrap_kde_counter = 0
    count = 0
    normal_coverages = []
    kde_coverages = []
    nonparametric_coverages = []
    bootstrap_coverages = []
    smooth_bootstrap_kde_coverages = []
    approx_normal_coverages = []
    approx_normal_widths = []
    approx_normal_counter = 0
    normal_widths = []
    kde_widths = []
    nonparametric_widths = []
    bootstrap_widths = []
    smooth_bootstrap_kde_widths = []
 

    for x in range(trials):
        print(x)
        print('Pareto Trial')

  

        sample = rng.choice(population, sample_size, replace=True)
        print(sample[:5])

        normal_ti = Tolerance_Intervals.compute_normal_ti(sample, proportion, confidence)
        print(normal_ti)

        within_bounds = np.logical_and(population >= normal_ti[0], population <= normal_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        normal_coverages.append(empirical_coverage)
        approx_normal_ti = Tolerance_Intervals.compute_approx_normal_ti(sample, proportion, confidence)

        within_bounds_approx = np.logical_and(population >= approx_normal_ti[0], population <= approx_normal_ti[1])
        empirical_coverage_approx = np.sum(within_bounds_approx)/len(population)
        if empirical_coverage_approx >= proportion:

            approx_normal_counter += 1  

        approx_normal_coverages.append(empirical_coverage_approx)
        approx_normal_widths.append(approx_normal_ti[1] - approx_normal_ti[0])
        normal_widths.append(normal_ti[1] - normal_ti[0])

        if empirical_coverage >= proportion:

            normal_counter += 1

        nonparametric_ti = Tolerance_Intervals.compute_nonparametric_ti(sample, proportion, confidence)
        within_bounds = np.logical_and(population >= nonparametric_ti[0], population <= nonparametric_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        nonparametric_coverages.append(empirical_coverage)
        nonparametric_widths.append(nonparametric_ti[1] - nonparametric_ti[0])

        if empirical_coverage >= proportion:

            nonparametric_counter += 1


        kde_ti = Tolerance_Intervals.compute_bootstrap_kde_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= kde_ti[0], population <= kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        kde_coverages.append(empirical_coverage)
        kde_widths.append(kde_ti[1] - kde_ti[0])

        if empirical_coverage >= proportion:

            kde_counter += 1


        bootstrap_ti = Tolerance_Intervals.compute_bootstrap_ti(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_ti[0], population <= bootstrap_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        bootstrap_coverages.append(empirical_coverage)
        bootstrap_widths.append(bootstrap_ti[1] - bootstrap_ti[0])

        if empirical_coverage >= proportion:

            bootstrap_counter += 1


        bootstrap_smooth_kde_ti = Tolerance_Intervals.compute_bootstrap_kde_shrunk_smooth(sample, proportion, confidence)

        within_bounds = np.logical_and(population >= bootstrap_smooth_kde_ti[0], population <= bootstrap_smooth_kde_ti[1])

        empirical_coverage = np.sum(within_bounds)/len(population)
        smooth_bootstrap_kde_coverages.append(empirical_coverage)
        smooth_bootstrap_kde_widths.append(bootstrap_smooth_kde_ti[1] - bootstrap_smooth_kde_ti[0])

        if empirical_coverage >= proportion:

            smooth_bootstrap_kde_counter += 1

    print("Normal TI Coverage")
    print(normal_counter/trials)

    results_dict['Pareto Distribution Normal TI Coverage'] = normal_counter/trials
    df['Pareto Distribution Normal TI Coverages'] = normal_coverages

    print("Bootstrap Coverage")
    print(bootstrap_counter/trials)
    results_dict['Pareto Distribution Bootstrap TI Coverage'] = bootstrap_counter/trials
    df['Pareto Distribution Bootstrap TI Coverages'] = bootstrap_coverages
    print('Non Parametric Coverage')

    print(nonparametric_counter/trials)
    results_dict['Pareto Distribution Non Parametric TI Coverage'] = nonparametric_counter/trials
    df['Pareto Distribution Non Parametric TI Coverages'] = nonparametric_coverages

    print("KDE Bootstrap Coverage")
    print(kde_counter/trials)
    results_dict['Pareto Distribution KDE Bootstrap TI Coverage'] = kde_counter/trials
    df['Pareto Distribution KDE Bootstrap TI Coverages'] = kde_coverages
    print("Smooth KDE BootstrapCoverage")
    print(smooth_bootstrap_kde_counter/trials)
    results_dict['Pareto Distribution Smooth KDE Bootstrap TI Coverage'] = smooth_bootstrap_kde_counter/trials

    df['Pareto Distribution Approx Normal Distribution Normal TI Coverages'] = approx_normal_coverages
    df['Pareto Distribution Approx Normal Distribution Normal TI Widths'] = approx_normal_widths
    df['Pareto Distribution Smooth KDE Bootstrap TI Coverages'] = smooth_bootstrap_kde_coverages

    df['Pareto Distribution Normal TI Widths'] = normal_widths
    df['Pareto Distribution KDE TI Widths'] = kde_widths
    df['Pareto Distribution Non Parametric TI Widths'] = nonparametric_widths
    df['Pareto Distribution Bootstrap TI Widths'] = bootstrap_widths
    df['Pareto Distribution Smooth KDE TI Widths'] = smooth_bootstrap_kde_widths
test_pareto_distribution_coverage(0,1, 100, 1000, 10000, 0.95,0.95)



# Convert to DataFrame
df2 = pd.DataFrame(list(results_dict.items()), columns=['Construction Type', 'EmpiricalCoverage'])

# Save to CSV
df2.to_csv('coverage_results.csv', index=False)

df.to_csv('detailed_coverage_results.csv', index=False)


print(df)
