# On every ERA5 grid, fit a CSGD hybrid distribution
# Yuan Liu
# 03/28/2023


import pandas as pd
import scipy.stats as st
import numpy as np
import sys
# from CSGD_hybrid import fit_regression_v2
# from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import scipy as sp
from scipy.special import beta
import math


def safe_exp(x):
    return np.exp(np.clip(x, -709, 709))  # 709 is roughly log(float_max)


def penal_mle_regression_crps(params, covariates, clim_params, obs_prcp):
    # parameter list: parameters including: slope coeff for covariates, intercept, beta0, beta4, beta5
    # covariates: dataframe of fitting covariates, (time_size, cov_number)
    # clim_params: climatological mean and standard deviation
    # obs_prcp: observed prcp time series (reduced by applying a transformation factor)
    # alpha: penalize coefficient default = 1

    # retrieve slope coefficeints
    slope_coefficients = params[:covariates.shape[1]]
    mu_intercept = params[covariates.shape[1]]
    beta_0 = params[-3]
    beta_4 = params[-2]
    beta_5 = params[-1]

    mu_clim = clim_params[0]
    sigma_clim = clim_params[1]

    # retrieve the linear coefficeint for std
    # std_slope = params[-1]

    # calculate the eta
    eta_series = np.sum(covariates * slope_coefficients, axis=1) + mu_intercept

    # compute the argument in log
    arg_series = 1 + (safe_exp(beta_0) - 1) * eta_series
    # make sure it is greater than 1
    arg_series = np.maximum(arg_series, 1.00001)

    # calcualte the mu
    mu_series = mu_clim / beta_0 * np.log(arg_series)
    # mu_series = np.exp(eta_series)
    # ensure mu is always positive
    mu_series = np.maximum(mu_series, 1e-8)  # or another small positive number

    # calculate the sigma
    sigma_series = beta_4 * sigma_clim * np.sqrt(mu_series / mu_clim) ** beta_5  #

    # CRPS-based Minimization Function
    delta = 0  # use climate delta as CSGD detla
    k = np.power(mu_series / sigma_series, 2)  # compute the shape parameter
    theta = np.power(sigma_series, 2) / mu_series  # compute the scale parameter
    betaf = beta(0.5, 0.5 + k)
    ysq = (obs_prcp - delta) / theta  # use stageIV to rescale
    csq = -delta / theta  # when y = 0
    Fysq = sp.stats.gamma.cdf(ysq, k, scale=1)
    Fcsq = sp.stats.gamma.cdf(csq, k, scale=1)
    FysqkP1 = sp.stats.gamma.cdf(ysq, k + 1, scale=1)
    FcsqkP1 = sp.stats.gamma.cdf(csq, k + 1, scale=1)
    Fcsq2k = sp.stats.gamma.cdf(2 * csq, 2 * k, scale=1)

    # calculate the crps metrics
    crps = ysq * (2. * Fysq - 1.) - csq * np.power(Fcsq, 2) + k * (
            1. + 2. * Fcsq * FcsqkP1 - np.power(Fcsq, 2) - 2. * FysqkP1) - k / math.pi * betaf * (1. - Fcsq2k)

    return 10000. * np.nanmean(theta * crps)


def fit_csgd_distribution(nonzero_aorc_series, nonzero_mtpr_series):
    """
    Simplified cropeed CSGD distrbution fitting.
    :param nonzero_aorc_series:
    :param nonzero_mtpr_series:
    :return:
    """

    # if the training data is not empty
    if nonzero_aorc_series.shape[0] != 0:

        # fit a gamma model
        st_gamma_a, st_gamma_loc, st_gamma_scale = st.gamma.fit(nonzero_aorc_series, method='MM', floc=0)
        # calculate the mean
        st_gamma_mean = st.gamma.mean(a=st_gamma_a, scale=st_gamma_scale)
        st_gamma_std = st.gamma.std(a=st_gamma_a, scale=st_gamma_scale)

        # get clim series
        clim_params = np.array([st_gamma_mean, st_gamma_std])

        obs_prcp = nonzero_aorc_series
        covariates = nonzero_mtpr_series

        # check if there is only one covaraites
        if len(covariates.shape) == 1:
            covariates = covariates[:, np.newaxis]  # add one extract columns dimension

        init_guess_slope_coeff = np.ones(covariates.shape[1]) * 0.1
        # set up initial guess for intercept
        init_guess_intercept = 0.001
        # set up initial guess for slope
        init_guess_beta_0 = 0.001
        init_guess_beta_4 = 1
        init_guess_beta_5 = 1
        # combine
        init_guess_params = np.concatenate(
            [init_guess_slope_coeff, [init_guess_intercept], [init_guess_beta_0], [init_guess_beta_4],
             [init_guess_beta_5]])

        bounds = []
        # check if there is only one covaraites
        # if len(covariates.shape) == 1:
        #     bounds.append((-10, 10))
        # else:
        for index in range(covariates.shape[1]):
            bounds.append((-10, 10))
        # append bounds for intercept
        bounds.append((-5, 5))
        # append bounds for beta 0
        bounds.append((0.001, 2))
        # append bounds for beta 4
        bounds.append((0.001, 10))
        # append bounds for beta 5
        bounds.append((0.001, 10))

        output = sp.optimize.minimize(penal_mle_regression_crps, init_guess_params,
                                         args=(covariates, clim_params, obs_prcp),
                                         method='L-BFGS-B', bounds=bounds)
        par = output.x # get the fitting parameters

    else:
        par = np.ones((1, 5)) * -9999# alpha 1-3, covariate coefficients, 3 stationary parameter, 1 gg_c parameter

        clim_params = np.array([-9999, -9999]) # set climate mean and sigma to -9999
        # loglikelihood_fitted_model = -9999
        # loglikelihood_constant_model = -9999
        # pseudo_rsq = -9999


    # generate a dataframe
    # par_df = pd.DataFrame(par, columns=alpha_col_names)
    par_df = pd.DataFrame()
    par_df['mu_slope'] = [par[0]] # this is the slope coefficient for covariate when calculating the mean
    par_df['mu_intercept'] = [par[1]] # this is the intercept coefficient for covariate when calculating the mean
    par_df['beta_0'] = [par[2]] # this is the parameter control linear-log relationship between Y and X
    par_df['beta_4'] = [par[3]] # this is the linear coefficient for variance
    par_df['beta_5'] = [par[4]] # this is the power coefficient for variance
    par_df['mu_clim'] = [clim_params[0]]
    par_df['sigma_clim'] = [clim_params[1]]


    return par_df



def logistic_regression(curr_aorc_series, curr_mtpr_series):

    # Get the record where there is rainfall
    # training_data = training_df[training_df['aorc'] > 0.01]
    nonzero_aorc_series = curr_aorc_series[curr_aorc_series > 0.01]

    # if it is not empty
    if nonzero_aorc_series.shape[0] != 0:
        # Fit a logistic regression model using sklearn
        y = np.where(curr_aorc_series > 0.01, 1, 0)
        X = curr_mtpr_series.reshape(-1, 1)
        clf = LogisticRegression(random_state=0).fit(X, y)
        # Get the logistic regression coefficients
        logit_intercept = clf.intercept_[0]
        mtpr_coef = clf.coef_[0, 0]

    else:
        logit_intercept = -9999
        mtpr_coef = -9999

    return logit_intercept, mtpr_coef


if __name__ == "__main__":

    # get the batch index
    batch_index = int(sys.argv[1])

    # load csv files
    aorc_df = pd.read_csv("{0}_aorc.csv".format(batch_index))
    mtpr_df = pd.read_csv("{0}_mtpr.csv".format(batch_index))


    # get the columns as grid index s
    full_grid_index_list = aorc_df.columns
    # grid_index_list = full_grid_index_list[full_grid_index_list != 'time_step'] # drop the time step column name

    # create a full dataframe
    batch_grid_df = pd.DataFrame()

    # for each grid
    for grid_index in full_grid_index_list:

        # get aorc and covariates for csgd fitting
        curr_aorc_series = aorc_df[str(grid_index)].values
        curr_mtpr_series = mtpr_df[str(grid_index)].values

        # get nonzero series
        nonzero_aorc_series = curr_aorc_series[curr_aorc_series > 0.01]
        nonzero_mtpr_series = curr_mtpr_series[curr_aorc_series > 0.01]

        # Fit the hybrid csgd model
        single_grid_df = fit_csgd_distribution(nonzero_aorc_series=nonzero_aorc_series, nonzero_mtpr_series=nonzero_mtpr_series)

        # Fit the logistic regression model
        logit_intercept, mtpr_coef = logistic_regression(curr_aorc_series=curr_aorc_series, curr_mtpr_series=curr_mtpr_series)
        # Save the logistic regression coefficient
        single_grid_df['logit_intercept'] = [logit_intercept]
        single_grid_df['logit_mtpr'] = [mtpr_coef]

        # Add the grid id
        single_grid_df['grid_id'] = [grid_index]
        single_grid_df['batch_id'] = [batch_index]

        batch_grid_df = pd.concat([batch_grid_df, single_grid_df], axis=0, ignore_index=True)

    # save the fit_param_df
    batch_grid_df.to_csv("csgd_param_{0}.csv".format(batch_index), index=False)