# Generate TNGD parameter field based on fitting results
# Yuan Liu
# 06/15/2023


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as st
import os


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def save_array(array, target_location, file_name):
    # target_file
    target_file_location = target_location + "/" + file_name
    np.save(target_file_location, array)


def get_season(month):
    if month in [12, 1, 2]:
        return "win"
    elif month in [3, 4, 5]:
        return "spr"
    elif month in [6, 7, 8]:
        return "sum"
    elif month in [9, 10, 11]:
        return "fal"
    else:
        return "Invalid month"

if __name__ == "__main__":


    season_list = ['win', 'spr', 'sum', 'fal']

    month_list = np.arange(1, 13)

    # if generate param with only mtpr as covariate
    # mtpr_only = False

    for season in season_list:
        print(f"Start to create parameter fields for season {season}")
        # merge then into a single dataframe
        full_csgd_param_df = pd.DataFrame()

        # get the season name
        # season = get_season(month)

        # fold_id = 0

        for batch_id in range(1000):
            # save_folder = r"/home/yliu2232/miss_design_storm_hlm_model/6_hour_tracking/6h_tngd_fitting/{0}".format(month)
            save_folder = fr"/home/yliu2232/model_based_pmp/denison_dam/cesm2_random_storms/csgd_flex_sigma_fitting_2002_2021/{season}"

            csgd_param_df = pd.read_csv(
                save_folder + "/" + f"csgd_param_{batch_id}.csv")
            # remove those has -9999
            # clean_csgd_param_df = csgd_param_df[csgd_param_df['1'] != -9999]
            # concat
            full_csgd_param_df = pd.concat([full_csgd_param_df, csgd_param_df], axis=0)

        # get the array of parameters
        mu_slope_array = full_csgd_param_df['mu_slope'].values.reshape(108, 260)
        mu_intercept_array = full_csgd_param_df['mu_intercept'].values.reshape(108, 260)
        beta_0_array = full_csgd_param_df['beta_0'].values.reshape(108, 260)
        beta_4_array = full_csgd_param_df['beta_4'].values.reshape(108, 260)
        beta_5_array = full_csgd_param_df['beta_5'].values.reshape(108, 260)

        # if mtpr_only == True:
        #     alpha_5_array = np.zeros((630, 1024))
        # else:
        # alpha_5_array = full_csgd_param_df['5'].values.reshape(108, 260)

        # gg_c_array = full_csgd_param_df['gg_c'].values.reshape(106, 280)

        logit_intercept_array = full_csgd_param_df['logit_intercept'].values.reshape(108, 260)
        logit_mtpr_array = full_csgd_param_df['logit_mtpr'].values.reshape(108, 260)
        # logit_tcwv_array = full_csgd_param_df['logit_tcwv'].values.reshape(106, 280)

        mu_clim_array = full_csgd_param_df['mu_clim'].values.reshape(108, 260)
        sigma_clim_array = full_csgd_param_df['sigma_clim'].values.reshape(108, 260)

        # get the loglikelihood array
        # loglik_fitted_array = full_csgd_param_df['loglik_fitted_model'].values.reshape(106, 280)
        # loglik_null_array = full_csgd_param_df['loglik_null_model'].values.reshape(106, 280)
        # rsq_array = full_csgd_param_df['pseudo_rsq'].values.reshape(106, 280)

        # save the parameter fields
        save_folder = rf"/home/yliu2232/model_based_pmp/denison_dam/cesm2_random_storms/csgd_flex_sigma_fitting_2002_2021/param_fields/{season}"
        create_folder(save_folder)

        # save the parameters for mean
        save_array(mu_slope_array, save_folder, 'mu_slope_array.npy')
        save_array(mu_intercept_array, save_folder, 'mu_intercept_array.npy')
        save_array(beta_0_array, save_folder, 'beta_0_array.npy')
        save_array(beta_4_array, save_folder, 'beta_4_array.npy')
        save_array(beta_5_array, save_folder, 'beta_5_array.npy')
        # save the shape parameter
        # save_array(gg_c_array, save_folder, 'gg_c_array.npy')

        # save logistic regression parameters
        save_array(logit_intercept_array, save_folder, 'logit_intercept_array.npy')
        save_array(logit_mtpr_array, save_folder, 'logit_mtpr_array.npy')
        # save_array(logit_tcwv_array, save_folder, 'logit_tcwv_array.npy')

        # save the climatological parameters
        save_array(mu_clim_array, save_folder, 'mu_clim_array.npy')
        save_array(sigma_clim_array, save_folder, 'sigma_clim_array.npy')

        # save the likelihood array
        # save_array(loglik_fitted_array, save_folder, 'loglik_fitted_model_array.npy')
        # save_array(loglik_null_array, save_folder, 'loglik_null_model_array.npy')
        # save_array(rsq_array, save_folder, 'pseudo_rsq_array.npy')

