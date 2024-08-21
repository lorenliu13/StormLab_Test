import os
import numpy as np
import xarray as xr
from scipy import interpolate
import sys
import gc
import warnings
warnings.simplefilter("ignore")


def linear_interpolation(array, old_x, old_y, new_x, new_y):
    """
    Linear interpolation of current 2-d field. Note that the latitude need to be flipped.
    :param array:
    :param old_x:
    :param old_y:
    :param new_x:
    :param new_y:
    :return:
    """
    # get current array
    curr_flip_array = np.flip(array, axis=0)

    # create interpolation function
    f = interpolate.interp2d(old_x, np.flip(old_y), curr_flip_array, kind='linear')

    # use it to interpolate to new grid
    new_z = f(new_x, np.flip(new_y))

    # flip it
    new_z = np.flip(new_z, axis=0)

    return new_z


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def replace_with_nan(array):
    # replace -9999 into nan
    array = np.where(array == -9999, np.nan, array)
    return array


def generate_TNGD_param_field(mtpr_array, ar_sesson):

    # get coefficients
    # if the coefficient is -9999, that means no data available here, so replace it with np.nan
    mu_slope_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/mu_slope_array.npy"))
    mu_intercept_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/mu_intercept_array.npy"))
    gg_c_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/gg_c_array.npy"))
    beta_0_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/beta_0_array.npy"))
    beta_4_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/beta_4_array.npy"))
    beta_5_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/beta_5_array.npy"))

    mu_clim_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/mu_clim_array.npy"))
    sigma_clim_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/sigma_clim_array.npy"))
    # gg_c_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/gg_c_array.npy"))

    # load logistic regression coefficients
    logit_intercept_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/logit_intercept_array.npy"))
    logit_mtpr_array = replace_with_nan(np.load(f"param_fields/{ar_sesson}/logit_mtpr_array.npy"))
    # logit_tcwv_array = replace_with_nan(np.load(f"6h_tngd_fitting_mtpr_only/{ar_sesson}/logit_tcwv_array.npy"))

    # set up AORC coordinates
    aorc_lat = np.linspace(36.49934, 32.932816, 108)
    aorc_lon = np.linspace(-104.367692, -95.734704, 260)

    cesm_lat = np.linspace(37.225131, 32.513089, 6)
    cesm_lon = np.linspace(-105, -95, 9)

    # print("AR id: {0}".format(ar_id))

    # load the aorc and era5 covariate xarray data for the ar event
    # mtpr_xarray = xr.load_dataset("prect_cesm_res.nc".format(ar_id))
    # tcwv_xarray = xr.load_dataset("{0}_tmq_cesm_res.nc".format(ar_id))

    # get time steps
    # ar_time_stamps = mtpr_xarray['time'].data
    # mtpr_array = cesm_array mtpr_xarray['prect'].data
    # tcwv_array = tcwv_xarray['tmq'].data

    # initialize high-res array
    high_res_mtpr_array = np.zeros((mtpr_array.shape[0], 108, 260))
    # high_res_tcwv_array = np.zeros((ar_time_stamps.shape[0], 630, 1100))

    for i in range(mtpr_array.shape[0]):
        # interpolate
        high_res_mtpr_array[i] = linear_interpolation(mtpr_array[i], cesm_lon, cesm_lat, aorc_lon, aorc_lat)
        # high_res_tcwv_array[i] = linear_interpolation(tcwv_array[i], cesm_lon, cesm_lat, aorc_lon, aorc_lat)

    # save the high resolution mtpr array
    # np.save('{ar_id}_high_res_mtpr_array.npy', high_res_mtpr_array)
    # create empty array to save data
    full_scipy_a_array = np.zeros((mtpr_array.shape[0], 108, 260))
    full_scipy_scale_array = np.zeros((mtpr_array.shape[0], 108, 260))

    for time_index in range(high_res_mtpr_array.shape[0]):

        # get current mtpr array
        curr_mtpr_array = high_res_mtpr_array[time_index]
        # curr_tcwv_array = high_res_tcwv_array[time_index]

        eta_array = curr_mtpr_array * mu_slope_array + mu_intercept_array
        mu = mu_clim_array / beta_0_array * np.log(1 + (np.exp(beta_0_array) - 1) * eta_array)
        # mu = mu_clim_array / alpha_1_array * np.log1p(np.expm1(alpha_1_array) * logarg)
        sigma = beta_4_array * sigma_clim_array * np.sqrt(mu / mu_clim_array) ** beta_5_array

        # compute current parameter a
        a = mu ** 2 / sigma ** 2
        # compute current parameter scale
        scale = sigma ** 2 / mu

        full_scipy_a_array[time_index] = a
        full_scipy_scale_array[time_index] = scale

    # Create the dataset for scipy a parameter
    full_scipy_a_array = full_scipy_a_array.astype(np.float32)
    # np.save('{0}_scipy_a.npy'.format(ar_id), full_scipy_a_array)
    # del full_scipy_a_array
    # gc.collect()

    full_scipy_scale_array = full_scipy_scale_array.astype(np.float32)
    # np.save('{0}_scipy_scale.npy'.format(ar_id), full_scipy_scale_array)
    # del full_scipy_scale_array
    # gc.collect()

    # gg_c_array = gg_c_array.astype(np.float32)
    # np.save('{0}_scipy_c.npy'.format(ar_id), gg_c_array)
    # del gg_c_array
    # gc.collect()

    full_wet_p_array = np.zeros((mtpr_array.shape[0], 108, 260))  # compute the probability of dry

    for time_index in range(high_res_mtpr_array.shape[0]):
        # get current mtpr array
        curr_mtpr_array = high_res_mtpr_array[time_index]
        # curr_tcwv_array = high_res_tcwv_array[time_index]

        # compute the wet probability
        eta_array = logit_intercept_array + logit_mtpr_array * curr_mtpr_array # + logit_tcwv_array * curr_tcwv_array
        wet_p_array = 1 / (1 + np.exp((-1.0) * eta_array))

        full_wet_p_array[time_index] = wet_p_array

    full_wet_p_array = full_wet_p_array.astype(np.float32)

    # not used 04/17/2024: set the wet probability to 0 if the cesm rainfall is lower than 0.01 mm
    # full_wet_p_array[high_res_mtpr_array <= 0.01] = 0

    # np.save('{0}_logit_wet_p.npy'.format(ar_id), full_wet_p_array)
    # del full_wet_p_array
    # gc.collect()

    return full_scipy_a_array, full_scipy_scale_array, full_wet_p_array, gg_c_array, high_res_mtpr_array

if __name__ == "__main__":

    # read argument input
    ar_id = int(sys.argv[1])
    ar_month = int(sys.argv[2])

    generate_TNGD_param_field(ar_id, ar_month)