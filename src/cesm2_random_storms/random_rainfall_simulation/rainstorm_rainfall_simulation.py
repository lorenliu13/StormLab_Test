# Generate simulated rainfall fields based on noise and conditional distribution parameter fields.
# Yuan Liu
# 2023/02/13


import xarray as xr
import os
import scipy.stats as st
import numpy as np


def create_folder(folder_name):
    try:
        # Create the folder
        os.makedirs(folder_name)
    except OSError:
        # If the folder already exists, ignore the error
        pass


def rainfall_generation(noise_array, logic_mu_array, scipy_a_array,
                        scipy_scale_array, gg_c_array, high_res_mtpr_array):

    # convert the gaussian noise field to 0-1 field
    first_prob_array = st.norm.cdf(noise_array, loc=0, scale=1)

    print('Start to calculate prob field.')
    # del noise_array
    # gc.collect()

    # replace nan with zero
    logic_mu_array = np.where(np.isnan(logic_mu_array), 0, logic_mu_array)
    # compute the probability of dry
    dry_prob_array = 1 - logic_mu_array

    # Compute the second probability
    second_prob_array = first_prob_array - dry_prob_array
    # change all negative probability to np.nan
    second_prob_array = np.where((second_prob_array < 0) | (dry_prob_array == 1), np.nan, second_prob_array)
    # print(second_prob_array[second_prob_array > 1])
    # rescale the remaining to (0, 1)
    second_prob_array = second_prob_array / logic_mu_array

    # force the second prob to be zero if cesm2 is < 0.01
    second_prob_array = np.where(high_res_mtpr_array > 0.01, second_prob_array, 0)

    # add an upper bound for probability
    # second_prob_array[second_prob_array > 0.9995] = 0.9995

    # clear memory
    # del first_prob_array, logic_mu_array, dry_prob_array
    # gc.collect()

    sim_rainfall_array = np.zeros(scipy_scale_array.shape)

    print('Start to calculate rainfall field.')
    # compute ppf for each time step
    for time_index in range(scipy_scale_array.shape[0]):
        # time_index = 0

        curr_second_prob_array = second_prob_array[time_index]
        # get those rainfall probability > 0
        rainfall_mask = ~np.isnan(curr_second_prob_array)

        curr_a_array = scipy_a_array[time_index]
        # curr_c_array = scipy_gg_c_array
        curr_scale_array = scipy_scale_array[time_index]

        # get the corresponding GG distribution params
        p = curr_second_prob_array[rainfall_mask]

        # get the rainfall magnitude through inverse pdf
        # an nan appears when p > 1
        rainfall_magnitude_vector = st.gamma.ppf(p, a=curr_a_array[rainfall_mask],
                                                 scale=curr_scale_array[rainfall_mask]
                                                 ) ** (1 / gg_c_array[rainfall_mask])
        rainfall_magnitude_vector[np.isnan(rainfall_magnitude_vector)] = 0
        rainfall_magnitude_vector[np.isinf(rainfall_magnitude_vector)] = 0

        # print(np.sum(np.isnan(rainfall_magnitude_vector)))
        # restore the rainfall to original location
        rainfall_array = np.zeros(curr_second_prob_array.shape)
        rainfall_array[rainfall_mask] = rainfall_magnitude_vector

        sim_rainfall_array[time_index, :, :] = rainfall_array

    # # clear memory
    # del scipy_a_array, scipy_scale_array, scipy_gg_c_array
    # gc.collect()

    return sim_rainfall_array


if __name__ == "__main__":

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the desired save folder relative to the script's directory
    save_folder = os.path.join(script_dir, '../../../output/cesm2_rainstorm_simulation')
    create_folder(save_folder)

    # Navigate to the desired load folder relative to the script's directory
    cesm_rainstorm_distr_param_folder = os.path.join(script_dir, '../../../data/cesm2/cesm2_rainstorms_distr_params')
    cesm_rainstorm_noise_folder = os.path.join(script_dir, '../../../data/cesm2/cesm2_rainstorm_simulation')

    # read argument input
    ar_id = 202200018
    realization = 1

    # load noise field realization
    noise_xarray = xr.load_dataset(cesm_rainstorm_noise_folder + "/" + "{0}_{1}_noise.nc".format(ar_id, realization))


    # load rainfall probability
    logic_mu_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_logit_wet_p.nc".format(ar_id))
    # load scipy parameters
    scipy_a_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_scipy_a.nc".format(ar_id))
    scipy_scale_xarray = xr.load_dataset(cesm_rainstorm_distr_param_folder + "/" + "{0}_scipy_scale.nc".format(ar_id))
    # scipy_loc_xarray = xr.load_dataset("{0}_scipy_loc.nc".format(ar_id))

    # get the noise array
    noise_array = noise_xarray['aorc'].data
    time_steps = noise_xarray['time'].data
    # get GG and logistic regression model parameters
    logic_mu_array = logic_mu_xarray['aorc'].data
    scipy_a_array = scipy_a_xarray['aorc'].data
    scipy_scale_array = scipy_scale_xarray['aorc'].data
    # scipy_loc_array = scipy_loc_xarray['aorc'].data

    # convert the gaussian noise field to 0-1 field
    first_prob_array = st.norm.cdf(noise_array, loc=0, scale=1)

    # 0.5 * (1 + erf(noise_array / np.sqrt(2)))

    # replace nan with zero
    logic_mu_array = np.where(np.isnan(logic_mu_array), 0, logic_mu_array)
    # compute the probability of dry
    dry_prob_array = 1 - logic_mu_array

    # Compute the second probability
    second_prob_array = first_prob_array - dry_prob_array
    # change all negative probability to np.nan
    second_prob_array = np.where((second_prob_array < 0) | (dry_prob_array == 1), np.nan, second_prob_array)
    # print(second_prob_array[second_prob_array > 1])
    # rescale the remaining to (0, 1)
    second_prob_array = second_prob_array / logic_mu_array

    # add an upper bound for probability
    # second_prob_array[second_prob_array > 0.995] = 0.995
    # print(second_prob_array[second_prob_array > 1])


    sim_rainfall_array = np.zeros(noise_array.shape)
    # compute ppf for each time step
    for time_index in range(noise_array.shape[0]):
        # time_index = 0

        curr_second_prob_array = second_prob_array[time_index]
        # get those rainfall probability > 0
        rainfall_mask = ~np.isnan(curr_second_prob_array)

        curr_a_array = scipy_a_array[time_index]
        curr_c_array = scipy_gg_c_array
        curr_scale_array = scipy_scale_array[time_index]

        # get the corresponding GG distribution params
        p = curr_second_prob_array[rainfall_mask]
        # print(p[p > 1])

        # get the rainfall magnitude through inverse pdf
        # an nan appears when p > 1
        rainfall_magnitude_vector = st.gamma.ppf(p, a=curr_a_array[rainfall_mask],
                                                 scale=curr_scale_array[rainfall_mask]
                                                 )
        rainfall_magnitude_vector[np.isnan(rainfall_magnitude_vector)] = 0
        rainfall_magnitude_vector[np.isinf(rainfall_magnitude_vector)] = 0

        # print(np.sum(np.isnan(rainfall_magnitude_vector)))
        # restore the rainfall to original location
        rainfall_array = np.zeros(curr_second_prob_array.shape)
        rainfall_array[rainfall_mask] = rainfall_magnitude_vector

        sim_rainfall_array[time_index, :, :] = rainfall_array

    # print the number of nan values
    # print("The number of nan: {0}".format(np.sum(np.isnan(sim_rainfall_array))))

    # set up AORC coordinates
    aorc_lat = np.linspace(50, 29, 630)
    aorc_lon = np.linspace(-113.16734, -79.068704, 1024)

    # save_location = r"/home/yliu2232/miss_design_storm/random_storms/rainfall_field/{0}/{1}".format(year, ar_id)

    # create a ds for rainfall field
    rainfall_ds = xr.Dataset(
        {'aorc': (['time', 'latitude', 'longitude'], sim_rainfall_array)},
        coords={
            'time': time_steps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Simulated rainfall field {0} for AORC AR id {1}".format(realization,
            ar_id)}
    )

    # save the dataset
    rainfall_ds.to_netcdf(save_folder + "/" + "{0}_{1}_sim_rainfall.nc".format(ar_id, realization), encoding={"aorc": {"dtype": "float32", "zlib": True}})
