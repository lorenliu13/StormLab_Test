import xarray as xr
import numpy as np
import sys
import random


def _tukey(R, alpha):
    W = np.ones_like(R)
    N = min(R.shape[0], R.shape[1])

    mask1 = R < int(N / 2)
    mask2 = R > int(N / 2) * (1.0 - alpha)
    mask = np.logical_and(mask1, mask2)
    W[mask] = 0.5 * (
        1.0 + np.cos(np.pi * (R[mask] / (alpha * 0.5 * N) - 1.0 / alpha + 1.0))
    )
    mask = R >= int(N / 2)
    W[mask] = 0.0

    return W

def tukey_window_generation(m, n):
    X, Y = np.meshgrid(np.arange(n), np.arange(m))
    R = np.sqrt((X - int(n / 2)) ** 2 + (Y - int(m / 2)) ** 2)
    window_mask = _tukey(R, alpha = 0.2)
    # add small value to avoid zero
    window_mask += 1e-6
    return window_mask

def compute_amplitude_spectrum(field):
    # perform 2-d Fourier transform
    F = do_fft2(field)
    # normalize the imagery and real part
    F.imag = (F.imag - np.mean(F.imag)) / np.std(F.imag)
    F.real = (F.real - np.mean(F.real)) / np.std(F.real)
    # get the amplitude
    F_abs = np.abs(F)

    return F_abs

def do_fft2(array):
    return np.fft.fft2(array)

def do_ifft2(array):
    return np.fft.ifft2(array)

def FFST_based_noise_generation(field, noise, win_size, overlap, ssft_war_thr):
    """
    Generate noise field using FFST method based on rainfall field
    :param field: rainfall field
    :param noise: white noise
    :param win_size: window size (128, 128)
    :param overlap: overlap ratio of windows
    :param ssft_war_thr: wet area ratio for FFST, 0.1
    :return: correated noise field
    """

    dim_x = field.shape[1]  # get the column number 1100
    dim_y = field.shape[0]  # get the row number 630

    # number of windows
    num_windows_y = int(np.ceil(float(dim_y) / win_size[0]))
    num_windows_x = int(np.ceil(float(dim_x) / win_size[1]))

    # perform global FFT
    global_F = compute_amplitude_spectrum(field)
    noise_F = do_fft2(noise)  # get the white noise FFT field

    # get global noise field
    global_noise_array = do_ifft2(noise_F * global_F).real
    # final_noise_array = global_noise_array.copy()
    final_noise_array = np.zeros(global_noise_array.shape)
    final_weight_array = np.zeros(global_noise_array.shape)

    # loop rows
    for i in range(num_windows_y):
        # loop columns: this performs row-major looping, which is faster
        for j in range(num_windows_x):

            # prepare indices
            idxi = np.zeros(2).astype(int)
            idxj = np.zeros(2).astype(int)

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))  # get the upper y index
            idxi[1] = int(np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y)))  # get the lower y index

            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0))) # get the left x index
            idxj[1] = int(np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x)))  # get the right x index

            # for each window, get the subregion
            window_rainfall_array = field[idxi[0]: idxi[1], idxj[0]: idxj[1]]

            # get the mask
            curr_window_dimension = (idxi[1] - idxi[0], idxj[1] - idxj[0])
            tukey_window = tukey_window_generation(m=curr_window_dimension[0], n=curr_window_dimension[1])

            # get the wet area ratio as the portion of wet grid in the local window
            weighted_window_rainfall_array = window_rainfall_array * tukey_window
            wet_area_raito = np.sum((weighted_window_rainfall_array) > 0.01) / (
                        curr_window_dimension[0] * curr_window_dimension[1])

            # get the full masked rainfall field
            full_mask = np.zeros((dim_y, dim_x))
            full_mask[idxi[0]: idxi[1], idxj[0]: idxj[1]] = tukey_window

            if wet_area_raito > ssft_war_thr:

                # get masked rainfall fields
                full_masked_rainfall_array = field * full_mask

                # perform fourier transform to calculate the amplitude spectrum
                local_F = compute_amplitude_spectrum(full_masked_rainfall_array)

                # generate local noise
                local_noise_array = do_ifft2(noise_F * local_F).real

                # update the final noise field
                final_noise_array += local_noise_array * full_mask
                final_weight_array += full_mask

            else:
                # add global noise
                final_noise_array += global_noise_array * full_mask
                final_weight_array += full_mask

    # compute the final noise as weighted average
    final_noise_array[final_weight_array > 0] /= final_weight_array[final_weight_array > 0]
    # normalize it
    final_noise_array = (final_noise_array - np.mean(final_noise_array)) / np.std(final_noise_array)

    return final_noise_array


def temporal_consecutive_noise_generation(prcp_array, match_acf_array,
                                          window_size, overlap_ratio, ssft_war_thr, seed):
    """
        prcp_array: matched aorc rainfall field
        match_acf_array: matched acf field
        window_size: window size
        overlap_ratio: overlap ratio of window
        ssft_war_thr: rainfall area threshold for FFST
        seed: random seed for noise generation
    """

    # initialize a time series of zero field for final noise
    final_prcp_noise_array = np.zeros(prcp_array.shape)

    # generate random Gaussian noise with the same dimension as the rainfall array
    randstate = np.random
    randstate.seed(seed)
    raw_noise_field = randstate.randn(prcp_array.shape[0], prcp_array.shape[1], prcp_array.shape[2])

    # generate acf noise field
    randstate.seed(seed + 1)
    acf_raw_noise_field = randstate.randn(prcp_array.shape[0], prcp_array.shape[1], prcp_array.shape[2])

    for time_step in range(prcp_array.shape[0]):

        # if it is the first time step
        if time_step == 0:
            # generate correlated random noise
            final_prcp_noise_array[time_step] = FFST_based_noise_generation(field = prcp_array[time_step],
                                                                            noise = raw_noise_field[time_step],
                                                                            win_size = window_size,
                                                                            overlap = overlap_ratio,
                                                                            ssft_war_thr = ssft_war_thr)

        else:

            # get current noise field
            current_noise_field = FFST_based_noise_generation(field=prcp_array[time_step],
                                                                            noise = raw_noise_field[time_step],
                                                                            win_size = window_size,
                                                                            overlap = overlap_ratio,
                                                                            ssft_war_thr = ssft_war_thr)

            # generate acf field, the acf might not share the same white noise as the spatial noise
            curr_acf_array = FFST_based_noise_generation(field=match_acf_array[time_step],
                                                         noise=acf_raw_noise_field[time_step],
                                                         win_size=(prcp_array.shape[1], prcp_array.shape[2]), # use full domain
                                                         overlap=0,
                                                         ssft_war_thr=0.05)

            # normalize the noise value to (-0.99, 0.99)
            # rescale it to (-0.99, 0.99)
            old_min, old_max = np.min(curr_acf_array), np.max(curr_acf_array)
            new_min, new_max = -0.99, 0.99
            rescaled_acf_field = (curr_acf_array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

            # check if the acf is nan, convert to 0
            rescaled_acf_field[np.isnan(rescaled_acf_field)] = 0

            # map no rainfall overlap region to 0
            # rescaled_acf_field = np.where(rainfall_overlap_mask[time_step] == 1, rescaled_acf_field, 0)

            # compute the final noise
            final_prcp_noise_array[time_step] = rescaled_acf_field * final_prcp_noise_array[time_step-1] + np.sqrt(
                1 - rescaled_acf_field ** 2) * current_noise_field


    return final_prcp_noise_array



if __name__ == "__main__":

    # read argument input
    ar_id = int(sys.argv[1])
    realization = int(sys.argv[2])

    # use ar id to generate a series of random numbers
    # generate a seed field
    random.seed(ar_id)
    # generate a list of random integers using ar_id as seed
    seed_list = []
    for i in range(1000):
        seed_list.append(random.randint(1, 2 ** 20))
    # get the seed for this realization
    seed = seed_list[realization]

    # print("Process year {0} month {1}".format(year, month))
    print("AR id: {0}".format(ar_id))
    print("Random state: {0}".format(realization))

    # load precipitation data
    raw_cesm_prcp_xarray = xr.load_dataset("prect_cesm_res.nc".format(ar_id))
    raw_cesm_prcp_array = raw_cesm_prcp_xarray['prect'].data

    # load precipitation data
    aorc_xarray = xr.load_dataset("{0}_sr_rainfall.nc".format(ar_id))
    aorc_array = aorc_xarray['aorc'].data  # the minimum value of this data is already zero

    # load matched lag-1 autocorrelation field
    acf_xarray = xr.load_dataset("{0}_acf_rainfall.nc".format(ar_id))
    acf_array = aorc_xarray['aorc'].data

    # get numpy array
    time_steps = raw_cesm_prcp_xarray['time'].data

    window_size = (64, 64)
    window_function = 'tukey'
    overlap_ratio = 0.3
    ssft_war_thr = 0.05

    # generate a copy
    # aorc_array = np.where(aorc_array < 0, 0, aorc_array) # no need to convert small rainfall values to zero

    final_prcp_noise_array = temporal_consecutive_noise_generation(prcp_array=aorc_array,
                                                                   match_acf_array=acf_array,
                                                                   window_size=window_size,
                                                                   overlap_ratio=overlap_ratio,
                                                                   ssft_war_thr=ssft_war_thr,
                                                                   seed=seed)

    # final_prcp_noise_array = final_prcp_noise_array.astype(np.float32)
    # save the noise field
    # np.save('{0}_{1}_noise.npy'.format(ar_id, realization), final_prcp_noise_array)
    # set up AORC coordinates
    aorc_lat = np.linspace(36.49934, 32.932816, 108)
    aorc_lon = np.linspace(-104.367692, -95.734704, 260)

    # Create the dataset for scipy scale parameter
    noise_ds = xr.Dataset(
        {'aorc': (['time', 'latitude', 'longitude'], final_prcp_noise_array)},
        coords={
            'time': time_steps,
            'latitude': aorc_lat,
            'longitude': aorc_lon
        },
        attrs={'description': "Noise field {0} for AORC AR id {1}".format(realization,
                                                                          ar_id)}
    )

    # save the dataset
    noise_ds.to_netcdf("{0}_{1}_noise.nc".format(ar_id, realization),
                       encoding={"aorc": {"dtype": "float32", "zlib": True}})

