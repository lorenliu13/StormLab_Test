# Extract the lag-1 autocorrelation from AORC data
import numpy as np
import xarray as xr
import os
from scipy import interpolate
import warnings


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


def calculate_lag_one_array(year):
    """
    Calculate the lag-1 autocorrelation function at each time step of the AORC rainfall field. The calculation uses
    AORC fields at time step t and t+1. First, the AORC field is divided into regions of local windows. Second, the
    method calculates the correlation coefficient of precipitation data in one local window between time t and t + 1.
    This correlation coefficient is used as the lag-1 autocorrelation in this window. Third, the calculated lag-1
    autocorrelations in the local windows are interpolated to the original AORC coordinates. This gives the lag-1
    autocorrelation of the AORC field at time step t.
    :param year:
    :return:
    """

    warnings.filterwarnings("ignore")

    print(f"Start to process {year}")
    # load one year of AORC rainfall field
    aorc_rainfall_xarray = xr.load_dataset(
        fr"/home/yliu2232/model_based_pmp/denison_dam/aorc/6h_annual_series/aorc_year_{year}_aorc_res_clip.nc")
    aorc_rainfall_array = aorc_rainfall_xarray['aorc'].data

    # set up AORC coordinates
    aorc_lat = aorc_rainfall_xarray['latitude'].data
    aorc_lon = aorc_rainfall_xarray['longitude'].data

    # create an empty array
    full_autocorrelation_array = np.zeros(aorc_rainfall_array.shape)

    # from the start
    for time_index in range(aorc_rainfall_array.shape[0] - 1): # skip the last time step, the lag-1 autocorrelation of the last time step is set as zero

        # current and next time step
        curr_aorc_array = aorc_rainfall_array[time_index]
        next_aorc_array = aorc_rainfall_array[time_index + 1]

        win_size = (4, 4)
        overlap = 0

        dim_col = curr_aorc_array.shape[1]  # get the column number
        dim_row = curr_aorc_array.shape[0]  # get the row number

        # number of windows
        num_windows_row = int(np.ceil(float(dim_row) / win_size[0]))  # along row direction
        num_windows_col = int(np.ceil(float(dim_col) / win_size[1]))  # along column direction

        window_center_lon_list = []
        window_center_lat_list = []
        window_center_corr_list = []
        window_center_rainfall_list = []

        window_center_next_rainfall_list = []

        # plt.figure(figsize = (10, 5))
        # loop rows
        for i in range(num_windows_row):
            # loop columns: this performs row-major looping, which is faster
            for j in range(num_windows_col):
                # prepare indices
                idxi = np.zeros(2).astype(int)  # this is row index
                idxj = np.zeros(2).astype(int)  # this is col index

                # compute indices of local window
                idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))  # get the upper y index
                idxi[1] = int(np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_row)))  # get the lower y index

                idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))  # get the left x index
                idxj[1] = int(np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_col)))  # get the right x index

                # calculate the lat index
                mean_lat_index = np.mean(aorc_lat[idxi[0]:idxi[1]])  # get average lat coord
                mean_lon_index = np.mean(aorc_lon[idxj[0]:idxj[1]])  # get average lon coord

                subset_curr_aorc_array = curr_aorc_array[idxi[0]:idxi[1], idxj[0]: idxj[1]]
                subset_next_aorc_array = next_aorc_array[idxi[0]:idxi[1], idxj[0]: idxj[1]]

                # calculate the correlation coefficient
                correlation_coefficient = np.corrcoef(subset_curr_aorc_array.flatten(),
                                                      subset_next_aorc_array.flatten())
                correlation_coefficient = correlation_coefficient[0, 1]

                # calculate the average precp in the window
                subset_curr_aorc_rainfall = np.mean(subset_curr_aorc_array)
                subset_next_aorc_rainfall = np.mean(subset_next_aorc_array)

                window_center_lon_list.append(mean_lon_index)
                window_center_lat_list.append(mean_lat_index)
                window_center_corr_list.append(correlation_coefficient)
                window_center_rainfall_list.append(subset_curr_aorc_rainfall)
                window_center_next_rainfall_list.append(subset_next_aorc_rainfall)

        window_center_lon_list = np.array(window_center_lon_list)
        window_center_lat_list = np.array(window_center_lat_list)
        window_center_corr_list = np.array(window_center_corr_list)
        # window_center_rainfall_list = np.array(window_center_rainfall_list)
        # window_center_next_rainfall_list = np.array(window_center_next_rainfall_list)

        # get the array of window coordinates and correlation
        window_lon_array = window_center_lon_list.reshape(num_windows_row, num_windows_col)
        window_lat_array = window_center_lat_list.reshape(num_windows_row, num_windows_col)
        window_corr_array = window_center_corr_list.reshape(num_windows_row, num_windows_col)

        window_lon_coords = window_lon_array[0, :]
        window_lat_coords = window_lat_array[:, 0]

        # interpolate to aorc coordinates
        # flip the y
        x = np.flip(window_lat_coords)  # along row
        y = window_lon_coords  # along column
        data = np.flip(window_corr_array, axis=0).copy()
        # set nan part as zero
        data[np.isnan(data)] = 0
        interp = interpolate.RegularGridInterpolator((x, y), data, method='linear', bounds_error=False, fill_value=None)

        # create meshgrid for new coordinates
        new_x = np.flip(aorc_lat)  # new x (latitude coordinates)
        new_y = aorc_lon  # new y (longitude coordinates)

        new_yy, new_xx = np.meshgrid(new_y, new_x)  # for mesh grids, it takes (longitude, latitude)

        # Interpolate values
        new_acf_array = interp((new_xx, new_yy))  # for interpolation, it takes (latitude, longitude)

        # flip along the row axis
        new_acf_array = np.flip(new_acf_array, axis=0)

        # crop by -1 and 1
        new_acf_array[new_acf_array >= 1] = 0.99
        new_acf_array[new_acf_array <= -1] = -0.99

        # append it
        full_autocorrelation_array[time_index] = new_acf_array

    # save it
    save_folder = fr"/home/yliu2232/model_based_pmp/denison_dam/aorc/6h_annual_series_acf"
    create_folder(save_folder)

    aorc_rainfall_xarray['aorc'].data = full_autocorrelation_array
    # np.save(save_folder + '/' + f"aorc_year_{year}_lag_one_coef_aorc_res.npy", full_autocorrelation_array)
    aorc_rainfall_xarray.to_netcdf(r"/home/yliu2232/model_based_pmp/denison_dam/aorc/6h_annual_series_acf" + "/" + f"aorc_year_{year}_spatial_window_acf_aorc_res.nc",
                             encoding={"aorc": {"dtype": "float32", "zlib": True}})


if __name__ == "__main__":

    year = 2020
    calculate_lag_one_array(year)