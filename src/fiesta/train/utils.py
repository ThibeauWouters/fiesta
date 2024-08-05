# """Utilities for training the surrogate models, such as reading and preprocessing the input data and more features to be added."""

# import numpy as np
# import jax
# import jax.numpy as jnp 
# from jaxtyping import Array, Float
# import pandas as pd
# import scipy.interpolate as interp
# import tqdm
# import copy
# import re

# #######################
# ### BULLA UTILITIES ###
# #######################

# def get_filters_bulla_file(filename: str,
#                            drop_times: bool = False) -> list[str]:
    
#     assert filename.endswith(".dat"), "File should be of type .dat"
    
#     # Open up the file and read the first line to get the header
#     with open(filename, "r") as f:
#         names = list(filter(None, f.readline().rstrip().strip("#").split(" ")))
#     # Drop the times column if required, to get only the filters
#     if drop_times:
#         names = [name for name in names if name != "t[days]"]
#     # Replace  colons with underscores
#     names = [name.replace(":", "_") for name in names]
    
#     return names

# def get_times_bulla_file(filename: str) -> list[str]:
    
#     assert filename.endswith(".dat"), "File should be of type .dat"
    
#     names = get_filters_bulla_file(filename, drop_times=False)
    
#     data = pd.read_csv(filename, 
#                        delimiter=" ", 
#                        comment="#", 
#                        header=None, 
#                        names=names, 
#                        index_col=False)
    
#     times = data["t[days]"].to_numpy()

#     return times

# def read_single_bulla_file(filename: str) -> dict:
#     """
#     Load lightcurves from Bulla type .dat files

#     Args:
#         filename (str): Name of the file

#     Returns:
#         dict: Dictionary containing the light curve data
#     """
    
#     # Extract the name of the file, without extensions or directories
#     name = filename.split("/")[-1].replace(".dat", "")
#     with open(filename, "r") as f:
#         names = get_filters_bulla_file(filename)
    
#     df = pd.read_csv(
#         filename,
#         delimiter=" ",
#         comment="#",
#         header=None,
#         names=names,
#         index_col=False,
#     )
#     df.rename(columns={"t[days]": "t"}, inplace=True)

#     lc_data = df.to_dict(orient="series")
#     lc_data = {
#         k.replace(":", "_"): v.to_numpy() for k, v in lc_data.items()
#     }
    
#     return lc_data

# ### Models

# # TODO: perhaps move models to a separate file, like in NMMA?

# BU2022YE_PARAMS = ["log10_mej_dyn",
#                    "vej_dyn",
#                    "Yedyn",
#                    "log10_mej_wind",
#                    "vej_wind",
#                    "KNtheta"
# ]

# BULLA_PARAMETER_NAMES = {"Bu2022Ye": BU2022YE_PARAMS}

# ### Bu2022Ye

# def extract_Bu2022Ye_parameters(filename: str) -> np.array:
#     """
#     Extract the parameter values from the filename of a Bulla file

#     Args:
#         filename (str): Bu2022Ye filename, e.g. `./nph1.0e+06_dyn0.005-0.12-0.30_wind0.050-0.03_theta25.84_dMpc0.dat`

#     Returns:
#         dict: Dictionary with the parameter values
#     """
#     # Extract the name like in the example above from the filename
#     name = filename.split("/")[-1].replace(".dat", "")

#     # Skip the first nph value
#     parameters_idx = [1, 2, 3, 4, 5, 6]
    
#     # Use regex to extract the values
#     rr = [
#         np.abs(float(x))
#         for x in re.findall(
#             r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", name
#         )
#     ]

#     # Best to interpolate mass in log10 space
#     rr[1] = np.log10(rr[1])
#     rr[4] = np.log10(rr[4])

#     parameter_values = np.array([rr[idx] for idx in parameters_idx])

#     return parameter_values

# #########################
# ### GENERAL UTILITIES ###
# #########################

# def interpolate_nans(data: dict[str, Float[Array, " n_files n_times"]],
#                      times: Array, 
#                      diagnose: bool = False,
#                      debug: bool = True) -> dict[str, Float[Array, " n_files n_times"]]:
#     """
#     Interpolate NaNs and infs in the raw light curve data. 

#     Args:
#         data (dict[str, Float[Array, 'n_files n_times']]): The raw light curve data
#         diagnose (bool): If True, print out the number of NaNs and infs in the data etc to inform about quality of the grid.

#     Returns:
#         dict[str, Float[Array, 'n_files n_times']]: Raw light curve data but with NaNs and infs interpolated
#     """
    
#     data = copy.deepcopy(data)
    
#     if diagnose:
#         percentages_nans = []
#         percentages_infs = []
    
#     for filt, lc_array in data.items():
        
#         n_files = np.shape(lc_array)[0]
        
#         if filt == "t":
#             continue
        
#         for i in range(n_files):
#             lc = lc_array[i]
#             # Get NaN or inf indices
#             nan_idx = np.isnan(lc)
#             inf_idx = np.isinf(lc)
#             bad_idx = nan_idx | inf_idx
#             good_idx = ~bad_idx
            
#             # TODO: the diagnose for inf is broken, it seems
#             if diagnose:
#                 percentages_nans.append(100 * (np.sum(nan_idx) / len(lc)))
#                 percentages_infs.append(100 * (np.sum(inf_idx) / len(lc))) # broken?
            
#             # Skip LC if there is no NaNs or infs
#             if not any(bad_idx):
#                 continue

#             # Do an interpolation for the bad values and replace it in the data
#             if len(good_idx) > 1:
#                 # Make interpolation routine at the good idx
#                 good_times = times[good_idx]
#                 good_mags = lc[good_idx]
#                 interpolator = interp.interp1d(good_times, good_mags, fill_value="extrapolate")
#                 # Apply it to all times to interpolate
#                 data[filt][i] = interpolator(times)

#         if diagnose:
#             print(f"Filter: {filt}")
#             print(f"Mean percentage of NaNs: {np.mean(percentages_nans)}")
#             print(f"Mean percentage of Infs: {np.mean(percentages_infs)}")
    
#     return data