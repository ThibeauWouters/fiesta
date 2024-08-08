"""Utilities regarding (surrogate) models."""
import jax
import jax.numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, Float
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import copy
import re
from sncosmo.bandpasses import _BANDPASSES
from astropy.time import Time

####################
### BULLA MODELS ###
####################

### Bu2022Ye ###

BU2022YE_PARAMETER_NAMES = ["log10_mej_dyn",
                   "vej_dyn",
                   "Yedyn",
                   "log10_mej_wind",
                   "vej_wind",
                   "KNtheta"
]

def extract_Bu2022Ye_parameters(filename: str) -> np.array:
    """
    Extract the parameter values from the filename of a Bulla file

    Args:
        filename (str): Bu2022Ye filename, e.g. `./nph1.0e+06_dyn0.005-0.12-0.30_wind0.050-0.03_theta25.84_dMpc0.dat`

    Returns:
        np.array: Array with the parameter values for this filename
    """
    # Extract the name like in the example above from the filename
    name = filename.split("/")[-1].replace(".dat", "")

    # Skip the first nph value
    parameters_idx = [1, 2, 3, 4, 5, 6]
    
    # Use regex to extract the values
    rr = [
        np.abs(float(x))
        for x in re.findall(
            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", name
        )
    ]

    # Best to interpolate mass in log10 space
    rr[1] = np.log10(rr[1])
    rr[4] = np.log10(rr[4])

    parameter_values = np.array([rr[idx] for idx in parameters_idx])

    return parameter_values

### Bu2019lm ###

BU2019LM_PARAMETER_NAMES = ["log10_mej_dyn", 
                            "log10_mej_wind", 
                            "KNphi", 
                            "KNtheta"]

def extract_Bu2019lm_parameters(filename: str) -> dict:
    """
    Extract the parameter values from the filename of a Bulla file

    Args:
        filename (str): Bu2019lm filename, e.g. `./nph1.0e+06_mejdyn0.005_mejwind0.130_phi45_theta84.26_dMpc0.dat`

    Returns:
        np.array: Array with the parameter values for this filename
    """

    # Skip nph entry
    parameters_idx = [1, 2, 3, 4]
    rr = [
        float(x)
        for x in re.findall(
            r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", filename.replace("nph1.0e+06_", "")
        )
    ]

    # Best to interpolate mass in log10 space
    rr[1] = np.log10(rr[1])
    rr[2] = np.log10(rr[2])

    parameter_values = np.array([rr[idx] for idx in parameters_idx])

    return parameter_values


BULLA_PARAMETER_NAMES = {"Bu2022Ye": BU2022YE_PARAMETER_NAMES,
                         "Bu2019lm": BU2019LM_PARAMETER_NAMES}
SUPPORTED_BULLA_MODELS = list(BULLA_PARAMETER_NAMES.keys())

EXTRACT_PARAMETERS_FUNCTIONS = {"Bu2022Ye": extract_Bu2022Ye_parameters,
                                "Bu2019lm": extract_Bu2019lm_parameters}

### All

# TODO: update this dict to include any other models
PARAMETER_NAMES = BULLA_PARAMETER_NAMES
SUPPORTED_MODELS = list(PARAMETER_NAMES.keys())