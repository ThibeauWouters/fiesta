import jax
import jax.numpy as jnp
from jax.scipy.stats import truncnorm
from jaxtyping import Array, Float
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import copy
import re
from astropy.time import Time
from fiesta.constants import pc_to_cm
import astropy
import scipy
from sncosmo.bandpasses import _BANDPASSES, _BANDPASS_INTERPOLATORS
import sncosmo

class MinMaxScalerJax(object):
    """
    MinMaxScaler like sklearn does it, but for JAX arrays since sklearn might not be JAX-compatible?
    
    Note: assumes that input has dynamical range: it will not catch errors due to constant input (leading to zero division)
    """
    
    def __init__(self,
                 min_val: Array = None,
                 max_val: Array = None):
        
        self.min_val = min_val
        self.max_val = max_val
    
    def fit(self, x: Array) -> None:
        self.min_val = jnp.min(x, axis=0)
        self.max_val = jnp.max(x, axis=0)
        
    def transform(self, x: Array) -> Array:
        return (x - self.min_val) / (self.max_val - self.min_val)
    
    def inverse_transform(self, x: Array) -> Array:
        return x * (self.max_val - self.min_val) + self.min_val
    
    def fit_transform(self, x: Array) -> Array:
        self.fit(x)
        return self.transform(x)
    
def inverse_svd_transform(x: Array, 
                          VA: Array, 
                          nsvd_coeff: int = 10) -> Array:

    # TODO: check the shapes etc, transforms and those things
    return jnp.dot(VA[:, :nsvd_coeff], x)

# @jax.jit
# TODO: change place!
def mag_app_from_mag_abs(mag_abs: Array,
                         luminosity_distance: Float) -> Array:
    return mag_abs + 5.0 * jnp.log10(luminosity_distance * 1e6 / 10.0)


#######################
### BULLA UTILITIES ###
#######################

def get_filters_bulla_file(filename: str,
                           drop_times: bool = False) -> list[str]:
    
    assert filename.endswith(".dat"), "File should be of type .dat"
    
    # Open up the file and read the first line to get the header
    with open(filename, "r") as f:
        names = list(filter(None, f.readline().rstrip().strip("#").split(" ")))
    # Drop the times column if required, to get only the filters
    if drop_times:
        names = [name for name in names if name != "t[days]"]
    # Replace  colons with underscores
    names = [name.replace(":", "_") for name in names]
    
    return names

def get_times_bulla_file(filename: str) -> list[str]:
    
    assert filename.endswith(".dat"), "File should be of type .dat"
    
    names = get_filters_bulla_file(filename, drop_times=False)
    
    data = pd.read_csv(filename, 
                       delimiter=" ", 
                       comment="#", 
                       header=None, 
                       names=names, 
                       index_col=False)
    
    times = data["t[days]"].to_numpy()

    return times

def read_single_bulla_file(filename: str) -> dict:
    """
    Load lightcurves from Bulla type .dat files

    Args:
        filename (str): Name of the file

    Returns:
        dict: Dictionary containing the light curve data
    """
    
    # Extract the name of the file, without extensions or directories
    name = filename.split("/")[-1].replace(".dat", "")
    with open(filename, "r") as f:
        names = get_filters_bulla_file(filename)
    
    df = pd.read_csv(
        filename,
        delimiter=" ",
        comment="#",
        header=None,
        names=names,
        index_col=False,
    )
    df.rename(columns={"t[days]": "t"}, inplace=True)

    lc_data = df.to_dict(orient="series")
    lc_data = {
        k.replace(":", "_"): v.to_numpy() for k, v in lc_data.items()
    }
    
    return lc_data



#########################
### GENERAL UTILITIES ###
#########################

def interpolate_nans(data: dict[str, Float[Array, " n_files n_times"]],
                     times: Array, 
                     output_times: Array = None) -> dict[str, Float[Array, " n_files n_times"]]:
    """
    Interpolate NaNs and infs in the raw light curve data. 

    Args:
        data (dict[str, Float[Array, 'n_files n_times']]): The raw light curve data
        diagnose (bool): If True, print out the number of NaNs and infs in the data etc to inform about quality of the grid.

    Returns:
        dict[str, Float[Array, 'n_files n_times']]: Raw light curve data but with NaNs and infs interpolated
    """
    
    if output_times is None:
        output_times = times
    
    # TODO: improve this function overall!
    copy_data = copy.deepcopy(data)
    output = {}
    
    for filt, lc_array in copy_data.items():
        
        n_files = np.shape(lc_array)[0]
        
        if filt == "t":
            continue
        
        for i in range(n_files):
            lc = lc_array[i]
            # Get NaN or inf indices
            nan_idx = np.isnan(lc)
            inf_idx = np.isinf(lc)
            bad_idx = nan_idx | inf_idx
            good_idx = ~bad_idx
            
            # Interpolate through good values on given time grid
            if len(good_idx) > 1:
                # Make interpolation routine at the good idx
                good_times = times[good_idx]
                good_mags = lc[good_idx]
                interpolator = interp.interp1d(good_times, good_mags, fill_value="extrapolate")
                # Apply it to all times to interpolate
                mag_interp = interpolator(output_times)
                
            else:
                raise ValueError("No good values to interpolate from")
            
            if filt in output:
                output[filt] = np.vstack((output[filt], mag_interp))
            else:
                output[filt] = np.array(mag_interp)

    return output

def truncated_gaussian(mag_det: Array, 
                       mag_err: Array, 
                       mag_est: Array, 
                       lim: Float = jnp.inf):
    
    """
    Evaluate log PDF of a truncated Gaussian with loc at mag_est and scale mag_err, truncated at lim above.

    Returns:
        _type_: _description_
    """
    
    loc, scale = mag_est, mag_err
    a_trunc = -999 # TODO: OK if we just fix this to a large number, to avoid infs?
    a, b = (a_trunc - loc) / scale, (lim - loc) / scale
    logpdf = truncnorm.logpdf(mag_det, a, b, loc=loc, scale=scale)
    return logpdf

def load_event_data(filename):
    # TODO: polish?
    lines = [line.rstrip("\n") for line in open(filename)]
    lines = filter(None, lines)

    sncosmo_filts = [val["name"] for val in _BANDPASSES.get_loaders_metadata()]
    sncosmo_maps = {name: name.replace(":", "_") for name in sncosmo_filts}

    data = {}
    for line in lines:
        lineSplit = line.split(" ")
        lineSplit = list(filter(None, lineSplit))
        mjd = Time(lineSplit[0], format="isot").mjd
        filt = lineSplit[1]

        if filt in sncosmo_maps:
            filt = sncosmo_maps[filt]

        mag = float(lineSplit[2])
        dmag = float(lineSplit[3])

        if filt not in data:
            data[filt] = np.empty((0, 3), float)
        data[filt] = np.append(data[filt], np.array([[mjd, mag, dmag]]), axis=0)

    return data

#########################
### Filters           ###
#########################


class Filter:

    def __init__(self,
                 name: str,):
        self.name = name
        if (self.name, None) in _BANDPASSES._primary_loaders:
            bandpass = sncosmo.get_bandpass(self.name)
            self.nu = scipy.constants.c/(bandpass.wave_eff*1e-10)
        elif (self.name, None) in _BANDPASS_INTERPOLATORS._primary_loaders:
            bandpass = sncosmo.get_bandpass(val["name"], 3)
            self.nu = scipy.constants.c/(bandpass.wave_eff*1e-10)
        elif self.name.endswith("GHz"):
            freq = re.findall(r"[-+]?(?:\d*\.*\d+)", self.name.replace("-",""))
            freq = float(freq[-1])
            self.nu = freq*1e9
        elif self.name.endswith("keV"):
            energy = re.findall(r"[-+]?(?:\d*\.*\d+)", self.name.replace("-",""))
            energy = float(energy[-1])
            self.nu = energy*1000*scipy.constants.eV / scipy.constants.h
        else:
            print(f"Warning: Filter {self.name} not recognized, setting nu to NaN for this filter")
            self.nu = jnp.nan
            
        self.wavelength = scipy.constants.c/self.nu


def get_all_bandpass_metadata():
    # TODO: taken over from NMMA, improve
    """
    Retrieves and combines the metadata for all registered bandpasses and interpolators.

    Returns:
        list: Combined list of metadata dictionaries from bandpasses and interpolators for sncosmo.
    """

    bandpass_metadata = _BANDPASSES.get_loaders_metadata()
    interpolator_metadata = _BANDPASS_INTERPOLATORS.get_loaders_metadata()

    combined_metadata = bandpass_metadata + interpolator_metadata

    return combined_metadata

def get_default_filts_lambdas(filters: list[str]=None):

    filts = [
        "u",
        "g",
        "r",
        "i",
        "z",
        "y",
        "J",
        "H",
        "K",
        "U",
        "B",
        "V",
        "R",
        "I",
        "radio-1.25GHz",
        "radio-3GHz",
        "radio-5.5GHz",
        "radio-6GHz",
        "X-ray-1keV",
        "X-ray-5keV",
    ]
    lambdas_sloan = 1e-10 * np.array(
        [3561.8, 4866.46, 6214.6, 7687.0, 7127.0, 7544.6, 8679.5, 9633.3, 12350.0]
    )
    lambdas_bessel = 1e-10 * np.array([3605.07, 4413.08, 5512.12, 6585.91, 8059.88])
    lambdas_radio = scipy.constants.c / np.array([1.25e9, 3e9, 5.5e9, 6e9])
    lambdas_Xray = scipy.constants.c / (
        np.array([1e3, 5e3]) * scipy.constants.eV / scipy.constants.h
    )

    lambdas = np.concatenate(
        [lambdas_sloan, lambdas_bessel, lambdas_radio, lambdas_Xray]
    )

    bandpasses = []
    for val in get_all_bandpass_metadata():
        if val["name"] in [
            "ultrasat",
            "megacampsf::u",
            "megacampsf::g",
            "megacampsf::r",
            "megacampsf::i",
            "megacampsf::z",
            "megacampsf::y",
        ]:
            bandpass = sncosmo.get_bandpass(val["name"], 3)
            bandpass.name = bandpass.name.split()[0]
        else:
            bandpass = sncosmo.get_bandpass(val["name"])

        bandpasses.append(bandpass)

    filts = filts + [band.name for band in bandpasses]
    lambdas = np.concatenate([lambdas, [1e-10 * band.wave_eff for band in bandpasses]])

    if filters is not None:
        filts_slice = []
        lambdas_slice = []
        transmittance_slice = []

        for filt in filters:
            if filt.startswith("radio") and filt not in filts:
                # for additional radio filters that not in the list
                # calculate the lambdas based on the filter name
                # split the filter name
                freq_string = filt.replace("radio-", "")
                freq_unit = freq_string[-3:]
                freq_val = float(freq_string.replace(freq_unit, ""))
                # make use of the astropy.units to be more flexible
                freq = astropy.units.Quantity(freq_val, unit=freq_unit)
                freq = freq.to("Hz").value
                # adding to the list
                filts_slice.append(filt)
                lambdas_slice.append([scipy.constants.c / freq])
                transmittance_slice.append([1])

            elif filt.startswith("X-ray-") and filt not in filts:
                # for additional X-ray filters that not in the list
                # calculate the lambdas based on the filter name
                # split the filter name
                energy_string = filt.replace("X-ray-", "")
                energy_unit = energy_string[-3:]
                energy_val = float(energy_string.replace(energy_unit, ""))
                # make use of the astropy.units to be more flexible
                energy = astropy.units.Quantity(energy_val, unit=energy_unit)
                freq = energy.to("eV").value * scipy.constants.eV / scipy.constants.h
                # adding to the list
                filts_slice.append(filt)
                lambdas_slice.append([scipy.constants.c / freq])
                transmittance_slice.append([1])

            else:
                try:
                    ii = filts.index(filt)
                    filts_slice.append(filts[ii])
                    lambdas_slice.append([lambdas[ii]])
                except ValueError:
                    ii = filts.index(filt.replace("_", ":"))
                    filts_slice.append(filts[ii].replace(":", "_"))
                    lambdas_slice.append(lambdas[ii])

        filts = filts_slice
        lambdas = np.array(lambdas_slice)

    return filts, lambdas, transmittance


def mJys_to_mag():
    pass