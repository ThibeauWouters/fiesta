"""Utilities for training the surrogate models, such as reading and preprocessing the input data and more features to be added."""

import numpy as np
import jax
import jax.numpy as jnp 
from jaxtyping import Array
import pandas as pd

DEBUG = True

def get_filters_bulla_file(filename: str,
                           drop_times: bool = False) -> list[str]:
    
    with open(filename, "r") as f:
        names = list(filter(None, f.readline().rstrip().strip("#").split(" ")))
    if drop_times:
        names = [name for name in names if name != "t[days]"]
    names = [name.replace(":", "_") for name in names]
    
    return names

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
    
    if DEBUG:
        print(f"Reading file: {filename}")
        print(f"name: {name}")
    
    with open(filename, "r") as f:
        names = get_filters_bulla_file(filename)
        if DEBUG:
            print("column names")
            print(names)
    
    df = pd.read_csv(
        filename,
        delimiter=" ",
        comment="#",
        # header=None,
        names=names,
        index_col=False,
    )
    df.rename(columns={"t[days]": "t"}, inplace=True)

    lc_data = df.to_dict(orient="series")
    lc_data = {
        k.replace(":", "_"): v.to_numpy() for k, v in lc_data.items()
    }
    
    if DEBUG:
        print("lc_data")
        print(lc_data)

    return lc_data