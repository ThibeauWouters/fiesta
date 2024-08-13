"""Store classes to load in trained models and give routines to let them generate lightcurves."""

# TODO: improve them with jax treemaps, since dicts are essentially pytrees

import os
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from beartype import beartype as typechecker
from flax.training.train_state import TrainState
import joblib

import fiesta.train.neuralnets as fiesta_nn
from fiesta.utils import MinMaxScalerJax, inverse_svd_transform
from fiesta.conversions import mJys_to_mag_jnp
from fiesta import models_utilities

########################
### ABSTRACT CLASSES ###
########################

class LightcurveModel:
    """Abstract class for general light curve models"""
    
    name: str 
    filters: list[str]
    parameter_names: list[str]
    times: Array
    
    def __init__(self, 
                 name: str) -> None:
        self.name = name
        self.filters = []
        self.parameter_names = []
        self.times = jnp.array([])
    
    def add_name(self, x: Array):
        return dict(zip(self.parameter_names, x))    
    
    def project_input(self, x: Array) -> dict[str, Array]:
        """
        Project the given input to whatever preprocessed input space we are in. 
        By default (i.e., in this base class), the projection is the identity function.

        Args:
            x (Array): Input array

        Returns:
            Array: Input array transformed to the preprocessed space.
        """
        return x
    
    def compute_output(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Compute the output (untransformed) from the given, transformed input. 
        This is the main method that needs to be implemented by subclasses.

        Args:
            x (Array): Input array

        Returns:
            Array: Output array
        """
        raise NotImplementedError
        
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Project the computed output to whatever preprocessed output space we are in. 
        By default (i.e., in this base class), the projection is the identity function.

        Args:
            y (Array): Output array

        Returns:
            Array: Output array transformed to the preprocessed space.
        """
        return y
    
    @partial(jax.jit, static_argnums=(0,))
    def predict(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Generate the lightcurve y from the unnormalized and untransformed input x.
        Chains the projections with the actual computation of the output. E.g. if the model is a trained
        surrogate neural network, they represent the map from x tilde to y tilde. The mappings from
        x to x tilde and y to y tilde take care of projections (e.g. SVD projections) and normalizations.

        Args:
            x (Array): Input array, unnormalized and untransformed.

        Returns:
            Array: Output dict[str, Array], i.e., the desired raw light curve per filter
        """
        
        # Use saved parameter names to extract the parameters in the correct order into an array
        x_array = jnp.array([x[name] for name in self.parameter_names])
        x_tilde = self.project_input(x_array)
        y_tilde = self.compute_output(x_tilde)
        y = self.project_output(y_tilde)
        return y
    
    def __repr__(self) -> str:
        return self.name
    
class SurrogateLightcurveModel(LightcurveModel):
    """Abstract class for models that rely on a surrogate, in the form of a neural network."""
    
    directory: str
    metadata: dict
    X_scaler: MinMaxScalerJax
    y_scaler: dict[str, MinMaxScalerJax]
    models: dict[str, TrainState]
    times: Array
    tmin: Float
    tmax: Float
    parameter_names: list[str]
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None) -> None:
        """_summary_

        Args:
            name (str): Name of the model
            directory (str): Directory with trained model states and projection metadata such as scalers.
            filters (list[str]): List of all the filters for which the model should be loaded.
        """
        super().__init__(name)
        self.directory = directory
        self.models = {}
        
        # Load the metadata for projections etc
        self.load_filters(filters)
        self.load_metadata()
        self.load_scalers()
        self.load_times(times)
        self.load_parameter_names()
        self.load_networks()
        
    def load_metadata(self) -> None:
        self.metadata_filename = os.path.join(self.directory, f"{self.name}.joblib")
        assert os.path.exists(self.metadata_filename), f"Metadata file {self.metadata_filename} not found - check the directory {self.directory}"
        self.metadata = joblib.load(self.metadata_filename)
        
    def load_filters(self, filters: list[str] = None) -> None:
        # Save those filters that were given and that were trained and store here already
        pkl_files = [file for file in os.listdir(self.directory) if file.endswith(".pkl") or file.endswith(".pickle")]
        all_available_filters = [file.split(".")[0] for file in pkl_files]
        
        if filters is None:
            # Use all filters that the surrogate model supports
            filters = all_available_filters
        else:
            # Fetch those filters specified by the user that are available
            filters = [f.replace(":", "_") for f in filters]
            filters = [f for f in filters if f in all_available_filters]
        
        if len(filters) == 0:
            raise ValueError(f"No filters found in {self.directory} that match the given filters {filters}")
        self.filters = filters
        print(f"Loaded SurrogateLightcurveModel with filters {filters}")
        
    def load_scalers(self):
        min_val, max_val = self.metadata["X_scaler_min"], self.metadata["X_scaler_max"]
        self.X_scaler = MinMaxScalerJax(min_val=min_val, max_val=max_val)
        
        min_val, max_val = self.metadata["y_scaler_min"], self.metadata["y_scaler_max"]
        self.y_scaler = {}
        for filt in self.filters:
            self.y_scaler[filt] = MinMaxScalerJax(min_val=min_val[filt], max_val=max_val[filt])
            
    def load_times(self, times: Array = None) -> None:
        # TODO: check for time range and trained model time range
        if times is None:
            times = jnp.array(self.metadata["times"])
        self.times = times
        self.tmin = jnp.min(times)
        self.tmax = jnp.max(times)
        
    def load_networks(self) -> None:
        self.models = {}
        for filter in self.filters:
            filename = os.path.join(self.directory, f"{filter}.pkl")
            state, _ = fiesta_nn.load_model(filename)
            self.models[filter] = state
        
    def load_parameter_names(self) -> None:
        """Implement in child classes"""
        raise NotImplementedError
    
    def project_input(self, x: Array) -> dict[str, Array]:
        """
        Project the given input to whatever preprocessed input space we are in.

        Args:
            x (dict[str, Array]): Original input array

        Returns:
            dict[str, Array]: Transformed input array
        """
        x_tilde = {filter: self.X_scaler.transform(x) for filter in self.filters}
        return x_tilde
    
    def compute_output(self, x: dict[str, Array]) -> dict[str, Array]:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters per filter

        Returns:
            dict[str, Array]: _description_
        """
        # TODO: too convoluted, simplify
        return {filter: self.models[filter].apply_fn({'params': self.models[filter].params}, x[filter]) for filter in self.filters}
        
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Project the computed output to whatever preprocessed output space we are in.

        Args:
            y (dict[str, Array]): Output array

        Returns:
            dict[str, Array]: Output array transformed to the preprocessed space.
        """
        return {filter: self.y_scaler[filter].inverse_transform(y[filter]) for filter in self.filters}
        
    
class SVDSurrogateLightcurveModel(SurrogateLightcurveModel):
    
    VA: dict[str, Array]
    nsvd_coeff: int
    
    def __init__(self, 
                 name: str, 
                 directory: str,
                 times: Array = None,
                 filters: list[str] = None):
        """
        Initialize a class to generate lightcurves from a Bulla trained model.
        
        """
        super().__init__(name=name, directory=directory, filters=filters, times=times)
        
        self.VA = self.metadata["VA"]
        self.nsvd_coeff = self.metadata["nsvd_coeff"]
        
    def load_parameter_names(self) -> None:
        raise NotImplementedError
            
    def project_output(self, y: dict[str, Array]) -> dict[str, Array]:
        """
        Apply the trained flax neural network on the given input x.

        Args:
            x (dict[str, Array]): Input array of parameters

        Returns:
            dict[str, Array]: _description_
        """
        output = {filter: inverse_svd_transform(y[filter], self.VA[filter], self.nsvd_coeff) for filter in self.filters}
        return super().project_output(output)
       
class BullaLightcurveModel(SVDSurrogateLightcurveModel):
    
    def __init__(self, 
                 name: str, 
                 directory: str,
                 times: Array = None,
                 filters: list[str] = None):
        
        super().__init__(name=name, directory=directory, times=times, filters=filters) 
        
    def load_parameter_names(self) -> None:
        self.parameter_names = models_utilities.BULLA_PARAMETER_NAMES[self.name]
    
class AfterglowpyLightcurvemodel(SVDSurrogateLightcurveModel):
    
    def __init__(self,
                 name: str,
                 directory: str,
                 filters: list[str] = None,
                 times: Array = None):
        super().__init__(name=name, directory=directory, filters=filters, times=times)
        
    def load_parameter_names(self) -> None:
        self.parameter_names = self.metadata["parameter_names"]