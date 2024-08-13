import os 
import time
import tqdm
import itertools
import numpy as np 
import matplotlib.pyplot as plt
import afterglowpy as grb

from fiesta.train.SurrogateTrainer import AfterglowpyTrainer
from fiesta.inference.lightcurve_model import AfterglowpyLightcurvemodel
from fiesta.train.neuralnets import NeuralnetConfig
from fiesta.utils import get_default_filts_lambdas
from fiesta.constants import days_to_seconds, pc_to_cm, c
from fiesta.conversions import mJys_to_mag_np

#############
### SETUP ###
#############

FILTERS = ["radio-3GHz"]
FILTERS, lambdas = get_default_filts_lambdas(FILTERS)
nus = c / lambdas
print(FILTERS)
print(nus)

tmin = 1
tmax = 1000

prior_ranges = {
    'inclination_EM': [0.0, np.pi/2],
    'log10_E0': [47.0, 57.0],
    'thetaCore': [0.01, np.pi/10],
    'log10_n0': [-6, 3.0],
    'p': [2.01, 3.0],
    'log10_epsilon_e': [-5, 0],
    'log10_epsilon_B': [-10, 0]
}

# TODO: how is distance handled? What about EM likelihood?
fixed_parameters = {"luminosity_distance": 40.0}


jet_name = "tophat"
jet_conversion = {"tophat": -1,
                  "gaussian": 0,
                  "powerlaw": 4}

name = "tophat_test"
outdir = f"./afterglowpy/{name}/"

###############
### TRAINER ###
###############

# TODO: perhaps also want to train on the full LC, without the SVD?
# TODO: train to output flux, not the mag?
trainer = AfterglowpyTrainer(name,
                             outdir,
                             FILTERS,
                             prior_ranges,
                             n_training_data= 2_000,
                             jet_type = jet_conversion[jet_name],
                             fixed_parameters=fixed_parameters,
                             tmin = tmin,
                             tmax = tmax,
                             plots_dir="./figures/",
                             svd_ncoeff=10,
                             save_raw_data=True,
                             save_preprocessed_data=True
                             )

###############
### FITTING ###
###############

config = NeuralnetConfig(output_size=trainer.svd_ncoeff,
                         nb_epochs=10_000,
                         layer_sizes = [128, 256, 128])

trainer.fit(config=config)
trainer.save()

#############
### TEST ###
#############

print("Producing example lightcurve . . .")

lc_model = AfterglowpyLightcurvemodel(name,
                                      outdir, 
                                      filters = FILTERS)

for filt in lc_model.filters:
    X_example = trainer.X_raw[0]
    y_raw = trainer.y_raw[filt][0]
    
    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt]
    
    plt.plot(lc_model.times, y_raw, color = "red", label="afterglowpy")
    plt.plot(lc_model.times, y_predict, color = "blue", label="Surrogate prediction")
    upper_bound = y_predict + 1
    lower_bound = y_predict - 1
    plt.fill_between(lc_model.times, lower_bound, upper_bound, color='blue', alpha=0.2)

    plt.ylabel(f"mag for {filt}")
    plt.legend()
    plt.gca().invert_yaxis()

    plt.savefig(f"./figures/afterglowpy_{name}_{filt}_example.png")
    plt.close()
    break # only show first filter