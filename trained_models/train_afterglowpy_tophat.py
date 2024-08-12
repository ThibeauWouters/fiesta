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
fixed_parameters = {"luminosity_distance": 40.0}

###############
### TRAINER ###
###############

jet_name = "tophat"
print(f"Making {jet_name} jet surrogate model . . .")
jet_conversion = {"tophat": -1,
                  "gaussian": 0,
                  "powerlaw": 4}

outdir = f"./afterglowpy/{jet_name}/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

trainer = AfterglowpyTrainer(jet_name,
                             outdir,
                             prior_ranges,
                             FILTERS,
                             fixed_parameters=fixed_parameters,
                             jet_type = jet_conversion[jet_name],
                             tmin = tmin,
                             tmax = tmax,
                             n_grid = 3,
                             plots_dir="./figures/",
                             save_data=True,
                             load_data=False
                             )


config = NeuralnetConfig(output_size=len(trainer.times),
                         nb_epochs=100,)

trainer.fit(config=config)

trainer.save()


########################
### LIGHTCURVE MODEL ###
########################

print("Producing example lightcurve . . .")

lc_model = AfterglowpyLightcurvemodel("tophat",
                                      outdir, 
                                      filters = FILTERS)

times = lc_model.times

for filt in lc_model.filters:
    X_example = trainer.X_raw[0]
    print("X_example")
    print(X_example)
    y_raw = trainer.y_raw[filt][0]
    
    # Turn into a dict: this is how the model expects the input
    X_example = {k: v for k, v in zip(lc_model.parameter_names, X_example)}
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt]
    
    plt.plot(times, y_raw, label="afterglowpy")
    plt.plot(times, y_predict, label="Surrogate prediction")
    plt.ylabel(f"mag for {filt}")
    plt.legend()
    plt.savefig(f"./figures/afterglowpy_{jet_name}_{filt}_example_prediction.png")
    plt.show()
    break # to only show the first filter
