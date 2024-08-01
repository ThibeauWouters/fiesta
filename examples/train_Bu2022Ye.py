import os
import numpy as np
import matplotlib.pyplot as plt

from fiesta.train.SurrogateTrainer import BullaSurrogateTrainer
from fiesta.inference.lightcurve_model import BullaLightcurveModel

from fiesta.train import utils
from fiesta.train.neuralnets import NeuralnetConfig

print("Checking whether we found a GPU:")
import jax
print(jax.devices())

params = {"axes.grid": True,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

FILTERS = ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y", "2massj", "2massh", "2massks", "sdssu"]

# TODO: need to find a way to locate the files
lc_dir = "/home/urash/twouters/KN_lightcurves/lightcurves/bulla_2022/"

outdir = "../trained_models/Bu2022Ye/"
plots_dir = "./figures/" # to make plots
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
if not os.path.exists(outdir):
    os.makedirs(outdir)

lc_files = os.listdir(lc_dir)
example_file = os.path.join(lc_dir, lc_files[0])    

print("example_filename")
print(example_file)

###############
### TRAINER ###
###############

print("Defining trainer object, will take around 2 minutes for loading and preprocessing")

bulla_trainer = BullaSurrogateTrainer("Bu2022Ye", 
                                      lc_dir, 
                                      outdir, 
                                      filters = FILTERS,
                                      plots_dir = plots_dir)
full_lc_files = [os.path.join(lc_dir, f) for f in lc_files]

for filename in full_lc_files[:4]:
    print("Example fetching parameters from filename:")
    print("filename")
    print(filename)
    p = utils.extract_Bu2022Ye_parameters(filename)
    print("Parameters extracted")
    print(p)
    
# TODO: show a more elaborate example to show how to modify the config for training and specify architecture etc

# Define the config if you want to change a default parameter
config = NeuralnetConfig()
print(f"Original number of training epochs: {config.nb_epochs}")
config.nb_epochs = 2_000
print(f"New number of training epochs: {config.nb_epochs}")
bulla_trainer.fit(config=config, verbose=True)
bulla_trainer.save()

########################
### LIGHTCURVE MODEL ###
########################

print("Producing example lightcurve . . .")

lc_model = BullaLightcurveModel("Bu2022Ye", 
                                 outdir, 
                                 filters = FILTERS)

times = bulla_trainer.times

for filt in bulla_trainer.filters:
    X_example = bulla_trainer.X_raw[0]
    y_raw = bulla_trainer.y_raw[filt][0]
    
    # Get the prediction lightcurve
    y_predict = lc_model.predict(X_example)[filt]
    
    plt.plot(times, y_raw, label="POSSIS")
    plt.plot(times, y_predict, label="Surrogate prediction")
    plt.ylabel(f"mag for {filt}")
    plt.legend()
    plt.savefig(f"./figures/{filt}_example_prediction.png")
    plt.show()
    break # to only show the first filter