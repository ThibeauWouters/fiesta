import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax 
print(f"GPU found? {jax.devices()}")
import jax.numpy as jnp
# jax.config.update("jax_enable_x64", True)
import jax.random as random
import numpy as np
import corner
import matplotlib.pyplot as plt

from fiesta.inference.lightcurve_model import BullaLightcurveModel
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.prior import Uniform, Composite
from fiesta.inference.fiesta import Fiesta
from fiesta.utils import load_event_data

import time
start_time = time.time()

################
### Preamble ###
################

jax.config.update("jax_enable_x64", True)

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

default_corner_kwargs = dict(bins=40, 
                        smooth=1., 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        color="blue",
                        plot_density=True, 
                        plot_datapoints=False, 
                        fill_contours=True,
                        max_n_ticks=4, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="red")

#############
### SETUP ###
#############

filters = ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y", "2massj", "2massh", "2massks", "sdssu"]
trigger_time = 57982.5285236896

##############
### MODEL  ###
##############

name = "Bu2019lm"
label = name
model = BullaLightcurveModel(name,
                             f"../../trained_models/KN/{name}/",
                             filters)

############
### DATA ###
############

data = load_event_data("../data/AT2017gfo.dat")

##################
### LIKELIHOOD ###
##################

detection_limit = None
likelihood = EMLikelihood(model,
                          data,
                          filters,
                          trigger_time=trigger_time,
                          tmin=0.5,
                          tmax=14.0,
                          detection_limit = detection_limit,
) 

def model():
    # Get the model parameter values
    KNphi = numpyro.sample("KNphi", dist.Uniform(15.0, 75.0))
    KNtheta = numpyro.sample("KNtheta", dist.Uniform(0.0, 90.0))
    log10_mej_dyn = numpyro.sample("log10_mej_dyn", dist.Uniform(-3.0, -1.0))
    log10_mej_wind = numpyro.sample("log10_mej_wind", dist.Uniform(-3.0, -0.5))
    luminosity_distance = numpyro.sample("luminosity_distance", dist.Uniform(30.0, 50.0))
    
    theta = {"KNphi": KNphi,
             "KNtheta": KNtheta,
             "log10_mej_dyn": log10_mej_dyn,
             "log10_mej_wind": log10_mej_wind,
             "luminosity_distance": luminosity_distance}
    
    # Get the log-likelihood from your likelihood function
    log_likelihood = likelihood.evaluate(theta)
    
    # Incorporate the likelihood into the model using numpyro.factor
    numpyro.factor("log_likelihood", log_likelihood)

# Set up NUTS sampler
print("Running NumPyro NUTS sampler . . .")
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1_000)

# Run MCMC
rng_key = random.PRNGKey(0)
mcmc.run(rng_key)

# Get samples
samples = mcmc.get_samples()
print("Running NumPyro NUTS sampler . . . DONE")

# Print summary of the samples
mcmc.print_summary()

# Convert JAX array to NumPy array for plotting
samples_np = {k: np.array(v) for k, v in samples.items()}

# Use corner to plot
data = np.column_stack([samples_np['KNphi'], 
                        samples_np['KNtheta'], 
                        samples_np['log10_mej_dyn'], 
                        samples_np['log10_mej_wind'], 
                        samples_np['luminosity_distance']])
corner.corner(data, labels=["KNphi", "KNtheta", "log10_mej_dyn", "log10_mej_wind", "luminosity_distance"], **default_corner_kwargs)

plt.savefig("./outdir_AT2017gfo_Bu2019lm/corner_NumPyro.png")
plt.show()

end_time = time.time()

print(f"Total time taken: {np.round(end_time - start_time, 2)} seconds. ")