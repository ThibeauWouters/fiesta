import os
import json
import copy
import jax 
print(f"GPU found? {jax.devices()}")
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import corner

from fiesta.inference.lightcurve_model import BullaLightcurveModel
from fiesta.inference.likelihood import EMLikelihood
from fiesta.inference.prior import Uniform, Composite
from fiesta.inference.injection import InjectionRecovery
from fiesta.inference.fiesta import Fiesta
from fiesta.utils import load_event_data

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
                        # quantiles=[],
                        # levels=[0.9],
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
trigger_time = 0.0 

###########################
### MODEL AND INJECTION ###
###########################

model = BullaLightcurveModel("Bu2022Ye",
                             "../trained_models/Bu2022Ye/",
                             filters)

injection_dict = {"KNtheta": jnp.pi / 4,
                  "log10_mej_dyn": -2.5,
                  "vej_dyn": 0.15,
                  "Yedyn": 0.25,
                  "log10_mej_wind": -1.5,
                  "vej_wind": 0.1,
                  "luminosity_distance": 44.0,}

injection = InjectionRecovery(model, 
                              filters,
                              injection_dict)

injection.create_injection()

#############################
### PRIORS AND LIKELIHOOD ###
#############################

inclination_EM      = Uniform(xmin=0.0, xmax=jnp.pi/2., naming=['KNtheta'])
log10_mej_dyn       = Uniform(xmin=-3.0, xmax=-1.7, naming=['log10_mej_dyn'])
vej_dyn             = Uniform(xmin=0.12, xmax=0.25, naming=['vej_dyn'])
Yedyn               = Uniform(xmin=0.15, xmax=0.3, naming=['Yedyn'])
log10_mej_wind      = Uniform(xmin=-2.0, xmax=-0.89, naming=['log10_mej_wind'])
vej_wind            = Uniform(xmin=0.03, xmax=0.15, naming=['vej_wind'])

prior_list = [inclination_EM, 
              log10_mej_dyn, 
              vej_dyn, 
              Yedyn, 
              log10_mej_wind, 
              vej_wind]

prior = Composite(prior_list)

likelihood = EMLikelihood(model,
                          filters,
                          injection.data,
                          fixed_params={"luminosity_distance": 44.0},
                          trigger_time=trigger_time)

##############
### FIESTA ###
##############

mass_matrix = jnp.eye(prior.n_dim)
eps = 1e-4
local_sampler_arg = {"step_size": mass_matrix * eps}

fiesta = Fiesta(likelihood,
                prior,
                n_chains = 1_000,
                n_loop_training = 3,
                n_loop_production = 3,
                num_layers = 4,
                hidden_size = [64, 64],
                local_sampler_arg=local_sampler_arg)

fiesta.sample(jax.random.PRNGKey(0))

# Save for postprocessing
outdir = "./outdir/"
if not os.path.exists(outdir):
    os.makedirs(outdir)
   
### New attempt

print("Saving injection dict")
save_name = os.path.join(outdir, "injection_dict.npz")
np.savez(save_name, **injection_dict)

name = outdir + f'results_training.npz'
print(f"Saving samples to {name}")
state = fiesta.Sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = state["chains"], state[
"log_prob"], state["local_accs"], state["global_accs"], state["loss_vals"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, log_prob=log_prob, local_accs=local_accs,
        global_accs=global_accs, loss_vals=loss_vals)

#  - production phase
name = outdir + f'results_production.npz'
print(f"Saving samples to {name}")
state = fiesta.Sampler.get_sampler_state(training=False)
chains, log_prob, local_accs, global_accs = state["chains"], state[
    "log_prob"], state["local_accs"], state["global_accs"]
local_accs = jnp.mean(local_accs, axis=0)
global_accs = jnp.mean(global_accs, axis=0)
np.savez(name, chains=chains, log_prob=log_prob,
            local_accs=local_accs, global_accs=global_accs)
    
print("DONE")

################
### PLOTTING ###
################

fig, ax = plt.subplots(nrows = len(filters), ncols = 1, figsize = (10, 20))
for i, filter_name in enumerate(filters):
    ax = plt.subplot(len(filters), 1, i + 1)
    
    # Load the data
    t, mag, err = injection.data[filter_name].T
    ax.plot(t, mag, color = "blue", label = "Model")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel(filter_name)
    ax.errorbar(t - trigger_time, mag, yerr = err, fmt = "o", color = "red", label = "Data")
    
    ax.invert_yaxis()
    
plt.savefig("./figures/test_injection_data.png")
plt.close()

# Fixed names: do not include them in the plotting, as will break corner
fixed_parameter_names = ["luminosity_distance"]
parameter_names = fiesta.likelihood.model.parameter_names
truths = np.array([injection_dict[name] for name in parameter_names if name not in fixed_parameter_names])

n_chains, n_steps, n_dim = np.shape(chains)
samples = np.reshape(chains, (n_chains * n_steps, n_dim))
samples = np.asarray(samples) # convert from jax.numpy array to numpy array for corner consumption

fig = corner.corner(samples, labels = parameter_names, truths = truths, hist_kwargs={'density': True}, **default_corner_kwargs)
fig.savefig("./figures/test_injection_corner.png", bbox_inches = 'tight')
plt.close()