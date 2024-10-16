"""
Example showing how to load the data
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import corner

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

# Load the samples
data = np.load("./outdir_AT2017gfo_Bu2019lm/results_production.npz")
chains = data["chains"]
labels = ["KNphi", "KNtheta", "log10_mej_dyn", "log10_mej_wind", "d_L"]

# # Make the cornerplot
# print("Plotting corner . . .")
# corner.corner(chains, labels = labels, hist_kwargs={'density': True}, **default_corner_kwargs)
# plt.savefig("./test.png", bbox_inches = 'tight')
# plt.close()
# print("Plotting corner . . . done")