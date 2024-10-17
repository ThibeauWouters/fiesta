import numpy as np 
import matplotlib.pyplot as plt

from fiesta.train.Benchmarker import Benchmarker
from fiesta.inference.lightcurve_model import AfterglowpyLightcurvemodel
from fiesta.utils import Filter


name = "Bu2019lm"
model_dir = f"../../trained_models/KN/{name}/"
FILTERS = ["ps1__g", "ps1__r", "ps1__i", "ps1__z", "ps1__y", "2massj", "2massh", "2massks", "sdssu"]

for metric_name in ["$\\mathcal{L}_2$", "$\\mathcal{L}_\infty$"]:    
    if metric_name == "$\\mathcal{L}_2$":
        file_ending = "L2"
    else:
        file_ending = "Linf"
    
    B = Benchmarker(name = name,
                    model_dir = model_dir,
                    filters = FILTERS,
                    n_test_data = 2_000,
                    metric_name = metric_name,
                    remake_test_data = True,
                    jet_type = -1)
    
    for filt in FILTERS:
        fig, ax = B.plot_lightcurves_mismatch(filter =filt)
        fig.savefig(f"./figures/benchmark_{filt}_{file_ending}.pdf", dpi = 200)
    
        B.print_correlations(filter = filt)
    
    
        fig, ax = B.plot_worst_lightcurve(filter = filt)
        fig.savefig(f"./figures/worst_lightcurve_{filt}_{file_ending}.pdf", dpi = 200)




