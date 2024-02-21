#basic libraries
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import corner
#SBI: Simulation based Inference
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.utils import MultipleIndependent
from sbi.inference import SNPE
from sbi.utils import get_density_thresholder, RestrictedPrior
from sbi.inference import SNLE, SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.analysis import pairplot
#torch
import torch
from torch import zeros, ones, eye
from torch.distributions import MultivariateNormal
import torch.distributions as D
from torch.distributions import Uniform

#reading the text file with COBE data and assigning the columns as arrays
with open("firas_monopole_spec_v1.txt", "r") as f:
    while True:
        line = f.readline()
        if not line.startswith("#"):
            break
    data = np.loadtxt(f)
    frequency_cm_over = data[:, 0] # frequency
    intensity_MJy_over_sr = data[:, 1] # FIRAS monopole spectrum
    intensity_residual = data[:, 2] # residual monopole spectrum
    intensity_KJy_err = data[:, 3] # spectrum uncertainty
    galaxy_spctrum = data[:, 4] # modeled Galaxy spectrum

#constants
c = const.c #299792458.0 m / s
h = const.h #6.626e-3  J s
k = const.k #1.38e-23 J / K

#change the intensity to SI units
frequency_hz = frequency_cm_over * c*1e2
print(f"Frequency in Hz: {frequency_hz}")
#change frequency to SI units
# 1 MJy/sr = 10^-20 W/m^-2*Hz
MJy_to_si = 1e-20
intensity_si = intensity_MJy_over_sr * MJy_to_si
print(f"Intensity in W/m^-2*Hz: {intensity_si}")

def func_planck(temp):
    freq = torch.tensor(frequency_hz)
    amplitude = 2 * h * freq**3 / c**2
    shape = (np.exp(h * freq / (k * temp)) - 1) ** -1
    return amplitude * shape
def simulator(pars):
    t1, t2 = pars
    freq = torch.tensor(frequency_hz)
    amplitude = 2 * h * freq**3 / c**2
    shape1 = (np.exp(h * freq / (k * t1)) - 1) ** -1
    shape2 = (np.exp(h * freq / (k * t2)) - 1) ** -1
    return amplitude * (shape1 + shape2)

#parameters controlling the SBI procedure
num_sim = 10000
num_sample = 100000
t_cmb_range = [0., 5]
t1_range = [0,5]
t2_range = [0,2]
prior = MultipleIndependent(
    [
        Uniform(low = t1_range[0] * torch.ones(1),   high = t1_range[1] * torch.ones(1)),
        Uniform(low = t2_range[0] * torch.ones(1),   high = t2_range[1] * torch.ones(1)),
    ],
    validate_args=False,
)

simulator, prior = prepare_for_sbi(simulator, prior)
inference = SNPE(prior=prior)
theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_sim)
inference = inference.append_simulations(theta, x)
density_estimator = inference.train()
posterior = inference.build_posterior(density_estimator)
observation = intensity_si
posterior_samples = posterior.sample((num_sample,), x = observation)
samples_arr = posterior_samples.numpy()
# Plot the numpy array using the corner.corner function
fig = corner.corner(samples_arr, show_titles=True, labels=[r'$T_{CMB}$', r'$T_{else}$'], smooth=0.5,
                    bins=50, quantiles=[0.16, 0.5, 0.84], range=[(-2, 20), (-2, 20) ], color='black',
                   plot_contours=True, fill_contours=True, titles=[r'$T_{CMB}$', r'$T_{else}$'], figsize=(10, 10))
fig.legend(handles=[plt.Line2D([], [], color="k", label=f"{num_sim} sim\n{num_sample} samples\n \nT_CMB range:{t1_range}\nT_else range: {t2_range}")], 
           fontsize=12, frameon=False, loc="upper right") #, bbox_to_anchor=(0.95, 0.05))
# Save or show the plot
# fig.savefig(f"CMB_corner_plot_with_{num_sim}.png", dpi=300)
plt.show()
