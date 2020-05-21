import math

import matplotlib.pyplot as plt
import torch as th
from torch.distributions.uniform import Uniform

from disco.controllers.amppi import AMPPI
from disco.models.cartpole import CartPoleModel
from disco.utils.helper import run_simulation, import_mog, save_progress
from disco.utils.utf import MerweScaledUTF


def terminal_cost(states):
    weight = 1e6
    terminal = (
        (states[:, 0] < -2.4)
        + (states[:, 0] > 2.4)
        + (states[:, 2] < -12 * 2 * math.pi / 360)
        + (states[:, 2] > 12 * 2 * math.pi / 360)
    )
    return weight * terminal.float()


def state_cost(states):
    # Original cost function on MPPI paper
    return (
        states[:, 0] ** 2
        + 500 * th.sin(states[:, 2]) ** 2
        + 1 * states[:, 1] ** 2
        + 1 * states[:, 3] ** 2
    )


# Constants../
PI = math.pi
ONE_DEG = 2 * PI / 360

# Simulation setup
ENV_NAME = "CartPole-v1"
ITERATIONS = 5
INIT_STATE = th.empty(4).uniform_(-0.05, 0.05)
VERBOSE = False
RENDER = True
SAVE = True

# Parameters distributions, uniform prior and pre-trained MoG posterior
PRIOR = Uniform(low=th.tensor([0.1, 0.1]), high=th.tensor([2.0, 2.0]))
POSTERIOR = import_mog("data/bayessim/cartpole_mog_L05_MP05.p")

# Model hyperparamaters
model_kwargs = {
    "mu_p": 0,  # no friction, like in gym
    "mu_c": 0,  # no friction, like in gym
    "length": 0.5,  # true parameter
    "mass_pole": 0.5,  # true parameter
}
model = CartPoleModel(**model_kwargs)

# Control hyperparameters
controller_kwargs = {
    "observation_space": model.observation_space,
    "action_space": model.action_space,
    "hz_len": 10,  # control horizon
    "n_samples": 200,  # sampled trajectories
    "lambda_": 10.0,  # inverse temperature
    "a_cov": th.eye(1),  # control exploration
    "inst_cost_fn": state_cost,
    "term_cost_fn": terminal_cost,
}

# UT hyperparameters
ut_kwargs = {"n": 2, "alpha": 0.5}  # number of sigma points and scaling
tf = MerweScaledUTF(**ut_kwargs)

# Simulation test cases
cases = {
    "baseline": {
        "model": {"uncertain_params": None, "params_dist": None},
        "controller": {"params_sampling": "none"},
        "plot_kwargs": {"color": "g", "label": r"Ground-truth: $\rho$"},
    },
    "mc_prior": {
        "model": {
            "uncertain_params": ("length", "mass_pole"),
            "params_dist": PRIOR,
        },
        "controller": {"params_sampling": "extended"},
        "plot_kwargs": {
            "color": "b",
            "label": r"MC: $\rho \sim \mathcal{{U}}$",
        },
    },
    "ut_prior": {
        "model": {
            "uncertain_params": ("length", "mass_pole"),
            "params_dist": PRIOR,
        },
        "controller": {"params_sampling": tf},
        "plot_kwargs": {
            "color": "r",
            "label": r"UT: $\rho \sim \mathcal{{U}}$",
        },
    },
    "mc_posterior": {
        "model": {
            "uncertain_params": ("length", "mass_pole"),
            "params_dist": POSTERIOR,
        },
        "controller": {"params_sampling": "extended"},
        "plot_kwargs": {"color": "k", "label": r"MC: $\rho \sim MoG$"},
    },
    "ut_posterior": {
        "model": {
            "uncertain_params": ("length", "mass_pole"),
            "params_dist": POSTERIOR,
        },
        "controller": {"params_sampling": tf},
        "plot_kwargs": {"color": "m", "label": r"UT: $\rho \sim MoG$"},
    },
}


if __name__ == "__main__":
    data_dict = dict()
    fig, ax = plt.subplots()
    for case in cases.keys():
        print("Running {} ...".format(case))
        model = CartPoleModel(**model_kwargs, **cases[case]["model"])
        controller = AMPPI(**controller_kwargs, **cases[case]["controller"])
        if case == list(cases.keys())[-1]:  # last key
            hold = False
        else:
            hold = True
        sim_dict = run_simulation(
            init_state=INIT_STATE,
            episodes=ITERATIONS,
            env_name=ENV_NAME,
            controller=controller,
            model=model,
            binary=True,
            verbose=VERBOSE,
            render=RENDER,
            reuse_ax=ax,
            hold=hold,
            **cases[case]["plot_kwargs"]
        )
        data_dict[case] = sim_dict

    if SAVE:
        folder = save_progress(data=data_dict, fig=fig)
        if folder is not None:
            print("Data saved to: {}".format(folder))
    print("Done")
