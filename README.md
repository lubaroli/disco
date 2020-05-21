DISCO: Double Likelihood-free Inference Stochastic Control
==========================================================

**Abstract**:
Accurate simulation of complex physical systems enables the development, 
testing, and certification of control strategies before they are deployed into 
the real systems. As simulators become more advanced, the analytical 
tractability of the differential equations and associated numerical solvers 
incorporated in the simulations diminishes, making them difficult to analyse. 
A potential solution is the use of probabilistic inference to assess the 
uncertainty of the simulation parameters given real observations of the 
system. Unfortunately the likelihood function required for inference is 
generally expensive to compute or totally intractable. In this paper we 
propose to leverage the power of modern simulators and recent techniques in 
Bayesian statistics for likelihood-free inference to design a control 
framework that is efficient and robust with respect to the uncertainty over 
simulation parameters. The posterior distribution over simulation parameters 
is propagated through a potentially non-analytical model of the system with 
the unscented transform, and a variant of the information theoretical model 
predictive control. This approach provides a more efficient way to evaluate 
trajectory roll outs than Monte Carlo sampling, reducing the online 
computation burden. Experiments show that the controller proposed attained 
superior performance and robustness on classical control and robotics tasks 
when compared to models not accounting for the uncertainty over model 
parameters.

Welcome to **DISCO**'s development page! **DISCO** is an open-source package 
for Python 3 providing modules and examples to quickly setup and run 
experiments with the control framework presented on our research paper.

This package is still under development. We are open to potential 
collaborations.

Installation
------------

DISCO is built for
[Python](https://www.python.org/) (version 3.6 or later), 
and depends on
[PyTorch](https://pytorch.org), 
[OpenAI Gym](https://gym.openai.com), 
[NumPy](https://www.numpy.org/), 
[matplotlib](https://matplotlib.org),
and [SciPy](https://www.scipy.org/).

Additionally, we use [Dill](https://pypi.org/project/dill/) to save session 
files for experiments. If these dependencies are not installed, you may install 
them with:

```shell
  $ pip install -r requirements.txt
  $ pip install -r optional-requirements.txt
```

### Using pip

```shell
  $ sudo pip install disco-rl
```

### From source

```shell
  $ git clone https://github.com/lubaroli/disco
  $ cd disco/
  $ sudo python setup.py install
```

Running Experiments
-------------------

For convenience, scripts for running the Pendulum experiment found on the paper
are provided on both for Python and Jupyter. These files are located on the 
`demo/` folder.

Although the Jupyter notebook describes the step-by-step design of the 
experiment, there are a few variables worth highlighting. The simulation is 
configured using the following variables:

```
    ENV_NAME   (required) The name of the environment. Models are provided for
                          'Pendulum-v0', 'CartPole-v1' and a custom 'SkidSteer'.
    INIT_STATE (required) The initial states for all episodes. A tensor of
                          appropriate dimensions.
    ITERATIONS (required) The number of episodes executed in for each test case.
    PRIOR      (optional) If using distributions over parameters of the forward
                          model, this is the distribution used during the first
                          epoch.
    POSTERIOR  (optional) If using distributions over parameters of the forward
                          model, this is the refined distribution used during
                          the subsequent epochs.
    RENDER     (optional) Controls whether the experiments should be rendered
                          by gym.
    VERBOSE    (optional) Controls whether progress messages are printed to the
                          console.
    SAVE       (optional) Controls whether the experiment data and plots are
                          saved to 'data/local/<date-time>' folder.
```

Furthermore, the test cases for baseline comparison are defined in a dictionary 
holding the `**kwargs` used by the forward model, controller and `pyplot`.

```python
cases = {
    "baseline": {
        "model": {"uncertain_params": None, "params_dist": None},
        "controller": {"params_sampling": "none"},
        "plot_kwargs": {"color": "g", "label": r"Ground-truth: $\rho$"},
    },  #...
}
```

Please refer to each module documentation for details on these arguments.

