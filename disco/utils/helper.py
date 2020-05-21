import os
import pickle
import time

import gym
import torch as th

from disco.utils import distributions


def save_progress(data=None, folder_name=None, fig=None):
    """Saves session to the project data folder.

     Path can be specified, if not, an auto generated folder based on the
     current date-time is used. May include a plot.

    :param data: A data object to save. If None, whole session is saved.
    :type data: object
    :param folder_name: A path-like string containing the output path.
    :type folder_name: str
    :param fig: A figure object.
    :type fig: matplotlib.figure.Figure
    """
    if folder_name is None:
        folder_name = time.strftime("%Y%m%d-%H%M%S")
        folder_name = "data/local/" + str(folder_name)
    if not os.path.exists(os.path.join(str(folder_name), "plots")):
        os.makedirs(os.path.join(str(folder_name), "plots"))
    if fig:
        plot_path = os.path.join(str(folder_name), "plots/plot.pdf")
        fig.savefig(plot_path)

    if data is None:
        try:
            import dill
        except ImportError:
            print("Couldn't import package dill. Aborting save progress.")
            return None
        sess_path = os.path.join(str(folder_name), "session.pkl")
        dill.dump_session(sess_path)
    else:
        data_path = os.path.join(str(folder_name), "data.pt")
        th.save(data, data_path, pickle_protocol=-1)
    return folder_name


def import_mog(filename):
    """Imports a Mixture of Gaussians from a file.

    :param filename: A path-like string containing the filename to load.
    :type filename: str
    :return: A Mixture of Gaussians.
    :rtype: utils.pdf.MoG
    """
    with open(filename, "rb") as f:
        mog_dict = pickle.load(f)
        mog = distributions.MoG(
            a=mog_dict["weights"],
            ms=mog_dict["means"],
            Ss=mog_dict["covariances"],
        )
    return mog


def _plot_mean_cost(ax, acc_costs, **plot_kwargs):
    """Creates an average cost plot with standard deviation."""
    steps = acc_costs.shape[1]
    cost_mean = acc_costs.mean(dim=0)
    cost_std = acc_costs.std(dim=0) / 2
    timesteps = th.arange(steps)
    mean_ln = ax.plot(timesteps.tolist(), cost_mean.tolist(), **plot_kwargs,)
    try:
        color = plot_kwargs["color"]
        fill_ln = ax.fill_between(
            timesteps.tolist(),
            (cost_mean - cost_std).clamp(min=0.0),
            cost_mean + cost_std,
            facecolor=color,
            alpha=0.3,
        )
    except KeyError:
        fill_ln = ax.fill_between(
            timesteps.tolist(),
            (cost_mean - cost_std).clamp(min=0.0),
            cost_mean + cost_std,
            alpha=0.3,
        )
    return mean_ln, fill_ln


def run_episode(
    env_name,
    init_state,
    controller,
    model,
    binary=False,
    steps=200,
    steps_per_msg=20,
    verbose=True,
    render=False,
):
    """Runs one episode of the simulation loop.

    :param env_name: A gym environment name.
    :type env_name: str
    :param init_state: The initial state for all episodes.
    :type init_state: th.Tensor
    :param controller: The controller used for the task.
    :type controller: disco.controllers.amppi.AMPPI
    :param model: The forward model used by the controller.
    :type model: disco.models.base.BaseModel
    :param binary: Whether or not the actions should be converted to a `bool`.
    :type binary: bool
    :param steps: Number of steps in each episode.
    :type steps: int
    :param steps_per_msg: Number of steps before printing update messages.
    :type steps_per_msg: int
    :param verbose: Controls whether information messages should be print to
        screen.
    :type verbose: bool
    :param render: Enable or disable gym rendering.
    :type render: bool
    :return: A dictionary with (states, actions, cost, predicted states,
        predicted actions, omega) for each timestep.
    :rtype: dict
    """
    env = gym.make(env_name)
    env.reset()
    env.env.state = init_state  # sets the initial state of the simulation...
    state = init_state  # ... and the forward model

    # create the aggregating tensors and set them to 'nan' to check if
    # simulation breaks
    ep_dict = dict.fromkeys(
        ["states", "actions", "costs", "states_pred", "actions_pred", "omegas"]
    )

    step = 0
    while step < steps:
        if render:
            env.render()
        # values are: est. cost, states, actions, omega
        values = controller.update_actions(model, state)
        action = controller.a_seq[0]
        if binary:
            # converts to a binary scalar if CartPole
            # TODO: generalise this to other models
            action = 1 if action > 0 else 0
        controller.roll(steps=1)
        _, _, done, _ = env.step(action)
        state = th.tensor(env.state)
        cost = controller.inst_cost_fn(state.view(1, -1)).squeeze()
        if verbose and not step % steps_per_msg:
            print(
                "Step {0}: action taken {1:.2f}, cost {2:.2f}".format(
                    step, float(action), float(cost)
                )
            )
            print("Current state: {0}".format(state))
            print(
                "Next actions 4 actions: {}".format(
                    controller.a_seq[:4].flatten()
                )
            )

        #  note zip is carefully ordered
        for key, value in zip(
            ep_dict.keys(),
            [state, controller.a_seq[0], cost] + list(values[1:]),
        ):
            if step == 0:
                # initialize each dictionary entry with `nan` values of size
                # [`steps` + value.size()]
                ep_dict[key] = th.full(
                    (th.Size([steps]) + value.size()),
                    fill_value=float("nan"),
                    dtype=value.dtype,
                )
            ep_dict[key][step] = value

        if done:
            env.close()
            break
        else:
            step += 1

    if verbose:
        print(
            "Last step {0}: action taken {1:.2f}, cost {2:.2f}".format(
                step, float(action), float(cost)
            )
        )
        print("Last state: theta={0[0]}, theta_dot={0[1]}".format(state))
        print("Next actions: {}".format(controller.a_seq.flatten()))
    env.close()
    return ep_dict


def run_simulation(
    env_name,
    init_state,
    episodes,
    controller,
    model,
    binary=False,
    steps=200,
    steps_per_msg=20,
    verbose=True,
    render=False,
    plot=True,
    reuse_ax=None,
    hold=False,
    **plot_kwargs,
):
    """Runs a number of episodes in simulation and plot average cost results.

    :param env_name: A gym environment name.
    :type env_name: str
    :param init_state: The initial state for all episodes.
    :type init_state: th.Tensor
    :param episodes: Number of episodes to execute.
    :type episodes: int
    :param controller: The controller used for the task.
    :type controller: disco.controllers.amppi.AMPPI
    :param model: The forward model used by the controller.
    :type model: disco.models.base.BaseModel
    :param binary: Whether or not the actions should be converted to a `bool`.
    :type binary: bool
    :param steps: Number of steps in each episode.
    :type steps: int
    :param steps_per_msg: Number of steps before printing update messages.
    :type steps_per_msg: int
    :param verbose: Controls whether information messages should be print to
        screen.
    :type verbose: bool
    :param render: Enable or disable gym rendering.
    :type render: bool
    :param plot: Whether an average cost plot should be created.
    :type plot: bool
    :param reuse_ax: If provided, will be used to render the plot.
    :type reuse_ax: matplotlib.axes.Axes
    :param hold: Whether the plot should be displayed or held for further
        processing.
    :type hold: bool
    :param plot_kwargs: kwargs which will be passed by reference to the plot
        function.
    :type plot_kwargs: dict
    """
    import matplotlib.pyplot as plt

    sim_dict = dict()
    for episode in range(episodes):
        if verbose:
            print("=== Episode {} of {} ===".format(episode + 1, episodes))
        ep_data_dict = run_episode(
            env_name,
            init_state,
            controller,
            model,
            binary,
            steps,
            steps_per_msg,
            verbose,
            render,
        )
        if episode == 0:
            for key, value in ep_data_dict.items():
                sim_dict[key] = value.unsqueeze(0)
        else:
            for key, value in ep_data_dict.items():
                sim_dict[key] = th.cat(
                    (sim_dict[key], value.unsqueeze(0)), dim=0
                )

    if plot:
        if reuse_ax is None:
            fig, ax = plt.subplots()
        elif reuse_ax is not None:
            ax = reuse_ax
        _, _ = _plot_mean_cost(ax, sim_dict["costs"], **plot_kwargs)
        if not hold:
            ax.set_xlabel("Step")
            ax.set_ylabel(
                "Mean cost over time for {} episodes".format(episodes)
            )
            ax.legend()
            ax.grid(True, axis="y")
            ax.autoscale(enable=True, axis="x", tight=True)
            plt.show()

    return sim_dict
