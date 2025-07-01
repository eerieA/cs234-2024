import pathlib
from typing import Tuple

import gym
import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from tqdm import trange

from data import load_data
from util import export_plot, np2torch, standard_error


LOGSTD_MIN = -10.0
LOGSTD_MAX = 2.0


class ActionSequenceModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        segment_len: int,
        lr: float = 1e-3,
    ):
        """Initialize an action sequence model.

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space
        action_dim : int
            Dimension of the action space
        hidden_dim : int
            Number of neurons in the hidden layer
        segment_len : int
            Action segment length
        lr : float, optional
            Optimizer learning rate, by default 1e-3

        TODO:
        Define self.net to be a neural network with a single hidden layer of size
        hidden_dim that takes as input an observation and outputs the parameters
        to define an action distribution for each time step of the sequence. Use
        ReLU activations, and have the last layer be a linear layer.

        Hint 1: We are predicting an action plan for the entire sequence given
                an observation. What should be the size of the output layer if we
                were to output the actions directly?
        Hint 2: We want the network outputs to be the mean and log standard
                deviation of a distribution from which we sample actions. How can
                we get the output size from the answer to the previous hint?

        Define also self.optimizer to optimize the network parameters. Use a default
        AdamW optimizer with learning rate lr.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.segment_len = segment_len
        #######################################################
        #########   YOUR CODE HERE - 3-9 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Return the mean and standard deviation of the action distribution for each observation.

        Parameters
        ----------
        obs : torch.Tensor
            Batch of observations

        Returns
        -------
        Tuple[torch.Tensor]
            The means and standard deviations for the actions at future timesteps

        TODO:
        Return mean and standard deviation vectors assuming that self.net predicts
        mean and log std.

        For each observation, your network should have output with dimension
        2 * self.segment_len * self.action_dim. Use the first half of these
        elements to set a mean vector of shape (self.segment_len, self.action_dim)
        in row major order. Use the second half to set a log_std vector of shape
        (self.segment_len, self.action_dim) in row major order. You may want to use
        https://pytorch.org/docs/stable/generated/torch.split.html.

        Hint 1: Apply tanh to the network output mean to force the mean values to
                lie between -1 and 1
        Hint 2: Clamp the log std predictions between LOGSTD_MIN and LOGSTD_MAX
                before converting it to the actual std
        """
        if isinstance(obs, np.ndarray):
            obs = np2torch(obs)
        assert obs.ndim == 2
        batch_size = len(obs)
        net_out = self.net(obs)

        #######################################################
        #########   YOUR CODE HERE - 3-9 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############
        return mean, std

    def distribution(self, obs: torch.Tensor) -> D.Distribution:
        """Take in a batch of observations and return a batch of action sequence distributions.

        Parameters
        ----------
        obs : torch.Tensor
            A tensor of observations

        Returns
        -------
        D.Distribution
            The action sequence distributions

        TODO: Given an observation, use self.forward to compute the mean and
        standard deviation of the action sequence distributions, and return the
        corresponding multivariate normal distribution.

        Use distributions.Independent in combination with distributions.Normal
        to create the multivariate normal instead of distributions.MultivariateNormal.
        See https://pytorch.org/docs/stable/distributions.html#independent and
        https://pytorch.org/docs/stable/distributions.html#normal
        """
        #######################################################
        #########   YOUR CODE HERE - 1-5 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return an action given an observation

        Parameters
        ----------
        obs : np.ndarray
            Single observation

        Returns
        -------
        np.ndarray
            The selected action

        TODO:
        Predict the full action sequence, and return the first action.

        Hint: Clamp the action values between -1 and 1
        """
        #######################################################
        #########   YOUR CODE HERE - 2-6 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############


class SFT(ActionSequenceModel):
    def update(self, obs: torch.Tensor, actions: torch.Tensor):
        """Pre-train a policy given an action sequence for an observation.

        Parameters
        ----------
        obs : torch.Tensor
            The start observation
        actions : torch.Tensor
            A plan of actions for the next timesteps

        TODO:
        Get the underlying action distribution, calculate the log probabilities
        of the given actions, and update the parameters in order to maximize their
        mean.

        Hint: Recall that Pytorch optimizers always try to minimize the loss.
        """
        #######################################################
        #########   YOUR CODE HERE - 4-6 lines.    ############

        #######################################################
        #########          END YOUR CODE.          ############
        return loss.item()


class DPO(ActionSequenceModel):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        segment_len: int,
        beta: float,
        lr: float = 1e-6,
    ):
        super().__init__(obs_dim, action_dim, hidden_dim, segment_len, lr=lr)
        self.beta = beta

    def update(
        self,
        obs: torch.Tensor,
        actions_w: torch.Tensor,
        actions_l: torch.Tensor,
        ref_policy: nn.Module,
    ):
        """Run one DPO update step

        Parameters
        ----------
        obs : torch.Tensor
            The current observation
        actions_w : torch.Tensor
            The actions of the preferred trajectory
        actions_l : torch.Tensor
            The actions of the other trajectory
        ref_policy : nn.Module
            The reference policy

        TODO:
        Implement the DPO update step.

        Hint 1: When calculating values using the reference policy, use the
                torch.no_grad() context to skip calculating gradients for it and
                achieve better performance
        Hint 2: https://pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html
        """
        #######################################################
        #########   YOUR CODE HERE - 8-14 lines.   ############

        #######################################################
        #########          END YOUR CODE.          ############
        return loss.item()


def evaluate(env, policy):
    total_reward = 0
    T = env.spec.max_episode_steps
    obs = env.reset()
    for _ in range(T):
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def get_batch(dataset, batch_size):
    obs1, obs2, act1, act2, label = dataset.sample(batch_size)
    obs = obs1[:, 0]
    assert torch.allclose(obs, obs2[:, 0])

    # Initialize assuming 1st actions preferred,
    # then swap where label = 1 (indicating 2nd actions preferred)
    actions_w = act1.clone()
    actions_l = act2.clone()
    swap_indices = label.nonzero()[:, 0]
    actions_w[swap_indices] = act2[swap_indices]
    actions_l[swap_indices] = act1[swap_indices]
    return obs, actions_w, actions_l


def main(args):
    output_path = pathlib.Path(__file__).parent.joinpath(
        "results_dpo",
        f"Hopper-v3-dpo-seed={args.seed}",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_pretrained_output = output_path.joinpath("model_sft.pt")
    model_output = output_path.joinpath("model.pt")
    scores_output = output_path.joinpath("scores.npy")
    plot_output = output_path.joinpath("scores.png")

    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # DPO assumes preferences are strict, so we ignore the equally preferred pairs
    pref_data = load_data(args.dataset_path, strict_pref_only=True)
    segment_len = pref_data.sample(1)[0].size(1)

    print("Training SFT policy")
    sft = SFT(obs_dim, action_dim, args.hidden_dim, segment_len)
    for _ in trange(args.num_sft_steps):
        obs, actions_w, _ = get_batch(pref_data, args.batch_size)
        sft.update(obs, actions_w)

    print("Evaluating SFT policy")
    returns = [evaluate(env, sft.act) for _ in range(args.num_eval_episodes)]
    print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")

    print("Training DPO policy")
    dpo = DPO(
        obs_dim, action_dim, args.hidden_dim, segment_len, args.beta, lr=args.dpo_lr
    )
    dpo.net.load_state_dict(sft.net.state_dict())  # init with SFT parameters
    all_returns = []
    for step in trange(args.num_dpo_steps):
        obs, actions_w, actions_l = get_batch(pref_data, args.batch_size)
        dpo.update(obs, actions_w, actions_l, sft)
        if (step + 1) % args.eval_period == 0:
            print("Evaluating DPO policy")
            returns = [evaluate(env, dpo.act) for _ in range(args.num_eval_episodes)]
            print(f"Return: {np.mean(returns):.2f} +/- {standard_error(returns):.2f}")
            all_returns.append(np.mean(returns))

    # Log the results
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with open(model_pretrained_output, "wb") as f:
        torch.save(sft, f)
    with open(model_output, "wb") as f:
        torch.save(dpo, f)
    np.save(scores_output, all_returns)
    export_plot(all_returns, "Returns", "Hopper-v3", plot_output)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--env-name", default="Hopper-v3")
    parser.add_argument(
        "--dataset-path",
        default=pathlib.Path(__file__).parent.joinpath("data", "prefs-hopper.npz"),
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-sft-steps", type=int, default=50000)
    parser.add_argument("--num-dpo-steps", type=int, default=50000)
    parser.add_argument("--dpo-lr", type=float, default=1e-6)
    parser.add_argument("--eval-period", type=int, default=1000)
    parser.add_argument("--num-eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
