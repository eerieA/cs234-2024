import gym
import torch
import numpy as np
from policy_gradient import PolicyGradient


# Dummy config with minimal viable values
class DummyConfig:
    env_name = "CartPole-v1"
    output_path = "./a2_code/results"
    log_path = "./a2_code/results/log.txt"
    record_path = "./a2_code/results/vid"
    scores_output = "./a2_code/results/scores.npy"
    plot_output = "./a2_code/results/plot.png"

    learning_rate = 1e-3
    n_layers = 1
    layer_size = 16
    batch_size = 1000
    max_ep_len = 200
    gamma = 0.99
    use_baseline = False
    normalize_advantage = True
    num_batches = 1
    summary_freq = 1
    record = False
    record_freq = 1


def test_get_returns():
    env = gym.make("CartPole-v1")
    pg = PolicyGradient(env, DummyConfig(), seed=42)

    # One episode with increasing rewards
    fake_paths = [
        {
            "reward": np.array([1.0, 2.0, 3.0]),
            "observation": np.zeros((3, env.observation_space.shape[0])),
            "action": np.array([0, 1, 0]),
        }
    ]

    returns = pg.get_returns(fake_paths)
    expected = [1.0 + 0.99 * 2.0 + 0.99**2 * 3.0, 2.0 + 0.99 * 3.0, 3.0]
    assert np.allclose(returns, expected), f"Returns incorrect: {returns} vs {expected}"


def test_normalize_advantage():
    env = gym.make("CartPole-v1")
    pg = PolicyGradient(env, DummyConfig(), seed=42)

    advantages = np.array([1.0, 2.0, 3.0])
    normalized = pg.normalize_advantage(advantages)
    assert np.allclose(np.mean(normalized), 0, atol=1e-6), "Mean not zero"
    assert np.allclose(np.std(normalized), 1, atol=1e-6), "Std not one"


def test_policy_update_step():
    env = gym.make("CartPole-v1")
    pg = PolicyGradient(env, DummyConfig(), seed=42)

    # Sample a small batch
    paths, _ = pg.sample_path(env, num_episodes=2)
    obs = np.concatenate([p["observation"] for p in paths])
    actions = np.concatenate([p["action"] for p in paths])
    returns = pg.get_returns(paths)
    advantages = pg.calculate_advantage(returns, obs)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Clone old params
    old_params = torch.cat([p.data.clone().flatten() for p in pg.policy.parameters()])
    pg.update_policy(obs, actions, advantages)
    new_params = torch.cat([p.data.clone().flatten() for p in pg.policy.parameters()])

    assert not torch.allclose(old_params, new_params), (
        "Policy parameters did not update"
    )


def test_training_loop_runs():
    env = gym.make("CartPole-v1")
    config = DummyConfig()
    config.num_batches = 2  # keep it fast
    pg = PolicyGradient(env, config, seed=42)
    pg.train()  # Should complete without errors


if __name__ == "__main__":
    test_get_returns()
    test_normalize_advantage()
    test_policy_update_step()
    test_training_loop_runs()
    print("All tests passed.")
