import torch
import numpy as np
from run_dpo import ActionSequenceModel
from run_dpo import SFT, DPO

def test_action_sequence_model_basic():
    obs_dim = 11
    action_dim = 3
    hidden_dim = 32
    segment_len = 5

    model = ActionSequenceModel(obs_dim, action_dim, hidden_dim, segment_len)

    # Dummy batch of observations
    batch_size = 4
    obs = torch.randn(batch_size, obs_dim)

    # Run forward pass
    mean, std = model(obs)

    assert mean.shape == (batch_size, segment_len, action_dim), \
        f"Expected mean shape {(batch_size, segment_len, action_dim)}, got {mean.shape}"
    assert std.shape == (batch_size, segment_len, action_dim), \
        f"Expected std shape {(batch_size, segment_len, action_dim)}, got {std.shape}"

    # Check value ranges
    assert torch.all(mean <= 1.0) and torch.all(mean >= -1.0), \
        "Mean values not clamped within [-1, 1]"
    assert torch.all(std > 0), "Std values must be positive"

    print("✅ forward() works correctly")


def test_distribution_and_sample():
    obs_dim = 11
    action_dim = 3
    hidden_dim = 32
    segment_len = 5
    model = ActionSequenceModel(obs_dim, action_dim, hidden_dim, segment_len)

    obs = torch.randn(2, obs_dim)  # batch size = 2

    dist = model.distribution(obs)
    sample = dist.sample()

    assert sample.shape == (2, segment_len, action_dim), \
        f"Sampled shape mismatch: got {sample.shape}"

    print("✅ distribution() works correctly")


def test_act_returns_first_action():
    obs_dim = 11
    action_dim = 3
    hidden_dim = 32
    segment_len = 5
    model = ActionSequenceModel(obs_dim, action_dim, hidden_dim, segment_len)

    obs = np.random.randn(obs_dim).astype(np.float32)
    action = model.act(obs)

    assert isinstance(action, np.ndarray), "act() did not return a numpy array"
    assert action.shape == (action_dim,), f"act() should return shape ({action_dim},), got {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), "Action not clamped between [-1, 1]"

    print("✅ act() works correctly")

def test_sft_update():
    print("Testing SFT update...")

    obs_dim = 10
    action_dim = 4
    segment_len = 5
    hidden_dim = 32
    batch_size = 8

    model = SFT(obs_dim, action_dim, hidden_dim, segment_len)
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.tanh(torch.randn(batch_size, segment_len, action_dim))  # simulate [-1, 1] action range

    # Track params before update
    old_params = [p.clone() for p in model.parameters()]

    loss = model.update(obs, actions)

    assert isinstance(loss, float), "SFT loss is not a float"
    assert np.isfinite(loss), "SFT loss is not finite"

    # Check that some parameter changed
    changed = any((not torch.allclose(p0, p1)) for p0, p1 in zip(old_params, model.parameters()))
    assert changed, "SFT update did not change model parameters"

    print("✅ SFT update test passed.")


def test_dpo_update():
    print("Testing DPO update...")

    obs_dim = 10
    action_dim = 4
    segment_len = 5
    hidden_dim = 32
    batch_size = 8
    beta = 0.1

    dpo_model = DPO(obs_dim, action_dim, hidden_dim, segment_len, beta)
    ref_model = SFT(obs_dim, action_dim, hidden_dim, segment_len)

    obs = torch.randn(batch_size, obs_dim)
    actions_w = torch.tanh(torch.randn(batch_size, segment_len, action_dim))
    actions_l = torch.tanh(torch.randn(batch_size, segment_len, action_dim))

    old_params = [p.clone() for p in dpo_model.parameters()]

    loss = dpo_model.update(obs, actions_w, actions_l, ref_model)

    assert isinstance(loss, float), "DPO loss is not a float"
    assert np.isfinite(loss), "DPO loss is not finite"

    changed = any((not torch.allclose(p0, p1)) for p0, p1 in zip(old_params, dpo_model.parameters()))
    assert changed, "DPO update did not change model parameters"

    print("✅ DPO update test passed.")

if __name__ == "__main__":
    test_action_sequence_model_basic()
    test_distribution_and_sample()
    test_act_returns_first_action()

    test_sft_update()
    test_dpo_update()