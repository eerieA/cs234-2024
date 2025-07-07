import torch
from run_rlhf import RewardModel

def test_forward_output_range():
    model = RewardModel(obs_dim=2, action_dim=1, hidden_dim=8, r_min=0.0, r_max=1.0)
    obs = torch.randn(10, 2)
    act = torch.randn(10, 1)
    rewards = model.forward(obs, act)
    assert rewards.shape == (10,)
    assert torch.all(rewards >= 0.0) and torch.all(rewards <= 1.0)
    print("✅ Forward pass outputs in correct shape and range.")

def test_compute_reward_matches_forward():
    model = RewardModel(obs_dim=2, action_dim=1, hidden_dim=8, r_min=0.0, r_max=1.0)
    obs = torch.tensor([0.1, -0.2]).numpy()
    act = torch.tensor([0.5]).numpy()
    reward = model.compute_reward(obs, act)
    assert isinstance(reward, float)
    # Compare against batched forward
    obs_tensor = torch.tensor(obs).unsqueeze(0)
    act_tensor = torch.tensor(act).unsqueeze(0)
    with torch.no_grad():
        ref = model.forward(obs_tensor, act_tensor).item()
    assert abs(reward - ref) < 1e-5
    print("✅ compute_reward matches forward() output.")

def test_update_learns_simple_preference():
    model = RewardModel(obs_dim=1, action_dim=1, hidden_dim=8, r_min=0.0, r_max=1.0)

    # Two 3-step trajectories: traj1 has higher feature values than traj2
    # So ideally, model should learn to prefer traj1 (label 0)
    obs1 = torch.tensor([[[1.0], [2.0], [3.0]]])  # shape (1, 3, 1)
    obs2 = torch.tensor([[[0.1], [0.2], [0.3]]])
    act1 = torch.tensor([[[0.0], [0.0], [0.0]]])
    act2 = torch.tensor([[[0.0], [0.0], [0.0]]])
    label = torch.tensor([0])  # traj1 preferred

    batch = (obs1, obs2, act1, act2, label)

    # Check initial loss
    initial_loss = model.update(batch)
    for _ in range(100):
        model.update(batch)
    final_loss = model.update(batch)
    assert final_loss < initial_loss, "Expected loss to decrease after training"
    print(f"✅ Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")

if __name__ == "__main__":
    test_forward_output_range()
    test_compute_reward_matches_forward()
    test_update_learns_simple_preference()
