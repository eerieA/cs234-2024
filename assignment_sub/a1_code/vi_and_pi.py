### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)


def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.0
    ############################
    # YOUR IMPLEMENTATION HERE #
    future = np.sum(T[state, action, :] * V)
    future *= gamma
    backup_val = R[state, action] + future
    ############################

    return backup_val


def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    err = float("inf")
    while err > tol:
        v_k = np.copy(value_function)
        err = 0.0

        # update value for every state while keeping track of the max of element wise error
        for s in range(num_states):
            backup = bellman_backup(s, policy[s], R, T, gamma, value_function)
            err = max(err, abs(backup - value_function[s]))
            v_k[s] = backup

        # update the whole value function
        value_function = v_k
    ############################
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    """ for every s
            for every a = a_0, a_1, ...
                it has Q under pi_i
            new value = the a that max Q(s, a)
            new pi [s] = new value
        return new pi
    """
    for s in range(num_states):
        a_q_vals = np.zeros(num_actions)
        for a in range(num_actions):
            # Reminder: V_policy is old policy's value function
            a_q_vals[a] = bellman_backup(s, a, R, T, gamma, V_policy)
        
        new_pi_s = np.argmax(a_q_vals)
        new_policy[s] = new_pi_s

    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    ############################
    return value_function, policy


def test_bellman_backup():
    # Test r(s,a)
    R = np.array(
        [
            [0, 1],
            [0, 0],
            [2, 0],
        ]
    )

    # Test p(s'|s,a)
    # Initialize by writing as          T[action][state][next_state]
    # but after transpose will become   T[state][action][next_state]
    T = np.array(
        [
            # Transitions for action 0
            [
                [0.5, 0.5, 0.0],  # from state 0
                [0.0, 1.0, 0.0],  # from state 1
                [0.0, 0.0, 1.0],  # from state 2
            ],
            # Transitions for action 1
            [
                [0.0, 1.0, 0.0],  # from state 0
                [0.3, 0.7, 0.0],  # from state 1
                [0.2, 0.3, 0.5],  # from state 2
            ],
        ]
    )

    # Transpose to match expected shape: (state, action, next_state)
    T = np.transpose(T, (1, 0, 2))

    V = np.array([1.0, 2.0, 3.0])  # current value estimates
    gamma = 0.9

    # Pick state=0, action=1
    # Expected:
    #   immediate reward = R[0, 1] = 1
    #   next state distribution = T[0,1,:] = [0.0, 1.0, 0.0]
    #   future value = 0*1.0 + 1.0*2.0 + 0.0*3.0 = 2.0
    #   total = 1 + 0.9 * 2.0 = 2.8
    expected = 2.8

    # Compute using function
    actual = bellman_backup(0, 1, R, T, gamma, V)
    print(f"Expected: {expected:.3f}, Actual: {actual:.3f}")
    assert np.isclose(actual, expected), "Test failed!"
    print("Test on bellman_backup() passed.")


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = "WEAK"
    assert RIVER_CURRENT in ["WEAK", "MEDIUM", "STRONG"]
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.99

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([["L", "R"][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([["L", "R"][a] for a in policy_vi])

    test_bellman_backup()
