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
    """ i = 0
    new policy = an array of randomly chosen actions
    while i==0 or ||new_policy - policy||_1 > tol
        curr policy value func = policy_evaluation(policy)
        new policy = policy_improvement(policy, R, T, curr policy value func, gamma)
        i += 1 """
    i = 0
    new_policy = np.random.choice(num_actions, size=num_states)
    # For recording the distance between pi_{i+1} and pi_i
    dist = float("inf")
    while i == 0 or dist > tol:
        if i > 10000:
            print("policy_iteration: Max iteration reached. Is there something wrong?")
            break

        V_policy = policy_evaluation(policy, R, T, gamma)
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)

        # Record distance before updating policy
        dist = np.linalg.norm(new_policy - policy, ord=1)
        policy = new_policy
        i += 1
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
    """ 
    k = 1
    value func = all zeros (already given)
    while ||value func k+1 = value func k||_inf <= tol
        for each s
            new value func [s] = max of bellman backup val for every a, at this s
        distance = ||new value func - value func||_inf
        k += 1
    after value func converges, extract policy
    for each s
        new policy [s] = action with max bellman backup val, at this s """
    k = 0
    # For tracking the inf norm distance
    err = float("inf")
    while err > tol:
        if k > 10000:
            print("value_iteration: Max iteration reached. Is there something wrong?")
            break

        err = 0.0
        new_value_func = np.copy(value_function)
        for s in range(num_states):
            a_q_vals = [
                bellman_backup(s, a, R, T, gamma, value_function)
                for a in range(num_actions)
            ]
            max_q = max(a_q_vals)

            err = max(err, abs(max_q - value_function[s]))
            new_value_func[s] = max_q
            k += 1
        value_function = new_value_func

    # Extract policy
    for s in range(num_states):
        a_q_vals = [
            bellman_backup(s, a, R, T, gamma, value_function)
            for a in range(num_actions)
        ]
        policy[s] = np.argmax(a_q_vals)
    ############################
    return value_function, policy


def find_max_gamma_goes_left(strength_name, tol=0.001):
    low, high = 0.0, 1.0
    gamma_result = 0.0

    while high - low > tol:
        mid = (low + high) / 2
        env = RiverSwim(strength_name)
        R, T = env.get_model()
        V, pi = value_iteration(R, T, gamma=mid)

        if pi[0] == 0:  # LEFT
            # Could work but try to increase gamma and see if that still works
            gamma_result = mid
            low = mid
        else:  # RIGHT
            # Would not work, reduce gamma
            high = mid

    return round(gamma_result, 3)


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
    test_bellman_backup()

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

    print("\nBinary search for the required gamma...")
    print("Weak current:", find_max_gamma_goes_left("WEAK"))
    print("Medium current:", find_max_gamma_goes_left("MEDIUM"))
    print("Strong current:", find_max_gamma_goes_left("STRONG"))
