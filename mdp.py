import numpy as np
from itertools import product

def apply_action(state, action):
    theta1, theta2 = state
    d_theta1, d_theta2 = action

    # Apply the action
    new_theta1 = np.clip(theta1 + d_theta1, 0, 170)
    new_theta2 = np.clip(theta2 + d_theta2, 0, 170)

    return (new_theta1, new_theta2)

def compute_reward(error, threshold=10):
    distance = error
    # Option 1: shaped reward (negative distance)
    reward = -distance

    # Option 2: bonus for being very close
    if distance < threshold:
        reward += 100  # big bonus for reaching target

    return reward

def value_iteration(states, actions, target_xy, gamma=0.9, iterations=100):
    V = {s: 0 for s in states}         # Value function
    policy = {s: actions[0] for s in states}  # Initial dummy policy

    for _ in range(iterations):
        new_V = {}
        for state in states:
            max_value = float('-inf')
            best_action = None

            for action in actions:
                next_state = apply_action(state, action)
                reward = compute_reward(next_state, target_xy)
                value = reward + gamma * V[next_state]

                if value > max_value:
                    max_value = value
                    best_action = action

            new_V[state] = max_value
            policy[state] = best_action
        V = new_V

    return V, policy


# Discretization
angle_step = 1
angles = np.arange(0, 180, angle_step)  # 0 to 170 degrees
n_angles = len(angles)

# Define state space: all combinations of (theta1, theta2)
states = list(product(angles, angles))  # 18 x 18 = 324 states

# Define action space: change in angles
# Actions = [d_theta1, d_theta2]
angle_changes = [-angle_step, 0, angle_step]
actions = list(product(angle_changes, angle_changes))  # 9 actions

# Print sample
print(f"Total states: {len(states)}")
print(f"Sample state: {states[2]}")
print(f"Actions (angle changes): {actions}")

test_state = (90, 90)
test_action = (10, -10)

# next_state = apply_action(test_state, test_action)
# print(f"From {test_state} + {test_action} → {next_state}")

