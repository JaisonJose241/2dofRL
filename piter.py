import numpy as np
import math
import sys
import pickle
from itertools import product

# Link lengths
L1 = 75
L2 = 75

target_x, target_y = -100, 70

# Initial joint angles (radians)
theta1 = np.radians(100)
theta2 = np.radians(30)
error = 1000

# Initial angular velocities (radians/frame)
theta1_vel = 0.0
theta2_vel = 0.0

# Discretization
angle_step = 1
angles = np.arange(0, 180, angle_step)  # 0 to 180 degrees
angles_t2 = np.arange(0, 180, angle_step)  # 0 to 180 degrees
# print(angles_t2)

# Define state space: all combinations of (theta1, theta2)threshold
states = list(product(angles, angles_t2))  # 18 x 18 = 324 states
# print(states)

# Define action space: change in angles
# Actions = [d_theta1, d_theta2]
angle_changes = [-angle_step, 0, angle_step]
actions = list(product(angle_changes, angle_changes))  # 9 actions

# Print sample
print(f"Total states: {len(states)}")
print(f"Sample state: {states[0]}")
print(f"Actions (angle changes): {actions}")

# --- Update function ---
def apply_action(state, action):
    global theta1, theta2
    theta1, theta2 = state
    d_theta1, d_theta2 = action

    # Apply the action
    new_theta1 = np.clip(theta1 + d_theta1, 0, 170)
    new_theta2 = np.clip(theta2 + d_theta2, 0, 170)

    theta1, theta2 = new_theta1, new_theta2
    return (new_theta1, new_theta2)

def compute_reward(error, threshold=10):
    distance = error
    # Option 1: shaped reward (negative distance)
    reward = -distance

    # Option 2: bonus for being very close
    if distance < threshold:
        reward += 100  # big bonus for reaching target
        # print("got high reward")

    # # Option 2: bonus for being very close
    # if distance < 4*threshold:
    #     reward += 50  # big bonus for reaching target
    #     # print("got high reward")

    return reward


def policy_iteration(states, actions, gamma=0.9, eval_iters=20):
    global error

    # Step 0: Initialize random policy
    V = {s: 0.0 for s in states}
    policy = {s: actions[0] for s in states}
    
    itr = 0
    is_policy_stable = False
    while not is_policy_stable:
        itr += 1
        # Step 1: Policy Evaluation
        for _ in range(eval_iters):
            new_V = {}
            for s in states:
                a = policy[s]
                s_prime = apply_action(s, a)
                update_plot()
                r = compute_reward(error)
                new_V[s] = r + gamma * V[s_prime]
            V = new_V

        # Step 2: Policy Improvement
        is_policy_stable = True
        for s in states:
            old_action = policy[s]
            best_value = float('-inf')
            best_action = None
            for a in actions:
                s_prime = apply_action(s, a)
                update_plot()
                r = compute_reward(error)
                value = r + gamma * V[s_prime]
                if value > best_value:
                    best_value = value
                    best_action = a
                # print(s, s_prime, r, error, value, best_value, best_action)
            policy[s] = best_action
            if best_action != old_action:
                is_policy_stable = False
        
        print(itr, best_value)

    return V, policy


def update_plot():
    global theta1, theta2, theta1_vel, theta2_vel, error
    
    # Joint positions
    x0, y0 = 0, 0
    x1 = L1 * np.cos(np.radians(theta1))
    y1 = L1 * np.sin(np.radians(theta1))
    x2 = x1 + L2 * np.cos(np.radians(theta1) + np.radians(theta2))
    y2 = y1 + L2 * np.sin(np.radians(theta1) + np.radians(theta2))

    # calculate error by distance
    error = ((target_y - y2)**2 + (target_x - x2)**2)**(1/2)



# try:
# update_plot()
# plt.show(block=False)
value, policy = policy_iteration(states, actions)
# print(policy)
with open('policypi.pkl', 'wb') as f:
    pickle.dump(policy, f)
    
# except Exception as e:
#     print(e)
#     sys.exit()