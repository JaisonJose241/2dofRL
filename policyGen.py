import numpy as np
import math
import sys
import pickle
from itertools import product

# Link lengths
L1 = 75
L2 = 75

target_x, target_y = 10, 50
target = target_x, target_y

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

    next_pose = update_plot()
    return (new_theta1, new_theta2), next_pose

def compute_reward(next_pose, target, threshold=1):

    target_x, target_y = target
    x2, y2 = next_pose

    # calculate error by distance
    # error = ((target_y - y2)**2 + (target_x - x2)**2)**(1/2)

    # distance = error
    # Option 1: shaped reward (negative distance)
    # reward = -distance
    reward = -abs(x2)*100 + abs(-200-y2)*30

    # Option 2: bonus for being very close
    # if distance < threshold:
    #     reward += 100  # big bonus for reaching target
        # print(y2)
        # print("got high reward")

    # # Option 2: bonus for being very close
    # if distance < 4*threshold:
    #     reward += 50  # big bonus for reaching target
    #     # print("got high reward")

    return reward

def value_iteration(states, actions, gamma=0.9, iterations=10):
    global error
    V = {s: 0 for s in states}         # Value function
    policy = {s: actions[0] for s in states}  # Initial dummy policy
    max_value = 0

    for itr in range(iterations):
        new_V = {}
        index = 0
        for state in states:
            
            index += 1
            max_value = float('-inf')
            best_action = None

            for action in actions:
                next_state, next_pose = apply_action(state, action)
                reward = compute_reward(next_pose, target)
                value = reward + gamma * V[next_state]

                if value > max_value:
                    max_value = value
                    best_action = action
                
                # print(itr, index, state, action, reward, value)
                # update_plot()
                # plt.pause(0.001)
                # plt.plot(block=False)


            new_V[state] = max_value
            policy[state] = best_action
        V = new_V
        # print(policy)
        print(itr, max_value)

    return V, policy

def update_plot():
    global theta1, theta2, theta1_vel, theta2_vel, error
    
    # Joint positions
    x0, y0 = 0, 0
    x1 = L1 * np.cos(np.radians(theta1))
    y1 = L1 * np.sin(np.radians(theta1))
    x2 = x1 + L2 * np.cos(np.radians(theta1) + np.radians(theta2))
    y2 = y1 + L2 * np.sin(np.radians(theta1) + np.radians(theta2))


    return (x2, y2)



# try:
# update_plot()
# plt.show(block=False)
value, policy = value_iteration(states, actions)
# print(policy)
with open('policy.pkl', 'wb') as f:
    pickle.dump(policy, f)
    
# except Exception as e:
#     print(e)
#     sys.exit()