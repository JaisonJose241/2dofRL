import numpy as np
import pickle
from itertools import product

# Link lengths
L1, L2 = 75, 75
target = (-50, 80)

# Discretization
angle_step = 5
angles = np.arange(0, 180, angle_step)
states = list(product(angles, angles))  # (theta1, theta2)

# Actions: change in angles
angle_changes = [-angle_step, 0, angle_step]
actions = list(product(angle_changes, angle_changes))  # 9 actions

# --- Environment dynamics ---
def apply_action(state, action):
    theta1, theta2 = state
    d_theta1, d_theta2 = action
    new_theta1 = np.clip(theta1 + d_theta1, 0, 170)
    new_theta2 = np.clip(theta2 + d_theta2, 0, 170)
    next_pose = update_pose(new_theta1, new_theta2)
    return (new_theta1, new_theta2), next_pose

def update_pose(theta1, theta2):
    x1 = L1 * np.cos(np.radians(theta1))
    y1 = L1 * np.sin(np.radians(theta1))
    x2 = x1 + L2 * np.cos(np.radians(theta1) + np.radians(theta2))
    y2 = y1 + L2 * np.sin(np.radians(theta1) + np.radians(theta2))
    return (x2, y2)

def compute_reward(next_pose, target, threshold=5):
    tx, ty = target
    x2, y2 = next_pose
    dist = np.sqrt((tx - x2)**2 + (ty - y2)**2)
    reward = -dist
    if dist < threshold:
        reward += 100  # bonus for reaching
    return reward

# --- Q-learning ---
def q_learning(episodes=6000, alpha=0.2, gamma=0.95, epsilon=0.2):
    Q = {}
    for s in states:
        for a in actions:
            Q[(s, a)] = 0.0

    for ep in range(episodes):
        # Start random state
        state = states[np.random.randint(len(states))]

        R = []
        for t in range(200):  # episode length limit
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = actions[np.random.randint(len(actions))]
            else:
                action = max(actions, key=lambda a: Q[(state, a)])

            # Take step
            next_state, next_pose = apply_action(state, action)
            reward = compute_reward(next_pose, target)

            # Q-learning update
            best_next = max(Q[(next_state, a)] for a in actions)
            Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])

            state = next_state
            R.append(reward)

        if ep % 500 == 0:
            print(f"Episode {ep} done", max(R))

    return Q

# Train
Q = q_learning()
# print(Q)
with open('qtable.pkl', 'wb') as f:
    pickle.dump(Q, f)
print("Training finished.")
