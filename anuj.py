#finalCode where we use the local MDP problem
#here i will try to do the action space based on the distance of end effector from the goal
#here we have reduce the number of the local grids points

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Problem Setup and Parameters
# ---------------------------------------------------
l1 = 5.0  # length of link 1 (cm)
l2 = 5.0  # length of link 2 (cm)

# Define goal position in task space (end-effector position)
goal = (-3.0, -4.0)
goal_threshold = 0.001  # if end-effector is within this distance, we consider goal reached

# Noise level in control execution (Gaussian noise)
sigma = 0.005

# Control inputs: we assume they are chosen from a discrete set
#control_vals = [-0.01, 0, 0.01]  # possible changes in theta1 and theta2

# Define local grid parameters for joint angles:
# At each iteration, we define a small local grid centered around the current joint angles.
local_delta = 0.00872665  # half-width of the local grid (radians)
num_local_points = 5     # number of grid points for each angle

# Discount factor for value iteration
gamma = 0.9

# Convergence threshold for value iteration in the local grid
vi_threshold = 0.05

# Maximum number of local MDP updates (global iterations)
max_global_iterations = 50

# ---------------------------------------------------
# 2. Forward Kinematics and Reward Functions
# ---------------------------------------------------
def forward_kinematics(theta1, theta2):
    """
    Given joint angles theta1 and theta2, compute the end-effector (x, y).
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return (x, y)

def compute_reward(theta1, theta2):
    """
    Compute reward for state (theta1, theta2) based on the end-effector position.
    If the distance to the goal is within goal_threshold, return +100,
    otherwise, return -distance.
    """
    x, y = forward_kinematics(theta1, theta2)
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    if dist < goal_threshold:
        return 100
    else:
        return -dist
    
def distance_from_goal(theta1, theta2):
    x, y = forward_kinematics(theta1, theta2)
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    return dist

# ---------------------------------------------------
# 3. Local MDP Solver using Value Iteration
# ---------------------------------------------------
def local_value_iteration(current_state):
    """
    Solve a local MDP around the current state (theta1, theta2) using a small grid.
    Returns the optimal action (Δθ1, Δθ2) for the local MDP.
    """
    theta1_center, theta2_center = current_state

    # Define local grid for theta1 and theta2 around the current state:
    theta1_grid = np.linspace(theta1_center - local_delta, theta1_center + local_delta, num_local_points)
    theta2_grid = np.linspace(theta2_center - local_delta, theta2_center + local_delta, num_local_points)
    dist_val= distance_from_goal(theta1_center, theta2_center)/10
    #con_val=max(dist_val,0.001)
    control_vals = [-dist_val, 0, dist_val] # Action Space: We can change it based on the admissible inputs
    
    # Create local state space as a list of (theta1, theta2) tuples.
    local_states = [(t1, t2) for t1 in theta1_grid for t2 in theta2_grid]
    
    # Initialize value function for the local grid
    V_local = {state: 0 for state in local_states}
    
    iteration = 0
    while True:
        delta_local = 0
        V_new = {}
        for state in local_states:
            t1, t2 = state
            best_value = -np.inf
            # Evaluate each possible control action from this state
            for dtheta1 in control_vals:
                for dtheta2 in control_vals:
                    # Deterministic update plus Gaussian noise sample for expectation
                    samples = []
                    num_samples = 5
                    for _ in range(num_samples):
                        # Transition: new angles = current angles + control + noise
                        t1_next = t1 + dtheta1 + np.random.normal(0, sigma)
                        t2_next = t2 + dtheta2 + np.random.normal(0, sigma)
                        # Clip to valid range [-pi, pi]
                        t1_next = np.clip(t1_next, -np.pi, np.pi)
                        t2_next = np.clip(t2_next, -np.pi, np.pi)
                        # Map next state to the nearest state in the local grid:
                        # We use simple nearest-neighbor search over the local_states
                        closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_next, t2_next])))
                        samples.append(V_local[closest_state])
                    expected_V = np.mean(samples)
                    # Bellman update: immediate reward + discounted future value
                    val = compute_reward(t1, t2) + gamma * expected_V
                    if val > best_value:
                        best_value = val
            V_new[state] = best_value
            delta_local = max(delta_local, abs(V_new[state] - V_local[state]))
        V_local = V_new
        iteration += 1
        if delta_local < vi_threshold:
            break
    
    # Now, extract the best action for the current state from the local grid:
    best_action = None
    best_val = -np.inf
    for dtheta1 in control_vals:
        for dtheta2 in control_vals:
            samples = []
            for _ in range(5):
                t1_next = theta1_center + dtheta1 + np.random.normal(0, sigma)
                t2_next = theta2_center + dtheta2 + np.random.normal(0, sigma)
                t1_next = np.clip(t1_next, -np.pi, np.pi)
                t2_next = np.clip(t2_next, -np.pi, np.pi)
                closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_next, t2_next])))
                samples.append(V_local[closest_state])
            expected_V = np.mean(samples)
            val = compute_reward(theta1_center, theta2_center) + gamma * expected_V
            if val > best_val:
                best_val = val
                best_action = (dtheta1, dtheta2)
    
    return best_action

# ---------------------------------------------------
# 4. Global Loop to Update the State using Local MDPs
# ---------------------------------------------------
# Start from an initial configuration (theta1, theta2) = (0, 0)
current_state = (0.0, 0.0)
global_states = [current_state]  # to record the trajectory
global_actions = []               # record chosen actions

max_global_iterations = 100
for i in range(max_global_iterations):
    # Solve local MDP around the current state using value iteration to get optimal action.
    optimal_action = local_value_iteration(current_state)
    global_actions.append(optimal_action)
    
    # Update state with the chosen control inputs (simulate transition with noise)
    theta1, theta2 = current_state
    dtheta1, dtheta2 = optimal_action
#     theta1_new = theta1 + dtheta1 + np.random.normal(0, sigma)
#     theta2_new = theta2 + dtheta2 + np.random.normal(0, sigma)
    
    theta1_new = theta1 + dtheta1
    theta2_new = theta2 + dtheta2
    
    theta1_new = np.clip(theta1_new, -np.pi, np.pi)
    theta2_new = np.clip(theta2_new, -np.pi, np.pi)
    current_state = (theta1_new, theta2_new)
    global_states.append(current_state)
    
    # Compute end-effector position
    x, y = forward_kinematics(theta1_new, theta2_new)
    print(f"Iteration {i+1}:")
    print(f"optimal control taken: {optimal_action}")
    dist_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    print(f"Current state (theta1, theta2): {current_state}, End-effector: ({x:.5f}, {y:.5f}), Distance to goal: {dist_to_goal:.5f}")
    
    if dist_to_goal < goal_threshold:
        print("Goal reached!")
        break

# ---------------------------------------------------
# 5. (Optional) Visualize the Trajectory in Task Space
# ---------------------------------------------------

trajectory = [forward_kinematics(*state) for state in global_states]
trajectory = np.array(trajectory)
x2,y2=trajectory[-1]
theta1,theta2=global_states[-1]
x1 = l1 * np.cos(theta1)
y1 = l1 * np.sin(theta1)

plt.figure(figsize=(6,6))
plt.plot([0, x1], [0, y1], marker='o', linestyle='-', color='b', label="Link 1")
plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color='r', label="Link 2")
plt.plot(trajectory[:,0], trajectory[:,1], marker='o', label="End-effector Trajectory")
plt.scatter(goal[0], goal[1], color='red', marker='X', s=100, label="Goal")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("End-effector Trajectory via Local MDP Updates")
plt.legend()
plt.grid()
plt.show()
