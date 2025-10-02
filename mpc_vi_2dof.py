import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import itertools

# ===================================================================
# --- 1. System Parameters ---
# ===================================================================

# --- Arm Link Lengths ---
L1 = 75.0
L2 = 75.0

# --- Target in Task Space ---
target_x, target_y = -100.0, 20.0
goal = (target_x, target_y)

# --- Initial State in Joint Space ---
theta1 = np.radians(100)
theta2 = np.radians(30)

# --- MPC Controller Parameters (<<< MODIFIED FOR RECURSIVE SEARCH >>>) ---
# H: How many steps to look ahead into the future.
# NOTE: Must be kept very small (e.g., 4-5) for this recursive approach.
PLANNING_HORIZON = 4
# A discount factor, similar to gamma in RL
GAMMA = 0.95 

# --- General Parameters ---
goal_threshold = 2.0  # How close is "close enough"

# ===================================================================
# --- 2. System Model and Cost Function ---
# ===================================================================

def forward_kinematics(t1, t2):
    """ The 'M' in MPC: The Model. Predicts end-effector position. """
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return (x, y)

def distance_from_goal(t1, t2):
    """ Helper function to calculate cost. """
    x, y = forward_kinematics(t1, t2)
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

# ===================================================================
# --- 3. The MPC Optimizer (<<< REWRITTEN WITH RECURSIVE BELLMAN-LIKE LOGIC >>>) ---
# ===================================================================

# Using a cache (memoization) to speed up the recursive search significantly.
# This is a key aspect of Dynamic Programming.
memoization_cache = {}

def find_best_sequence_recursively(state, steps_remaining):
    """
    This function recursively finds the best action sequence from a given state.
    It's a naive implementation of Dynamic Programming, which is the
    essence of the Bellman equation for finite-horizon problems.
    """
    # Use a tuple for the state so it can be a dictionary key
    state_tuple = tuple(state)
    
    # Check if we've already computed the result for this state and depth
    if (state_tuple, steps_remaining) in memoization_cache:
        return memoization_cache[(state_tuple, steps_remaining)]

    # === BASE CASE of the RECURSION ===
    # If we are at the end of the horizon, there's no more cost to accumulate.
    if steps_remaining == 0:
        return 0, []

    min_cost = float('inf')
    best_sequence = []

    # Define the magnitude of actions to try for the current state
    dist_val = distance_from_goal(state[0], state[1]) / 50.0
    control_step = np.clip(dist_val, 0.005, 0.05)
    possible_moves = [-control_step, 0, control_step]
    possible_actions = list(itertools.product(possible_moves, repeat=2))
    
    # === RECURSIVE STEP ===
    # Evaluate all possible first actions from the current state
    for action in possible_actions:
        next_state = state + np.array(action)
        
        # Immediate cost is the distance after taking this one action
        immediate_cost = distance_from_goal(next_state[0], next_state[1])
        
        # Recursively find the minimum cost for the rest of the path
        # This is where the Bellman-like structure lies:
        # Cost(s) = Cost(s,a) + Future_Cost(s')
        future_cost, future_sequence = find_best_sequence_recursively(next_state, steps_remaining - 1)
        
        # Apply discount factor to future costs
        total_cost = immediate_cost + GAMMA * future_cost
        
        if total_cost < min_cost:
            min_cost = total_cost
            # The best sequence is this action followed by the best future sequence
            best_sequence = [action] + future_sequence

    # Store the result in the cache before returning
    memoization_cache[(state_tuple, steps_remaining)] = (min_cost, best_sequence)
    return min_cost, best_sequence


def mpc_optimizer(current_state):
    """
    Finds the best action by initiating a recursive search that embodies
    the Bellman equation logic.
    """
    # Clear the cache at the start of each new planning step
    global memoization_cache
    memoization_cache = {}

    _, best_sequence = find_best_sequence_recursively(np.array(current_state), PLANNING_HORIZON)

    # Return only the FIRST action of the best found sequence
    return best_sequence[0] if best_sequence else np.array([0, 0])


# ===================================================================
# --- 4. VISUALIZATION SETUP (Mostly Unchanged) ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', linewidth=1, label='Path')
path_history = []

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("2DOF Arm with MPC (Recursive Bellman-like Planner)")
ax.legend()

ax_theta1 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.1, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))

# ===================================================================
# --- 5. Main Control Loop (Now an MPC Loop) ---
# ===================================================================

def update_plot():
    """This function is now ONLY for drawing, not for calculation."""
    x0, y0 = 0, 0
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    
    path_history.append((x2, y2))
    if len(path_history) > 1:
        path_x, path_y = zip(*path_history)
        end_effector_path.set_data(path_x, path_y)
    
    slider1.set_val(np.degrees(theta1))
    slider2.set_val(np.degrees(theta2))
    fig.canvas.draw_idle()

def mpc_control_step(event):
    """ The main MPC loop: sense, plan, act. """
    global theta1, theta2

    # --- SENSE ---
    current_state = (theta1, theta2)
    current_dist = distance_from_goal(current_state[0], current_state[1])
    print(f"Distance to Goal: {current_dist:.2f}")

    if current_dist < goal_threshold:
        print("Goal Reached!")
        timer.stop()
        return

    # --- PREDICT & OPTIMIZE ---
    # Call the MPC planner to get the single best action for this instant
    best_first_action = mpc_optimizer(current_state)
    dtheta1, dtheta2 = best_first_action
    
    # --- ACT ---
    # Apply only that first action
    theta1 += dtheta1
    theta2 += dtheta2
    
    # Redraw the plot with the new state
    update_plot()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=150) # Increased interval for more computation
timer.add_callback(mpc_control_step, None)

ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()

