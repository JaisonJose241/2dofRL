import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import itertools

# ===================================================================
# --- 1. UPDATED PARAMETERS FOR 5DOF ARM ---
# ===================================================================

# --- Arm Link Lengths (made shorter to fit on screen) ---
DOF = 5
LINK_LENGTHS = [40.0, 40.0, 35.0, 35.0, 30.0]

# --- Target in Task Space ---
target_x, target_y = 90.0, 120.0
goal = (target_x, target_y)

# --- Initial State in Joint Space ---
thetas = np.radians(np.array([120, 45, -30, 20, 10], dtype=float))

# --- Local MDP Solver Parameters ---
goal_threshold = 2.0
gamma = 0.9
vi_threshold = 0.1
local_delta = 0.1
# NOTE: num_local_points=3 creates 3^5 = 243 states. This will be VERY SLOW.
num_local_points = 3 

# ===================================================================
# --- 2. GENERALIZED HELPER FUNCTIONS FOR N-DOF ---
# ===================================================================

def forward_kinematics(current_thetas):
    """Given N joint angles, compute the end-effector (x, y)."""
    x, y, angle_sum = 0.0, 0.0, 0.0
    for i in range(DOF):
        angle_sum += current_thetas[i]
        x += LINK_LENGTHS[i] * np.cos(angle_sum)
        y += LINK_LENGTHS[i] * np.sin(angle_sum)
    return (x, y)

def get_full_arm_coords(current_thetas):
    """Helper to get coordinates of all N joints for plotting."""
    coords_x, coords_y = [0.0], [0.0]
    x, y, angle_sum = 0.0, 0.0, 0.0
    for i in range(DOF):
        angle_sum += current_thetas[i]
        x += LINK_LENGTHS[i] * np.cos(angle_sum)
        y += LINK_LENGTHS[i] * np.sin(angle_sum)
        coords_x.append(x)
        coords_y.append(y)
    return (coords_x, coords_y)

def distance_from_goal(current_thetas):
    x, y = forward_kinematics(current_thetas)
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

def compute_reward(current_thetas):
    dist = distance_from_goal(current_thetas)
    if dist < goal_threshold:
        return 100
    else:
        return -dist

# ===================================================================
# --- 3. THE LOCAL MDP VALUE ITERATION SOLVER FOR 5DOF ---
# ===================================================================

def local_value_iteration_5dof(current_state):
    """
    Solves a local MDP for the 5DOF arm using value iteration.
    WARNING: This is computationally very expensive due to the high dimensionality.
    """
    # Define 5D local grid
    # Create a list of 5 linspace arrays, one for each joint
    grid_axes = [np.linspace(current_state[i] - local_delta, current_state[i] + local_delta, num_local_points) for i in range(DOF)]
    # Use itertools.product to create the 5D grid states
    local_states = list(itertools.product(*grid_axes))
    
    # Adaptive action space
    dist_val = distance_from_goal(current_state) / 75.0
    control_step = np.clip(dist_val, 0.01, 0.05)
    control_vals = [-control_step, 0, control_step]
    
    # Create all 3^5 = 243 possible actions
    possible_actions = list(itertools.product(control_vals, repeat=DOF))
    
    V_local = {state: 0 for state in local_states}
    
    # Value Iteration on the 5D local grid
    for _ in range(5): # Keeping iterations low to manage performance
        delta_local = 0
        for state in local_states:
            v_old = V_local[state]
            best_value = -float('inf')
            
            # Evaluate all 243 actions from this state
            for action in possible_actions:
                next_state_continuous = np.array(state) + np.array(action)
                # This part is slow as it searches the 243 grid points
                closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - next_state_continuous))
                
                # We use the reward of the state we are evaluating, not the next one
                val = compute_reward(state) + gamma * V_local[closest_state]
                if val > best_value:
                    best_value = val
            
            V_local[state] = best_value
            delta_local = max(delta_local, abs(v_old - V_local[state]))

        if delta_local < vi_threshold:
            break
            
    # Policy Extraction for the current_state
    best_action = np.zeros(DOF)
    best_val = -float('inf')
    for action in possible_actions:
        next_state_continuous = np.array(current_state) + np.array(action)
        closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - next_state_continuous))
        val = compute_reward(current_state) + gamma * V_local[closest_state]
        if val > best_val:
            best_val = val
            best_action = action
            
    return best_action

# ===================================================================
# --- 4. VISUALIZATION SETUP WITH 5 SLIDERS ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.40) # Made more room at bottom
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', linewidth=1, label='Path')
path_history = []

total_arm_length = sum(LINK_LENGTHS)
ax.set_xlim(-total_arm_length, total_arm_length)
ax.set_ylim(-total_arm_length, total_arm_length)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("5DOF Arm with Local MDP (Value Iteration)")
ax.legend()

# Axes for five sliders
slider_axes = []
sliders = []
slider_height = 0.03
slider_spacing = 0.05
for i in range(DOF):
    ax_pos = [0.25, 0.30 - i * slider_spacing, 0.65, slider_height]
    slider_axes.append(plt.axes(ax_pos))
    sliders.append(Slider(slider_axes[i], f'Theta{i+1} (°)', -180, 180, valinit=np.degrees(thetas[i])))

# ===================================================================
# --- 5. MODIFIED UPDATE AND ANIMATION LOGIC FOR 5DOF ---
# ===================================================================

def update_plot():
    """This function ONLY draws the 5-link arm."""
    arm_coords_x, arm_coords_y = get_full_arm_coords(thetas)
    arm_line.set_data(arm_coords_x, arm_coords_y)

    path_history.append((arm_coords_x[-1], arm_coords_y[-1]))
    if len(path_history) > 1:
      path_x, path_y = zip(*path_history)
      end_effector_path.set_data(path_x, path_y)

    for i in range(DOF):
        sliders[i].set_val(np.degrees(thetas[i]))

    fig.canvas.draw_idle()

def mdp_controller_step(event):
    """The 'brain' of the animation, now for 5DOF."""
    global thetas

    current_dist = distance_from_goal(thetas)
    print(f"Current Distance: {current_dist:.2f}")

    if current_dist < goal_threshold:
        print("Goal Reached!")
        timer.stop()
        return

    # Use the value iteration based planner
    d_thetas = local_value_iteration_5dof(thetas)
    thetas += np.array(d_thetas)

    update_plot()

# --- Setup and start the animation timer ---
# NOTE: Interval is very long to allow for heavy computation
timer = fig.canvas.new_timer(interval=500)
timer.add_callback(mdp_controller_step, None)

ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()

