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

# --- Planner Parameters ---
goal_threshold = 2.0
# <<< NEW: Instead of a grid, we sample actions >>>
NUM_RANDOM_SAMPLES = 200 # Number of random actions to check at each step

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
# --- 3. THE NEW SCALABLE LOCAL OPTIMIZER (REPLACES VALUE ITERATION) ---
# ===================================================================

def local_optimizer_5dof(current_state):
    """
    Solves the local planning problem by sampling random actions,
    avoiding the "Curse of Dimensionality".
    """
    best_action = np.zeros(DOF)
    best_value = -float('inf')

    # Adaptive action space
    dist_val = distance_from_goal(current_state) / 75.0
    control_step = np.clip(dist_val, 0.01, 0.05)
    possible_moves = [-control_step, 0, control_step]

    # --- Monte Carlo Action Selection ---
    # Instead of iterating through all 3^5=243 actions, we just try N random ones.
    for _ in range(NUM_RANDOM_SAMPLES):
        # 1. Create one random action for all 5 joints
        random_action = np.array([np.random.choice(possible_moves) for _ in range(DOF)])

        # 2. Evaluate this one action by seeing the reward of the next state
        next_state = current_state + random_action
        value = compute_reward(next_state)

        # 3. Keep track of the best random action found so far
        if value > best_value:
            best_value = value
            best_action = random_action

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
ax.set_title("5DOF Arm with Local Sampling Planner")
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

    # Use the new, scalable optimizer
    d_thetas = local_optimizer_5dof(thetas)
    thetas += d_thetas

    update_plot()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(mdp_controller_step, None)

ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()
