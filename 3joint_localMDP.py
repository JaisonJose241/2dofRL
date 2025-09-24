import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ===================================================================
# <<< 1. UPDATED PARAMETERS FOR 3DOF ARM >>>
# ===================================================================

# --- Arm Link Lengths ---
L1 = 60.0
L2 = 60.0
L3 = 50.0  # <<< Added third link

# --- Target in Task Space ---
target_x, target_y = -80.0, 100.0
goal = (target_x, target_y)

# --- Initial State in Joint Space ---
theta1 = np.radians(140)
theta2 = np.radians(170)
theta3 = np.radians(-150) # <<< Added third angle

# --- Local MDP Solver Parameters ---
goal_threshold = 2.0
gamma = 0.9
vi_threshold = 0.05
# <<< Reduced local grid size to manage computational load >>>
local_delta = 0.1
num_local_points = 3 # <<< Reduced from 5 to 3 (3^3=27 states vs 5^3=125)

# ===================================================================
# <<< 2. MODIFIED HELPER & LOCAL MDP FUNCTIONS FOR 3DOF >>>
# ===================================================================

def forward_kinematics(t1, t2, t3): # <<< Now takes 3 angles
    """Given 3 joint angles, compute the end-effector (x, y)."""
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)
    x3 = x2 + L3 * np.cos(t1 + t2 + t3) # <<< Calculate 3rd joint
    y3 = y2 + L3 * np.sin(t1 + t2 + t3)
    return (x3, y3)

def get_full_arm_coords(t1, t2, t3):
    """Helper to get coordinates of all joints for plotting."""
    x0, y0 = 0, 0
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)
    x3 = x2 + L3 * np.cos(t1 + t2 + t3)
    y3 = y2 + L3 * np.sin(t1 + t2 + t3)
    return ([x0, x1, x2, x3], [y0, y1, y2, y3])

def distance_from_goal(t1, t2, t3):
    x, y = forward_kinematics(t1, t2, t3)
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

def compute_reward(t1, t2, t3):
    # dist = distance_from_goal(t1, t2, t3)
    # if dist < goal_threshold:
    #     return 100
    # else:
    #     return -dist
    
    x2, y2 = forward_kinematics(t1, t2, t3)
    return(-abs(x2)*100 + abs(y2)*100)

def local_value_iteration(current_state):
    """Solves a local MDP for the 3DOF arm."""
    theta1_c, theta2_c, theta3_c = current_state # <<< Unpack 3 angles
    
    # Define 3D local grid
    t1_grid = np.linspace(theta1_c - local_delta, theta1_c + local_delta, num_local_points)
    t2_grid = np.linspace(theta2_c - local_delta, theta2_c + local_delta, num_local_points)
    t3_grid = np.linspace(theta3_c - local_delta, theta3_c + local_delta, num_local_points)
    local_states = [(t1, t2, t3) for t1 in t1_grid for t2 in t2_grid for t3 in t3_grid]
    
    # Adaptive action space
    dist_val = distance_from_goal(theta1_c, theta2_c, theta3_c) / 75.0
    control_step = np.clip(dist_val, 0.05, 0.1)
    control_vals = [-control_step, 0, control_step]
    
    V_local = {state: 0 for state in local_states}
    
    # Value Iteration on the 3D local grid
    for _ in range(10): # Fixed iterations for performance
        V_new = {}
        for state in local_states:
            t1, t2, t3 = state
            best_value = -np.inf
            # <<< Triple-nested loop for 3D action space >>>
            for d1 in control_vals:
                for d2 in control_vals:
                    for d3 in control_vals:
                        t1_next, t2_next, t3_next = t1 + d1, t2 + d2, t3 + d3
                        closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_next, t2_next, t3_next])))
                        val = compute_reward(t1, t2, t3) + gamma * V_local[closest_state]
                        if val > best_value:
                            best_value = val
            V_new[state] = best_value
        V_local = V_new
            
    # Extract best action for the current_state
    best_action = (0, 0, 0)
    best_val = -np.inf
    # <<< Triple-nested loop for 3D action space >>>
    for d1 in control_vals:
        for d2 in control_vals:
            for d3 in control_vals:
                t1_n, t2_n, t3_n = theta1_c + d1, theta2_c + d2, theta3_c + d3
                closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_n, t2_n, t3_n])))
                val = compute_reward(theta1_c, theta2_c, theta3_c) + gamma * V_local[closest_state]
                if val > best_val:
                    best_val = val
                    best_action = (d1, d2, d3)
                
    return best_action

# ===================================================================
# <<< 3. VISUALIZATION SETUP WITH 3 SLIDERS >>>
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.35) # <<< Made more room at bottom
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', linewidth=1, label='Path')
path_history = []

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("3DOF Arm Controlled by Local MDPs")
ax.legend()

# <<< Axes for three sliders >>>
ax_t1 = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_t2 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_t3 = plt.axes([0.25, 0.10, 0.65, 0.03])
slider1 = Slider(ax_t1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_t2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))
slider3 = Slider(ax_t3, 'Theta3 (°)', -180, 180, valinit=np.degrees(theta3))

# ===================================================================
# <<< 4. MODIFIED UPDATE AND ANIMATION LOGIC FOR 3DOF >>>
# ===================================================================

def update_plot():
    """This function ONLY draws the 3-link arm."""
    arm_coords_x, arm_coords_y = get_full_arm_coords(theta1, theta2, theta3)
    arm_line.set_data(arm_coords_x, arm_coords_y)
    
    # Update and draw the end-effector path
    path_history.append((arm_coords_x[-1], arm_coords_y[-1]))
    path_x, path_y = zip(*path_history)
    end_effector_path.set_data(path_x, path_y)
    
    # Update sliders
    slider1.set_val(np.degrees(theta1))
    slider2.set_val(np.degrees(theta2))
    slider3.set_val(np.degrees(theta3)) # <<< Update 3rd slider
    
    fig.canvas.draw_idle()

def mdp_controller_step(event):
    """The 'brain' of the animation, now for 3DOF."""
    global theta1, theta2, theta3 # <<< Control all 3 angles
    
    current_dist = distance_from_goal(theta1, theta2, theta3)
    print(f"Current Distance: {current_dist:.2f}")

    if current_dist < goal_threshold:
        print("Goal Reached!")
        timer.stop()
        return

    current_state = (theta1, theta2, theta3)
    d1, d2, d3 = local_value_iteration(current_state)
    
    theta1 += d1
    theta2 += d2
    theta3 += d3
    
    update_plot()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=100) # <<< Slower interval due to more computation
timer.add_callback(mdp_controller_step, None)

ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()