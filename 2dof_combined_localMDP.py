import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ===================================================================
# <<< 1. MERGED PARAMETERS FROM BOTH SCRIPTS >>>
# ===================================================================

# --- Visualization & Arm Parameters ---
L1 = 75.0  # Link 1 length (was l1 in MDP code)
L2 = 75.0  # Link 2 length (was l2 in MDP code)

# --- Target in Task Space ---
target_x, target_y = 0, 150
goal = (target_x, target_y)

# --- Initial State in Joint Space ---
theta1 = np.radians(190)
theta2 = np.radians(190)

# --- Local MDP Solver Parameters ---
goal_threshold = 2.0  # Increased threshold for visual satisfaction
sigma = 0.005         # Noise for MDP model (optional, can be 0 for deterministic)
gamma = 0.9           # Discount factor
vi_threshold = 0.05   # Value iteration convergence threshold
local_delta = 0.05    # Half-width of the local grid (radians)
num_local_points = 5  # Number of grid points for each angle

# ===================================================================
# <<< 2. HELPER & LOCAL MDP FUNCTIONS (FROM YOUR FIRST SCRIPT) >>>
# ===================================================================

def forward_kinematics(t1, t2):
    """
    Given joint angles, compute the end-effector (x, y).
    Uses L1, L2 from the global scope.
    """
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return (x, y)

def distance_from_goal(t1, t2):
    x, y = forward_kinematics(t1, t2)
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

def compute_reward(t1, t2):
    # dist = distance_from_goal(t1, t2)
    # if dist < goal_threshold:
    #     return 100
    # else:
    #     return -dist
    x2, y2 = forward_kinematics(t1, t2)
    return(-abs(x2)*100 + abs(y2)*100)

def local_value_iteration(current_state):
    """
    Solve a local MDP around the current state (theta1, theta2).
    Returns the optimal action (Δθ1, Δθ2).
    """
    theta1_center, theta2_center = current_state
    
    # Define local grid
    theta1_grid = np.linspace(theta1_center - local_delta, theta1_center + local_delta, num_local_points)
    theta2_grid = np.linspace(theta2_center - local_delta, theta2_center + local_delta, num_local_points)
    local_states = [(t1, t2) for t1 in theta1_grid for t2 in theta2_grid]
    
    # Adaptive action space based on distance
    dist_val = distance_from_goal(theta1_center, theta2_center) / 50 # Scaled down for larger links
    control_step = np.clip(dist_val, 0.005, 0.05) # Clip to avoid too large/small steps
    # control_vals = [-control_step, 0, control_step]
    control_vals = [-0.05, 0, 0.05]
    
    V_local = {state: 0 for state in local_states}
    
    # Value Iteration on the local grid
    policy = {s: (0,0) for s in local_states} 
    for _ in range(15): # Using a fixed number of iterations for speed
        delta_local = 0
        V_new = {}

        for state in local_states:
            t1, t2 = state
            best_value = -np.inf
            best_action = (0, 0)
            for dtheta1 in control_vals:
                for dtheta2 in control_vals:
                    t1_next = t1 + dtheta1 + np.random.normal(0, 0.01)
                    t2_next = t2 + dtheta2 + np.random.normal(0, 0.01)
                    closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_next, t2_next])))
                    val = compute_reward(t1_next, t2_next) + gamma * V_local[closest_state]
                    if val > best_value:
                        best_value = val
                        best_action = (dtheta1, dtheta2)

            V_new[state] = best_value
            policy[state] = best_action
            delta_local = max(delta_local, abs(V_new[state] - V_local[state]))

        V_local = V_new
        if delta_local < vi_threshold:
            break

    # Extract best action for the current_state
    # best_action = (0, 0)
    # best_val = -np.inf
    # for dtheta1 in control_vals:
    #     for dtheta2 in control_vals:
    #         t1_next = theta1_center + dtheta1
    #         t2_next = theta2_center + dtheta2
    #         closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_next, t2_next])))
    #         val = compute_reward(theta1_center, theta2_center) + gamma * V_local[closest_state]
    #         if val > best_val:
    #             best_val = val
    #             best_action = (dtheta1, dtheta2)
                
    # return best_action
    return policy[min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array(current_state)))]

# ===================================================================
# <<< 3. VISUALIZATION SETUP (MOSTLY UNCHANGED) >>>
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = ax.plot([], [], 'g:', linewidth=1, label='Path')
path_history = []

ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("2DOF Arm Controlled by Local MDPs")
ax.legend()

ax_theta1 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.1, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))

# ===================================================================
# <<< 4. MODIFIED UPDATE AND ANIMATION LOGIC >>>
# ===================================================================

def update_plot():
    """This function is now ONLY for drawing, not for calculation."""
    # Joint positions using Forward Kinematics
    x0, y0 = 0, 0
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2)
    
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    
    # Update and draw the end-effector path
    path_history.append((x2, y2))
    path_x, path_y = zip(*path_history)
    end_effector_path.set_data(path_x, path_y)
    
    # Update sliders
    slider1.set_val(np.degrees(theta1))
    slider2.set_val(np.degrees(theta2))
    
    fig.canvas.draw_idle()

def mdp_controller_step(event):
    """
    This is the new "brain" of the animation. It runs the MDP solver
    and updates the state for the next frame.
    """
    global theta1, theta2
    
    current_dist = distance_from_goal(theta1, theta2)
    print(f"Current Distance to Goal: {current_dist:.2f}")

    # Check for goal condition
    if current_dist < goal_threshold:
        print("Goal Reached!")
        timer.stop()
        return

    # 1. Get the current state
    current_state = (theta1, theta2)
    
    # 2. Solve the local MDP to find the best action
    dtheta1, dtheta2 = local_value_iteration(current_state)
    
    # 3. Apply the action to update the state
    theta1 += dtheta1
    theta2 += dtheta2
    
    # 4. Redraw the plot with the new state
    update_plot()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=50)  # ~20 FPS
timer.add_callback(mdp_controller_step, None)

# --- Add a button to start the simulation ---
ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()