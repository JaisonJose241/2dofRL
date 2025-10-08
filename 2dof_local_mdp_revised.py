import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ===================================================================
# --- 1. System Parameters (Unchanged) ---
# ===================================================================

# --- Visualization & Arm Parameters ---
L1 = 75.0
L2 = 75.0

# --- Target in Task Space ---
target_x, target_y = 100, 0
goal = (target_x, target_y)

# --- Initial State in Joint Space ---
theta1 = np.radians(190)
theta2 = np.radians(190)

# --- Local MDP Solver Parameters ---
goal_threshold = 2.0
gamma = 0.9           # Discount factor
vi_threshold = 0.05   # Value iteration convergence threshold
local_delta = 0.5    # Half-width of the local grid (radians)
num_local_points = 5  # Number of grid points for each angle

# ===================================================================
# --- 2. Helper Functions (Unchanged) ---
# ===================================================================

def forward_kinematics(t1, t2):
    """
    Given joint angles, compute the end-effector (x, y).
    """
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return (x, y)

def distance_from_goal(t1, t2):
    x, y = forward_kinematics(t1, t2)
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

def compute_reward(t1, t2):
    """
    Reward is the negative distance to the goal. A high positive reward
    is given for reaching the goal. This function also serves as our heuristic.
    """
    # dist = distance_from_goal(t1, t2)
    # if dist < goal_threshold:
        # return 100
    # else:
        # return -dist
    x2, y2 = forward_kinematics(t1, t2)
    return(-abs(x2)*100 + abs(y2)*100)

# ===================================================================
# --- 3. The Local MDP Solver (<<< MODIFIED WITH HEURISTIC >>>) ---
# ===================================================================

def local_value_iteration(current_state):
    """
    Solves a local MDP using a heuristic terminal cost for out-of-bounds states
    to eliminate the "Edge Effect".
    """
    theta1_center, theta2_center = current_state
    
    # Define local grid and its boundaries
    theta1_grid = np.linspace(theta1_center - local_delta, theta1_center + local_delta, num_local_points)
    theta2_grid = np.linspace(theta2_center - local_delta, theta2_center + local_delta, num_local_points)
    grid_bounds = {
        't1_min': theta1_grid[0], 't1_max': theta1_grid[-1],
        't2_min': theta2_grid[0], 't2_max': theta2_grid[-1]
    }
    local_states = [(t1, t2) for t1 in theta1_grid for t2 in theta2_grid]
    
    # Adaptive action space
    dist_val = distance_from_goal(theta1_center, theta2_center) / 50.0
    control_step = np.clip(dist_val, 0.005, 0.05)
    control_vals = [-control_step, 0, control_step]
    
    V_local = {state: 0 for state in local_states}
    policy = {s: (0,0) for s in local_states} 

    # Value Iteration on the local grid
    for _ in range(30):
        delta_local = 0
        for state in local_states:
            t1, t2 = state
            best_value = -np.inf
            best_action = (0, 0)
            
            for dtheta1 in control_vals:
                for dtheta2 in control_vals:
                    # Calculate the true continuous next state
                    t1_next = t1 + dtheta1
                    t2_next = t2 + dtheta2
                    
                    # --- <<< CORE OF THE HEURISTIC METHOD >>> ---
                    # Check if the next state is inside our detailed local grid
                    is_in_bounds = (grid_bounds['t1_min'] <= t1_next <= grid_bounds['t1_max'] and
                                     grid_bounds['t2_min'] <= t2_next <= grid_bounds['t2_max'])
                    
                    future_value = 0
                    if is_in_bounds:
                        # If IN BOUNDS, use the high-fidelity value from our grid
                        closest_state = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array([t1_next, t2_next])))
                        future_value = V_local[closest_state]
                        # print(future_value)
                    else:
                        # If OUT OF BOUNDS, use the heuristic (-distance) to estimate the future value.
                        # This prevents the "Edge Effect" by giving the planner a glimpse of the outside world.
                        # We use compute_reward as our heuristic, as it's the negative distance.
                        future_value = compute_reward(t1_next, t2_next)
                        # print(future_value)
                    
                    # Bellman update using the intelligently calculated future_value
                    # The immediate reward is for arriving at the next state
                    val = compute_reward(t1_next, t2_next) + gamma * future_value
                    
                    if val > best_value:
                        best_value = val
                        best_action = (dtheta1, dtheta2)

            V_local[state] = best_value
            policy[state] = best_action
            delta_local = max(delta_local, abs(best_value - V_local[state]))

        if delta_local < vi_threshold:
            break

    # Return the best action for the grid point closest to our true continuous state
    closest_grid_state_to_current = min(local_states, key=lambda s: np.linalg.norm(np.array(s) - np.array(current_state)))
    print (policy[closest_grid_state_to_current])
    return policy[closest_grid_state_to_current]

# ===================================================================
# --- 4. VISUALIZATION SETUP (Unchanged) ---
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
ax.set_title("2DOF Arm with Heuristic Local MDP")
ax.legend()

ax_theta1 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.1, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))

# ===================================================================
# --- 5. Main Control Loop (Unchanged) ---
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

def mdp_controller_step(event):
    """
    The main control loop. The only change is that it now calls the
    new and improved local_value_iteration function.
    """
    global theta1, theta2
    
    current_dist = distance_from_goal(theta1, theta2)
    print(f"Current Distance to Goal: {current_dist:.2f}")

    if current_dist < goal_threshold:
        print("Goal Reached!")
        timer.stop()
        return

    current_state = (theta1, theta2)
    dtheta1, dtheta2 = local_value_iteration(current_state)
    
    theta1 += dtheta1
    theta2 += dtheta2
    
    update_plot()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=100)
timer.add_callback(mdp_controller_step, None)

ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()
