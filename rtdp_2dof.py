import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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

# ===================================================================
# --- RTDP Controller Parameters (<<< NEW >>>) ---
# ===================================================================
# H: How many steps to look ahead into the future during each trial.
PLANNING_HORIZON = 10
# N: Number of simulation trials to run at each time step to update the value function.
NUM_TRIALS = 200

# --- Discretization Parameters for the Value Function ---
# We create a grid over the joint angle space.
# The higher the number of bins, the more precise but slower the learning.
NUM_BINS_THETA1 = 100
NUM_BINS_THETA2 = 100
# Define the range of joint angles
theta1_range = (-np.pi, np.pi)
theta2_range = (-np.pi, np.pi)

# --- General Parameters ---
goal_threshold = 2.0  # How close is "close enough"

# ===================================================================
# --- 2. Global Value Function and Helper Functions (<<< NEW >>>) ---
# ===================================================================

# The persistent "memory" of the RTDP agent. Stores cost-to-go for each discrete state.
value_function = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2))

def discretize_state(t1, t2):
    """Converts continuous joint angles to discrete grid indices."""
    t1_norm = (t1 - theta1_range[0]) / (theta1_range[1] - theta1_range[0])
    t2_norm = (t2 - theta2_range[0]) / (theta2_range[1] - theta2_range[0])
    
    t1_idx = int(t1_norm * (NUM_BINS_THETA1 - 1))
    t2_idx = int(t2_norm * (NUM_BINS_THETA2 - 1))
    
    # Clamp indices to be within bounds
    return np.clip(t1_idx, 0, NUM_BINS_THETA1 - 1), np.clip(t2_idx, 0, NUM_BINS_THETA2 - 1)

def get_state_center(t1_idx, t2_idx):
    """Gets the continuous angle at the center of a discrete grid cell."""
    t1_norm = t1_idx / (NUM_BINS_THETA1 - 1)
    t2_norm = t2_idx / (NUM_BINS_THETA2 - 1)
    
    t1 = t1_norm * (theta1_range[1] - theta1_range[0]) + theta1_range[0]
    t2 = t2_norm * (theta2_range[1] - theta2_range[0]) + theta2_range[0]
    return t1, t2

# ===================================================================
# --- 3. System Model and Cost Function (Unchanged) ---
# ===================================================================

def forward_kinematics(t1, t2):
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return (x, y)

def distance_from_goal(t1, t2):
    x, y = forward_kinematics(t1, t2)
    return np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

# ===================================================================
# --- 4. The RTDP Optimizer (<<< THE NEW BRAIN >>>) ---
# ===================================================================

def rtdp_optimizer(current_state):
    """
    Finds the best action by running simulated trials and updating
    the persistent value function.
    """
    t1_current, t2_current = current_state

    # Define the discrete set of possible actions
    control_step = 0.1  # A fixed step for actions
    possible_d_thetas = [-control_step, 0, control_step]
    actions = []
    for dt1 in possible_d_thetas:
        for dt2 in possible_d_thetas:
            if dt1 == 0 and dt2 == 0: continue # Skip the no-op action
            actions.append(np.array([dt1, dt2]))

    # --- 1. Run N trials to refine the value function near the current state ---
    for _ in range(NUM_TRIALS):
        sim_state = np.array(current_state)
        # Each trial runs for a short horizon
        for _ in range(PLANNING_HORIZON):
            
            # Get discrete index for the current simulation state
            sim_t1_idx, sim_t2_idx = discretize_state(sim_state[0], sim_state[1])
            
            min_future_cost = float('inf')
            best_sim_action = None

            # --- One-step lookahead to find the best action and update V(s) ---
            for action in actions:
                # Predict next state
                next_sim_state = sim_state + action
                
                # Get cost of being in the next state
                cost = distance_from_goal(next_sim_state[0], next_sim_state[1])
                
                # Get estimated future cost from the value function
                next_t1_idx, next_t2_idx = discretize_state(next_sim_state[0], next_sim_state[1])
                future_cost_estimate = value_function[next_t1_idx, next_t2_idx]
                
                total_expected_cost = cost + future_cost_estimate

                if total_expected_cost < min_future_cost:
                    min_future_cost = total_expected_cost
                    best_sim_action = action
            
            # --- 2. The Bellman Backup ---
            # Update the value of the current state with the best value found
            value_function[sim_t1_idx, sim_t2_idx] = min_future_cost
            
            # --- 3. Advance the simulation ---
            sim_state += best_sim_action

    # --- 4. After all trials, choose the best REAL action from the current state ---
    min_final_cost = float('inf')
    best_real_action = np.array([0, 0])
    for action in actions:
        next_state = np.array(current_state) + action
        cost = distance_from_goal(next_state[0], next_state[1])
        next_t1_idx, next_t2_idx = discretize_state(next_state[0], next_state[1])
        total_cost = cost + value_function[next_t1_idx, next_t2_idx]
        if total_cost < min_final_cost:
            min_final_cost = total_cost
            best_real_action = action
            
    return best_real_action

# ===================================================================
# --- 5. VISUALIZATION AND MAIN LOOP (Mostly Unchanged) ---
# ===================================================================

# --- Visualization setup (identical to before) ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', linewidth=1, label='Path')
path_history = []
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect('equal')
ax.grid(True); ax.set_title("2DOF Arm with Real-Time Dynamic Programming"); ax.legend()
ax_theta1 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.1, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))

def update_plot():
    """This function is now ONLY for drawing, not for calculation."""
    x0, y0 = 0, 0; x1 = L1 * np.cos(theta1); y1 = L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    path_history.append((x2, y2)); path_x, path_y = zip(*path_history)
    end_effector_path.set_data(path_x, path_y)
    slider1.set_val(np.degrees(theta1)); slider2.set_val(np.degrees(theta2))
    fig.canvas.draw_idle()

def rtdp_control_step(event):
    """ The main RTDP loop: sense, plan, act. """
    global theta1, theta2
    current_state = (theta1, theta2)
    current_dist = distance_from_goal(current_state[0], current_state[1])
    print(f"Distance to Goal: {current_dist:.2f}")

    if current_dist < goal_threshold:
        print("Goal Reached!"); timer.stop(); return

    # Call the RTDP planner to get the single best action
    best_action = rtdp_optimizer(current_state)
    dtheta1, dtheta2 = best_action
    
    theta1 += dtheta1; theta2 += dtheta2
    update_plot()

# ===================================================================
# --- 6. INITIALIZATION AND START ---
# ===================================================================

def initialize_value_function():
    """Initialize V(s) with a heuristic: distance to goal."""
    print("Initializing Value Function with heuristic... please wait.")
    for i in range(NUM_BINS_THETA1):
        for j in range(NUM_BINS_THETA2):
            t1, t2 = get_state_center(i, j)
            value_function[i, j] = distance_from_goal(t1, t2)
    print("Initialization complete.")

# --- Initialize the value function before starting ---
initialize_value_function()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(rtdp_control_step, None)
ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

update_plot()
plt.show()