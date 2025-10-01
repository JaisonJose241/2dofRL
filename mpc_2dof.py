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

# --- MPC Controller Parameters (<<< NEW >>>) ---
# H: How many steps to look ahead into the future.
PLANNING_HORIZON = 15
# N: Number of random action sequences to evaluate at each time step.
NUM_SEQUENCES = 100
# The higher H and N are, the smarter the planner, but the slower the computation.

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
# --- 3. The MPC Optimizer (<<< THE NEW BRAIN >>>) ---
# ===================================================================

def mpc_optimizer(current_state):
    """
    Finds the best action by planning over a short horizon.
    This is the core of the MPC.
    """
    t1_current, t2_current = current_state
    best_sequence = None
    min_cost = float('inf')

    # Define the magnitude of actions to try, adapted to distance
    dist_val = distance_from_goal(t1_current, t2_current) / 100000.0
    control_step = np.clip(dist_val, 0.005, 0.05)
    possible_moves = [-control_step, 0, control_step]

    # 1. Generate and evaluate N random sequences
    for _ in range(NUM_SEQUENCES):
        # Generate one random sequence of H actions
        random_sequence = []
        for _ in range(PLANNING_HORIZON):
            # Each action is a random move for each joint
            d_t1 = np.random.choice(possible_moves)
            d_t2 = np.random.choice(possible_moves)
            random_sequence.append(np.array([d_t1, d_t2]))

        # 2. Simulate this sequence to calculate its total cost
        sim_state = np.array(current_state)
        total_cost = 0
        for action in random_sequence:
            # Apply action to get next simulated state
            sim_state += action
            # The cost is the distance to the goal at that future step
            total_cost += distance_from_goal(sim_state[0], sim_state[1])

        # 3. Keep track of the best sequence found so far
        if total_cost < min_cost:
            min_cost = total_cost
            best_sequence = random_sequence

    # 4. Return only the FIRST action of the best sequence
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
ax.set_title("2DOF Arm with Model Predictive Control")
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
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(mpc_control_step, None)

ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

# --- Initial draw ---
update_plot()
plt.show()
