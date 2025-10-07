import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ===================================================================
# --- 1. System and RL Parameters ---
# ===================================================================

# --- Arm Link Lengths & Target ---
L1 = 75.0
L2 = 75.0
target_x, target_y = -100.0, 20.0
goal = (target_x, target_y)
initial_theta1 = np.radians(100)
initial_theta2 = np.radians(30)
goal_threshold = 5.0

# --- RL Parameters (<<< NEW >>>) ---
LEARNING_RATE = 0.1  # Alpha: How quickly the agent learns.
DISCOUNT_FACTOR = 0.99 # Gamma: Preference for immediate rewards over future ones.
EPSILON = 0.1        # Epsilon: Probability of choosing a random action (exploration).
NUM_EPISODES = 1000  # Number of training attempts.
MAX_STEPS_PER_EPISODE = 200

# --- Discretization Parameters (Same as RTDP) ---
NUM_BINS_THETA1 = 50
NUM_BINS_THETA2 = 50
theta1_range = (-np.pi, np.pi)
theta2_range = (-np.pi, np.pi)

# ===================================================================
# --- 2. Global Value Function and Helper Functions ---
# ===================================================================

# The value function table, initialized to zeros. It will be learned.
value_function = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2))

def discretize_state(t1, t2):
    t1_norm = (t1 - theta1_range[0]) / (theta1_range[1] - theta1_range[0])
    t2_norm = (t2 - theta2_range[0]) / (theta2_range[1] - theta2_range[0])
    t1_idx = int(t1_norm * (NUM_BINS_THETA1 - 1))
    t2_idx = int(t2_norm * (NUM_BINS_THETA2 - 1))
    return np.clip(t1_idx, 0, NUM_BINS_THETA1 - 1), np.clip(t2_idx, 0, NUM_BINS_THETA2 - 1)

def forward_kinematics(t1, t2):
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return x, y

def get_reward(t1, t2):
    """Calculates reward. Negative distance is a common choice."""
    x, y = forward_kinematics(t1, t2)
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    
    if dist < goal_threshold:
        return 100.0  # Large positive reward for reaching the goal
    return -dist / 100.0 # Negative reward proportional to distance

# ===================================================================
# --- 3. The TD(0) Learning Brain (<<< THE NEW PART >>>) ---
# ===================================================================

# Define the discrete set of possible actions
control_step = 0.1
possible_d_thetas = [-control_step, 0, control_step]
actions = []
for dt1 in possible_d_thetas:
    for dt2 in possible_d_thetas:
        if dt1 == 0 and dt2 == 0: continue
        actions.append(np.array([dt1, dt2]))

def choose_action(state_continuous, epsilon):
    """Chooses an action using an epsilon-greedy policy."""
    if random.random() < epsilon:
        # --- Exploration ---
        return random.choice(actions)
    else:
        # --- Exploitation ---
        best_action = None
        max_value = -float('inf')
        for action in actions:
            next_state = state_continuous + action
            next_state_discrete = discretize_state(next_state[0], next_state[1])
            if value_function[next_state_discrete] > max_value:
                max_value = value_function[next_state_discrete]
                best_action = action
        return best_action

def run_learning_episodes():
    """The main offline training loop."""
    print(f"--- Starting TD(0) Training for {NUM_EPISODES} episodes ---")
    for episode in range(NUM_EPISODES):
        # Reset state for each episode
        state = np.array([initial_theta1, initial_theta2])
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # 1. Choose an action
            action = choose_action(state, EPSILON)
            
            # 2. Take the action and observe the outcome
            next_state = state + action
            reward = get_reward(next_state[0], next_state[1])
            
            # 3. Perform the TD(0) update
            current_state_discrete = discretize_state(state[0], state[1])
            next_state_discrete = discretize_state(next_state[0], next_state[1])
            
            # The TD Update Rule
            td_target = reward + DISCOUNT_FACTOR * value_function[next_state_discrete]
            td_error = td_target - value_function[current_state_discrete]
            value_function[current_state_discrete] += LEARNING_RATE * td_error
            
            # 4. Move to the next state
            state = next_state
            
            # End episode if goal is reached
            if reward > 50: # Reached the goal
                break
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES} completed.")
    print("--- Training Finished ---")

# ===================================================================
# --- 4. VISUALIZATION OF THE LEARNED POLICY ---
# ===================================================================

# --- Global state for visualization ---
theta1 = initial_theta1
theta2 = initial_theta2
path_history = []

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', linewidth=1, label='Path')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect('equal')
ax.grid(True); ax.set_title("2DOF Arm with Learned TD(0) Policy"); ax.legend()

def update_plot():
    global theta1, theta2
    x0, y0 = 0, 0; x1 = L1 * np.cos(theta1); y1 = L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    path_history.append((x2, y2)); path_x, path_y = zip(*path_history)
    end_effector_path.set_data(path_x, path_y)
    fig.canvas.draw_idle()

def execute_learned_policy_step(event):
    """Executes one step of the final, greedy policy."""
    global theta1, theta2
    state = np.array([theta1, theta2])
    dist = np.sqrt((forward_kinematics(theta1, theta2)[0] - goal[0])**2 + (forward_kinematics(theta1, theta2)[1] - goal[1])**2)
    
    if dist < goal_threshold:
        print("Goal Reached!"); timer.stop(); return

    # Choose the BEST action (Epsilon = 0)
    action = choose_action(state, epsilon=0)
    theta1 += action[0]; theta2 += action[1]
    update_plot()

# --- Setup animation ---
timer = fig.canvas.new_timer(interval=50)
timer.add_callback(execute_learned_policy_step, None)

ax_start = plt.axes([0.4, 0.025, 0.2, 0.04])
btn_start = Button(ax_start, 'Run Learned Policy')
btn_start.on_clicked(lambda event: timer.start())

# ===================================================================
# --- 5. RUN THE PROGRAM ---
# ===================================================================
if __name__ == '__main__':
    # --- PHASE 1: OFFLINE TRAINING ---
    run_learning_episodes()
    
    # --- PHASE 2: VISUALIZATION ---
    print("\nClose the plot window to exit.")
    update_plot()
    plt.show()