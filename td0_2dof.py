import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ===================================================================
# --- 1. System and RL Parameters (Unchanged) ---
# ===================================================================

L1 = 75.0
L2 = 75.0
target_x, target_y = -100.0, 20.0
goal = (target_x, target_y)
initial_theta1 = np.radians(100)
initial_theta2 = np.radians(30)
goal_threshold = 5.0

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.1
NUM_EPISODES = 900
MAX_STEPS_PER_EPISODE = 250

NUM_BINS_THETA1 = 50
NUM_BINS_THETA2 = 50
theta1_range = (-np.pi, np.pi)
theta2_range = (-np.pi, np.pi)

# ===================================================================
# --- 2. Global Value and Policy Tables (<<< MODIFIED >>>) ---
# ===================================================================

# Stores the estimated long-term reward for each state.
value_function = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2))

# (<<< NEW >>>) Stores the best action [d_theta1, d_theta2] for each state.
# The third dimension of size 2 is for the two action components.
policy_table = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2, 2)) 

# --- Helper Functions (Unchanged) ---
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
    x, y = forward_kinematics(t1, t2)
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    if dist < goal_threshold:
        return 100.0
    return -dist / 100.0

# ===================================================================
# --- 3. The TD(0) Learning Brain (<<< MODIFIED >>>) ---
# ===================================================================

control_step = 0.1
possible_d_thetas = [-control_step, 0, control_step]
actions = [np.array([dt1, dt2]) for dt1 in possible_d_thetas for dt2 in possible_d_thetas if not (dt1 == 0 and dt2 == 0)]

def choose_action(state_discrete, epsilon):
    """Chooses an action using an epsilon-greedy policy from the policy table."""
    if random.random() < epsilon:
        # --- Exploration ---
        return random.choice(actions)
    else:
        # --- Exploitation (now a fast table lookup) ---
        return policy_table[state_discrete]

def initialize_tables():
    """Initializes the policy table with random actions."""
    for i in range(NUM_BINS_THETA1):
        for j in range(NUM_BINS_THETA2):
            policy_table[i, j] = random.choice(actions)

def run_learning_episodes():
    """The main offline training loop with integrated policy improvement."""
    print(f"--- Starting TD(0) Training for {NUM_EPISODES} episodes ---")
    for episode in range(NUM_EPISODES):
        state = np.array([initial_theta1, initial_theta2])
        for step in range(MAX_STEPS_PER_EPISODE):
            current_state_discrete = discretize_state(state[0], state[1])
            
            # 1. Choose action from policy table
            action = choose_action(current_state_discrete, EPSILON)
            
            # 2. Take action, observe outcome
            next_state = state + action
            reward = get_reward(next_state[0], next_state[1])
            next_state_discrete = discretize_state(next_state[0], next_state[1])
            
            # --- 3. Policy Evaluation (TD Update on V) ---
            td_target = reward + DISCOUNT_FACTOR * value_function[next_state_discrete]
            td_error = td_target - value_function[current_state_discrete]
            value_function[current_state_discrete] += LEARNING_RATE * td_error
            
            # --- 4. Policy Improvement (Update π) (<<< NEW >>>) ---
            # After updating V(s), find the new best action from s and update the policy.
            best_action_for_current_state = None
            max_value = -float('inf')
            for improv_action in actions:
                check_next_state = state + improv_action
                check_next_discrete = discretize_state(check_next_state[0], check_next_state[1])
                # We check the value of the successor state in our updated value function
                if value_function[check_next_discrete] > max_value:
                    max_value = value_function[check_next_discrete]
                    best_action_for_current_state = improv_action
            policy_table[current_state_discrete] = best_action_for_current_state
            
            # 5. Move to next state
            state = next_state
            if reward > 50:
                break
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES} completed.")
    print("--- Training Finished ---")

# ===================================================================
# --- 4. VISUALIZATION AND EXECUTION (Now much faster) ---
# ===================================================================
theta1, theta2 = initial_theta1, initial_theta2
path_history = []
fig, ax = plt.subplots(); plt.subplots_adjust(bottom=0.2)
arm_line, = plt.plot([], [], 'b-o', lw=3, ms=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', ms=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', lw=1, label='Path')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect('equal')
ax.grid(True); ax.set_title("2DOF Arm with Learned TD(0) Policy Table"); ax.legend()

def update_plot():
    global theta1, theta2
    x0,y0=0,0; x1=L1*np.cos(theta1); y1=L1*np.sin(theta1)
    x2,y2=forward_kinematics(theta1,theta2)
    arm_line.set_data([x0,x1,x2],[y0,y1,y2])
    path_history.append((x2,y2)); path_x,path_y=zip(*path_history)
    end_effector_path.set_data(path_x,path_y); fig.canvas.draw_idle()

def execute_learned_policy_step():
    global theta1, theta2
    dist = np.sqrt((forward_kinematics(theta1, theta2)[0]-goal[0])**2 + (forward_kinematics(theta1, theta2)[1]-goal[1])**2)
    if dist < goal_threshold:
        print("Goal Reached!"); timer.stop(); return

    # Action selection is now just a direct, fast lookup from the policy table
    state_discrete = discretize_state(theta1, theta2)
    action = policy_table[state_discrete] # Epsilon = 0
    theta1 += action[0]; theta2 += action[1]
    update_plot()

timer = fig.canvas.new_timer(interval=50)
timer.add_callback(execute_learned_policy_step)
ax_start = plt.axes([0.4, 0.025, 0.2, 0.04])
btn_start = Button(ax_start, 'Run Learned Policy')
btn_start.on_clicked(lambda event: timer.start())

if __name__ == '__main__':
    initialize_tables()
    run_learning_episodes()
    print("\nClose the plot window to exit.")
    update_plot()
    plt.show()