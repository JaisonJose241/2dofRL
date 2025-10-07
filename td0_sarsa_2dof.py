import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ===================================================================
# --- 1. System and RL Parameters ---
# ===================================================================

L1 = 75.0
L2 = 75.0
target_x, target_y = -100.0, 20.0
goal = (target_x, target_y)
initial_theta1 = np.radians(10)
initial_theta2 = np.radians(150)
goal_threshold = 10.0 # Increased threshold to make reaching goal more frequent

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 0.5   # Start with more exploration
EPSILON_END = 0.01    # End with less exploration
EPSILON_DECAY = 2000  # How fast epsilon decays
MAX_STEPS_PER_EPISODE = 150

NUM_BINS_THETA1 = 50
NUM_BINS_THETA2 = 50
theta1_range = (-np.pi, np.pi)
theta2_range = (-np.pi, np.pi)

# ===================================================================
# --- 2. Q-Table and Action Definitions ---
# ===================================================================

control_step = 0.1
ACTIONS = [np.array([dt1, dt2]) for dt1 in [-control_step, 0, control_step] for dt2 in [-control_step, 0, control_step] if not (dt1 == 0 and dt2 == 0)]
NUM_ACTIONS = len(ACTIONS)

q_table = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2, NUM_ACTIONS))

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
    # dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    # if dist < goal_threshold:
    #     return 100.0
    # return -dist / 100.0
    # return -abs(x)*100 - abs(y)*30
    # return (abs(x)*100 - abs(y)*1)
    # print(x, y)
    return(abs(x)*10 - abs(y)*100)

# ===================================================================
# --- 3. The Live SARSA Learning Step (<<< THE NEW BRAIN >>>) ---
# ===================================================================

# --- Global variables to manage the online learning process ---
episode_count = 0
step_count = 0
total_steps = 0
# We need to store S and A to use in the next step's update
last_state_discrete = None
last_action_idx = None
is_first_step_of_episode = True

# --- Title for the plot that will be updated ---
plot_title = None

def choose_action(state_discrete, epsilon):
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        return np.argmax(q_table[state_discrete])

def online_sarsa_step():
    """Performs one step of interaction and learning, then updates the plot."""
    global theta1, theta2, episode_count, step_count, total_steps
    global last_state_discrete, last_action_idx, is_first_step_of_episode, plot_title

    # --- CHOOSE ACTION A' FOR THE CURRENT STATE S' ---
    state = np.array([theta1, theta2])
    state_discrete = discretize_state(state[0], state[1])
    
    # Decay epsilon over time to shift from exploration to exploitation
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * total_steps / EPSILON_DECAY)
    
    action_idx = choose_action(state_discrete, epsilon)
    action = ACTIONS[action_idx]

    # --- SARSA UPDATE (uses info from the *previous* step) ---
    # We can only perform an update if this is NOT the first step of an episode
    if not is_first_step_of_episode:
        # We are now in state S', and we've chosen action A'.
        # We have S (last_state_discrete) and A (last_action_idx) from the previous step.
        # We can calculate the reward R for arriving at S'.
        reward = get_reward(state[0], state[1])

        # Get Q-values for the update rule
        last_q = q_table[last_state_discrete][last_action_idx]
        current_q = q_table[state_discrete][action_idx]
        
        # The SARSA update
        td_target = reward + DISCOUNT_FACTOR * current_q
        new_q = last_q + LEARNING_RATE * (td_target - last_q)
        q_table[last_state_discrete][last_action_idx] = new_q

    # --- ACT: Take the chosen action ---
    theta1 += action[0]
    theta2 += action[1]
    
    # --- STORE CURRENT STATE AND ACTION FOR THE NEXT UPDATE ---
    last_state_discrete = state_discrete
    last_action_idx = action_idx
    is_first_step_of_episode = False
    
    # --- MANAGE EPISODE ---
    step_count += 1
    total_steps += 1
    
    current_dist = np.sqrt((forward_kinematics(theta1, theta2)[0]-goal[0])**2 + (forward_kinematics(theta1, theta2)[1]-goal[1])**2)
    
    if current_dist < goal_threshold or step_count >= MAX_STEPS_PER_EPISODE:
        # Episode is over, reset for the next one
        episode_count += 1
        step_count = 0
        theta1, theta2 = initial_theta1, initial_theta2
        is_first_step_of_episode = True
        path_history.clear() # Clear the drawn path for the new episode
    
    # --- UPDATE VISUALS ---
    plot_title.set_text(f"Episode: {episode_count}, Step: {step_count}, Epsilon: {epsilon:.2f}")
    update_plot()

# ===================================================================
# --- 4. VISUALIZATION AND EXECUTION ---
# ===================================================================

fig, ax = plt.subplots(); plt.subplots_adjust(bottom=0.2)
arm_line, = plt.plot([], [], 'b-o', lw=3, ms=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', ms=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', lw=1, label='Path')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect('equal'); ax.grid(True)
plot_title = ax.set_title("Press Start to Begin Online Learning")
ax.legend()

path_history = []
theta1, theta2 = initial_theta1, initial_theta2

def update_plot():
    x0,y0=0,0; x1=L1*np.cos(theta1); y1=L1*np.sin(theta1)
    x2,y2=forward_kinematics(theta1,theta2)
    arm_line.set_data([x0,x1,x2],[y0,y1,y2])
    path_history.append((x2,y2));
    if path_history:
        path_x,path_y=zip(*path_history)
        end_effector_path.set_data(path_x,path_y)
    fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=10) # A shorter interval for faster learning
timer.add_callback(online_sarsa_step)
ax_start = plt.axes([0.4, 0.025, 0.2, 0.04])
btn_start = Button(ax_start, 'Start Learning')
btn_start.on_clicked(lambda event: timer.start())

if __name__ == '__main__':
    update_plot()
    plt.show()