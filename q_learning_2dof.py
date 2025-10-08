import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time
import os
import itertools

# ===================================================================
# --- 1. Environment and Arm Parameters ---
# ===================================================================

# --- Arm Link Lengths ---
L1 = 75.0
L2 = 75.0

# --- Target in Task Space ---
target_x, target_y = 100.0, 50.0
goal = (target_x, target_y)

# --- A fixed starting state for the arm ---
INITIAL_THETAS = np.radians(np.array([170.0, -90.0]))

# --- Q-Learning Hyperparameters ---
NUM_EPISODES = 200
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_END = 0.0
MAX_STEPS_PER_EPISODE = 100
Q_TABLE_FILENAME = "q_table_2dof.npy" # <<< File to save/load the Q-table

# --- State and Action Space Discretization ---
NUM_BINS_THETA1 = 20
NUM_BINS_THETA2 = 20
ACTION_STEP = np.radians(5)

# Using the 9-action space as requested
x = ACTION_STEP
possible_moves = [-x, 0, x]
action_combinations = list(itertools.product(possible_moves, repeat=2))
ACTIONS = {i: np.array(action) for i, action in enumerate(action_combinations)}
NUM_ACTIONS = len(ACTIONS)

# ===================================================================
# --- 2. The Q-Learning Agent ---
# ===================================================================

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2, NUM_ACTIONS))
        self.epsilon = EPSILON_START
        self.angle_bins = [
            np.linspace(0, np.pi, NUM_BINS_THETA1),
            np.linspace(-np.pi, np.pi, NUM_BINS_THETA2)
        ]

    def discretize_state(self, thetas):
        t1_bin = np.digitize(thetas[0], self.angle_bins[0]) - 1
        t2_bin = np.digitize(thetas[1], self.angle_bins[1]) - 1
        return (np.clip(t1_bin, 0, NUM_BINS_THETA1-1), np.clip(t2_bin, 0, NUM_BINS_THETA2-1))

    def choose_action(self, state_bins):
        if np.random.random() < self.epsilon:
            return np.random.choice(NUM_ACTIONS)
        else:
            return np.argmax(self.q_table[state_bins])

    def update_q_table(self, state_bins, action, reward, next_state_bins):
        old_q_value = self.q_table[state_bins][action]
        max_future_q = np.max(self.q_table[next_state_bins])
        new_q_value = old_q_value + ALPHA * (reward + GAMMA * max_future_q - old_q_value)
        self.q_table[state_bins][action] = new_q_value

    def decay_epsilon(self):
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

# ===================================================================
# --- 3. Environment Interaction Functions ---
# ===================================================================

def forward_kinematics(thetas):
    t1, t2 = thetas
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return (x, y)

def get_reward(thetas, current_step):
    x, y = forward_kinematics(thetas)
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    if dist < 10.0:
        return 100.0
    return -dist / 100.0

# ===================================================================
# --- 4. VISUALIZATION AND INTEGRATED TRAINING LOOP ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25) # Made room for buttons
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Learning Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
ax.set_aspect('equal'); ax.grid(True)
title = ax.set_title("2DOF Arm with Q-Learning")
ax.legend()

agent = QLearningAgent()
thetas_vis = np.copy(INITIAL_THETAS)
current_episode = 1; current_step = 0; timer = None

def update_plot(thetas_to_draw):
    x0,y0=0,0; x1=L1*np.cos(thetas_to_draw[0]); y1=L1*np.sin(thetas_to_draw[0]); x2,y2=forward_kinematics(thetas_to_draw)
    arm_line.set_data([x0,x1,x2],[y0,y1,y2]); fig.canvas.draw_idle()

def training_step_and_visualize():
    global current_episode, current_step, thetas_vis
    if current_episode > NUM_EPISODES:
        title.set_text(f"Training Finished! Click 'Save' to store Q-Table.")
        timer.stop(); return

    state_bins = agent.discretize_state(thetas_vis)
    action = agent.choose_action(state_bins)
    thetas_vis += ACTIONS[action]
    thetas_vis[0] = np.clip(thetas_vis[0], 0, np.pi); thetas_vis[1] = np.clip(thetas_vis[1], -np.pi, np.pi)
    reward = get_reward(thetas_vis, current_step)
    next_state_bins = agent.discretize_state(thetas_vis)
    agent.update_q_table(state_bins, action, reward, next_state_bins)
    update_plot(thetas_vis)
    title.set_text(f"Episode: {current_episode}/{NUM_EPISODES} | Epsilon: {agent.epsilon:.3f}")

    current_step += 1
    goal_reached = reward > 50
    if goal_reached or current_step >= MAX_STEPS_PER_EPISODE:
        if goal_reached: print(f"Goal reached in episode {current_episode}!")
        current_episode += 1; current_step = 0
        agent.decay_epsilon(); thetas_vis = np.copy(INITIAL_THETAS)

def start_training_visualization(event):
    global agent, thetas_vis, current_episode, current_step, timer
    agent = QLearningAgent(); thetas_vis = np.copy(INITIAL_THETAS)
    current_episode = 1; current_step = 0
    print("--- Starting New Q-Learning Training ---")
    if timer: timer.stop()
    timer = fig.canvas.new_timer(interval=10); timer.add_callback(training_step_and_visualize); timer.start()

def run_trained_agent(event):
    global thetas_vis, timer
    if timer: timer.stop()
    thetas_vis = np.copy(INITIAL_THETAS)
    agent.epsilon = 0 # Turn off exploration
    title.set_text("Running Trained Agent (Epsilon = 0)")
    def update_animation():
        global thetas_vis
        state_bins = agent.discretize_state(thetas_vis)
        action = agent.choose_action(state_bins)
        thetas_vis += ACTIONS[action]
        thetas_vis[0] = np.clip(thetas_vis[0], 0, np.pi); thetas_vis[1] = np.clip(thetas_vis[1], -np.pi, np.pi)
        update_plot(thetas_vis)
        if get_reward(thetas_vis, current_step) > 50:
            print("Goal reached by trained agent!"); timer.stop()
    timer = fig.canvas.new_timer(interval=50); timer.add_callback(update_animation); timer.start()

# --- <<< NEW: Save and Load Functionality >>> ---
def save_q_table(event):
    if timer: timer.stop()
    try:
        np.save(Q_TABLE_FILENAME, agent.q_table)
        title.set_text(f"Q-Table saved to {Q_TABLE_FILENAME}")
        print(f"Q-Table saved successfully to {Q_TABLE_FILENAME}")
    except Exception as e:
        print(f"Error saving Q-Table: {e}")

def load_q_table(event):
    if timer: timer.stop()
    try:
        if os.path.exists(Q_TABLE_FILENAME):
            loaded_table = np.load(Q_TABLE_FILENAME)
            if loaded_table.shape == agent.q_table.shape:
                agent.q_table = loaded_table
                agent.epsilon = 0 # Ready for performance
                title.set_text("Loaded Q-Table | Ready to Run")
                print(f"Q-Table loaded successfully from {Q_TABLE_FILENAME}")
            else:
                print("Error: Saved table shape does not match current agent configuration.")
        else:
            title.set_text("No saved Q-Table file found!")
            print(f"Error: File '{Q_TABLE_FILENAME}' not found.")
    except Exception as e:
        print(f"Error loading Q-Table: {e}")

# --- Button Layout ---
ax_train = plt.axes([0.15, 0.1, 0.2, 0.075])
btn_train = Button(ax_train, 'Start/Reset Training')
btn_train.on_clicked(start_training_visualization)

ax_run = plt.axes([0.40, 0.1, 0.15, 0.075])
btn_run = Button(ax_run, 'Run Trained')
btn_run.on_clicked(run_trained_agent)

ax_save = plt.axes([0.60, 0.1, 0.15, 0.075])
btn_save = Button(ax_save, 'Save Q-Table')
btn_save.on_clicked(save_q_table)

ax_load = plt.axes([0.80, 0.1, 0.15, 0.075])
btn_load = Button(ax_load, 'Load Q-Table')
btn_load.on_clicked(load_q_table)

plt.show()


