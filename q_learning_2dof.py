import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

# ===================================================================
# --- 1. Environment and Arm Parameters ---
# ===================================================================

# --- Arm Link Lengths ---
L1 = 75.0
L2 = 75.0

# --- Target in Task Space ---
target_x, target_y = 100.0, 50.0
goal = (target_x, target_y)

# <<< NEW: Define a fixed starting state for the arm >>>
INITIAL_THETAS = np.radians(np.array([170.0, -90.0])) # A fixed starting position

# --- Q-Learning Hyperparameters ---
NUM_EPISODES = 200
ALPHA = 0.1          # Learning Rate
GAMMA = 0.99         # Discount Factor
EPSILON_START = 1.0  # Starting exploration rate
EPSILON_DECAY = 0.99 # How fast epsilon shrinks
EPSILON_END = 0.0   # Minimum exploration rate
MAX_STEPS_PER_EPISODE = 200

# --- State and Action Space Discretization ---
NUM_BINS_THETA1 = 20
NUM_BINS_THETA2 = 20
ACTION_STEP = np.radians(5) # 5 degrees
ACTIONS = {
    0: np.array([ACTION_STEP, 0]),         # Increase theta1
    1: np.array([-ACTION_STEP, 0]),        # Decrease theta1
    2: np.array([0, ACTION_STEP]),         # Increase theta2
    3: np.array([0, -ACTION_STEP]),        # Decrease theta2
}
NUM_ACTIONS = len(ACTIONS)

# ===================================================================
# --- 2. The Q-Learning Agent ---
# ===================================================================

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2, NUM_ACTIONS))
        self.epsilon = EPSILON_START
        self.angle_bins = [
            # <<< MODIFIED: Bins for theta1 are now from 0 to 180 degrees >>>
            np.linspace(0, np.pi, NUM_BINS_THETA1),
            np.linspace(-np.pi, np.pi, NUM_BINS_THETA2)
        ]

    def discretize_state(self, thetas):
        """Converts continuous angles into discrete bin indices."""
        t1_bin = np.digitize(thetas[0], self.angle_bins[0]) - 1
        t2_bin = np.digitize(thetas[1], self.angle_bins[1]) - 1
        return (np.clip(t1_bin, 0, NUM_BINS_THETA1-1), np.clip(t2_bin, 0, NUM_BINS_THETA2-1))

    def choose_action(self, state_bins):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(NUM_ACTIONS)
        else:
            return np.argmax(self.q_table[state_bins])

    def update_q_table(self, state_bins, action, reward, next_state_bins):
        """Updates the Q-table using the Bellman equation."""
        old_q_value = self.q_table[state_bins][action]
        max_future_q = np.max(self.q_table[next_state_bins])
        new_q_value = old_q_value + ALPHA * (reward + GAMMA * max_future_q - old_q_value)
        self.q_table[state_bins][action] = new_q_value

    def decay_epsilon(self):
        """Shrinks the exploration rate over time."""
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
    
    return -dist / 100.0 - (current_step*10)

# ===================================================================
# --- 4. VISUALIZATION AND INTEGRATED TRAINING LOOP ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.2)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Learning Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
ax.set_xlim(-200, 200)
ax.set_ylim(-200, 200)
ax.set_aspect('equal')
ax.grid(True)
title = ax.set_title("2DOF Arm with Q-Learning")
ax.legend()

# --- Global variables to manage the training state ---
agent = QLearningAgent()
# <<< MODIFIED: Initialize to the fixed starting state >>>
thetas_vis = np.copy(INITIAL_THETAS)
current_episode = 1
current_step = 0
timer = None

def update_plot(thetas_to_draw):
    """Helper function to redraw the arm."""
    x0, y0 = 0, 0
    x1 = L1 * np.cos(thetas_to_draw[0])
    y1 = L1 * np.sin(thetas_to_draw[0])
    x2, y2 = forward_kinematics(thetas_to_draw)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    fig.canvas.draw_idle()

def training_step_and_visualize(frame):
    """This function is the core training loop, called at each animation frame."""
    global current_episode, current_step, thetas_vis

    if current_episode > NUM_EPISODES:
        print("--- Training Finished ---")
        timer.stop()
        title.set_text(f"Training Finished! | Final Policy")
        return

    # 1. Get current discrete state
    state_bins = agent.discretize_state(thetas_vis)
    
    # 2. Choose action (with exploration)
    action = agent.choose_action(state_bins)
    
    # 3. Apply action and get outcome
    thetas_vis += ACTIONS[action]
    # <<< MODIFIED: Clip theta1 and theta2 to their respective limits >>>
    thetas_vis[0] = np.clip(thetas_vis[0], 0, np.pi)      # Theta1 limit: 0 to 180
    thetas_vis[1] = np.clip(thetas_vis[1], -np.pi, np.pi) # Theta2 limit: -180 to 180
    
    reward = get_reward(thetas_vis, current_step)
    next_state_bins = agent.discretize_state(thetas_vis)
    
    # 4. Update the agent's brain (Q-table)
    agent.update_q_table(state_bins, action, reward, next_state_bins)
    
    # 5. Update visualization
    update_plot(thetas_vis)
    title.set_text(f"Episode: {current_episode}/{NUM_EPISODES} | Epsilon: {agent.epsilon:.3f}")

    # 6. Handle end of episode (goal reached or timeout)
    current_step += 1
    goal_reached = reward > 50
    if goal_reached or current_step >= MAX_STEPS_PER_EPISODE:
        if goal_reached:
            print(f"Goal reached in episode {current_episode}!")
        
        # Start next episode
        current_episode += 1
        current_step = 0
        agent.decay_epsilon()
        # Reset arm to the same starting position for the next episode
        # <<< MODIFIED: Reset to the fixed starting state instead of random >>>
        thetas_vis = np.copy(INITIAL_THETAS)

def start_training_visualization(event):
    """Callback for the 'Start Training' button."""
    global agent, thetas_vis, current_episode, current_step, timer
    # Reset everything to start a new training run
    agent = QLearningAgent()
    # <<< MODIFIED: Reset to the fixed starting state instead of random >>>
    thetas_vis = np.copy(INITIAL_THETAS)
    current_episode = 1
    current_step = 0
    print("--- Starting New Q-Learning Training ---")
    
    # Stop any old timer and start a new one
    if timer:
        timer.stop()
    # A shorter interval makes the visualization feel faster
    timer = fig.canvas.new_timer(interval=10) 
    timer.add_callback(training_step_and_visualize, None)
    timer.start()

ax_run = plt.axes([0.65, 0.05, 0.25, 0.075])
btn_run = Button(ax_run, 'Start/Reset Training')
btn_run.on_clicked(start_training_visualization)

plt.show()


