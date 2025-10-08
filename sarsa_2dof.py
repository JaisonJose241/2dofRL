import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ===================================================================
# --- 1. Environment and Arm Parameters ---
# ===================================================================

# --- Arm Link Lengths ---
L1 = 75.0
L2 = 75.0

# --- Target in Task Space ---
target_x, target_y = 0.0, 150.0
goal = (target_x, target_y)

# --- A fixed starting state for the arm ---
INITIAL_THETAS = np.radians(np.array([170.0, -90.0]))

# --- SARSA Hyperparameters ---
NUM_EPISODES = 100
ALPHA = 0.2          # Learning Rate
GAMMA = 0.99         # Discount Factor
EPSILON_START = 1.0
EPSILON_DECAY = 0.99 # Faster decay for quicker visual learning
EPSILON_END = 0.01
MAX_STEPS_PER_EPISODE = 200

# --- State and Action Space Discretization ---
NUM_BINS_THETA1 = 20 # Bins for theta1 (0 to 180 degrees)
NUM_BINS_THETA2 = 20 # Bins for theta2 (-180 to 180 degrees)
ACTION_STEP = np.radians(5)
ACTIONS = {
    0: np.array([ACTION_STEP, 0]),
    1: np.array([-ACTION_STEP, 0]),
    2: np.array([0, ACTION_STEP]),
    3: np.array([0, -ACTION_STEP]),
}
NUM_ACTIONS = len(ACTIONS)

# ===================================================================
# --- 2. The SARSA Agent ---
# ===================================================================

class SarsaAgent:
    def __init__(self):
        self.q_table = np.zeros((NUM_BINS_THETA1, NUM_BINS_THETA2, NUM_ACTIONS))
        self.epsilon = EPSILON_START
        self.angle_bins = [
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

    def update_q_table(self, s, a, r, s_prime, a_prime):
        """
        Updates the Q-table using the SARSA Bellman equation.
        Note that it takes the *next action* 'a_prime' as an input.
        """
        old_q_value = self.q_table[s][a]
        
        # The core of SARSA: use the Q-value of the *actual* next state-action pair
        future_q_value = self.q_table[s_prime][a_prime]
        
        # The SARSA update formula
        new_q_value = old_q_value + ALPHA * (r + GAMMA * future_q_value - old_q_value)
        self.q_table[s][a] = new_q_value

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

def get_reward(thetas):
    x, y = forward_kinematics(thetas)
    dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
    if dist < 10.0:
        return 100.0
    return -dist / 100.0

# ===================================================================
# --- 4. VISUALIZATION AND INTEGRATED TRAINING LOOP ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.2)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Learning Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
ax.set_aspect('equal'); ax.grid(True)
title = ax.set_title("2DOF Arm with SARSA Learning")
ax.legend()

# --- Global variables to manage the training state ---
agent = SarsaAgent()
thetas_vis = np.copy(INITIAL_THETAS)
current_episode = 1
current_step = 0
# For SARSA, we need to know the *next* action before we can update
next_action = None
timer = None

def update_plot(thetas_to_draw):
    x0, y0 = 0, 0
    x1 = L1 * np.cos(thetas_to_draw[0])
    y1 = L1 * np.sin(thetas_to_draw[0])
    x2, y2 = forward_kinematics(thetas_to_draw)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    fig.canvas.draw_idle()

def training_step_and_visualize(frame):
    global current_episode, current_step, thetas_vis, next_action

    if current_episode > NUM_EPISODES:
        print("--- Training Finished ---")
        timer.stop()
        title.set_text(f"Training Finished! | Final Policy")
        return

    # S, A are from the PREVIOUS step; R, S' are from the CURRENT step
    state_bins = agent.discretize_state(thetas_vis)
    action = next_action # Use the action we decided on in the last frame
    
    # Take the action
    thetas_vis += ACTIONS[action]
    thetas_vis[0] = np.clip(thetas_vis[0], 0, np.pi)
    thetas_vis[1] = np.clip(thetas_vis[1], -np.pi, np.pi)
    
    # Observe Reward (R) and Next State (S')
    reward = get_reward(thetas_vis)
    next_state_bins = agent.discretize_state(thetas_vis)
    
    # Choose the Next Action (A') from the Next State (S')
    next_action = agent.choose_action(next_state_bins)
    
    # Update the agent's brain using the full (S, A, R, S', A') quintuple
    agent.update_q_table(state_bins, action, reward, next_state_bins, next_action)
    
    # Update visualization and title
    update_plot(thetas_vis)
    title.set_text(f"Episode: {current_episode}/{NUM_EPISODES} | Epsilon: {agent.epsilon:.3f}")

    # Handle end of episode
    current_step += 1
    goal_reached = reward > 50
    if goal_reached or current_step >= MAX_STEPS_PER_EPISODE:
        if goal_reached: print(f"Goal reached in episode {current_episode}!")
        current_episode += 1; current_step = 0
        agent.decay_epsilon()
        thetas_vis = np.copy(INITIAL_THETAS)
        # We need to choose the very first action for the new episode
        next_action = agent.choose_action(agent.discretize_state(thetas_vis))

def start_training_visualization(event):
    global agent, thetas_vis, current_episode, current_step, timer, next_action
    agent = SarsaAgent()
    thetas_vis = np.copy(INITIAL_THETAS)
    current_episode = 1; current_step = 0
    print("--- Starting New SARSA Training ---")
    
    # SARSA needs to choose the very first action before the loop starts
    initial_state_bins = agent.discretize_state(thetas_vis)
    next_action = agent.choose_action(initial_state_bins)
    
    if timer: timer.stop()
    timer = fig.canvas.new_timer(interval=10) 
    timer.add_callback(training_step_and_visualize, None)
    timer.start()

ax_run = plt.axes([0.65, 0.05, 0.25, 0.075])
btn_run = Button(ax_run, 'Start/Reset Training')
btn_run.on_clicked(start_training_visualization)

plt.show()
