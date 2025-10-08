import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import collections
import time

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

# --- DQN Hyperparameters ---
NUM_EPISODES = 100
GAMMA = 0.99         # Discount Factor
EPSILON_START = 1.0
EPSILON_DECAY = 0.9
EPSILON_END = 0.01
TARGET_UPDATE_FREQUENCY = 10 # How often to update the target network
BATCH_SIZE = 128
REPLAY_MEMORY_SIZE = 10000
LR = 1e-3            # Learning Rate for the network
MAX_STEPS_PER_EPISODE = 100

# --- Action Space (still discrete for DQN) ---
ACTION_STEP = np.radians(5)
ACTIONS = {
    0: np.array([ACTION_STEP, 0]),
    1: np.array([-ACTION_STEP, 0]),
    2: np.array([0, ACTION_STEP]),
    3: np.array([0, -ACTION_STEP]),
}
ACTION_SPACE_SIZE = len(ACTIONS)
STATE_SPACE_SIZE = 2 # [theta1, theta2]

# ===================================================================
# --- 2. The DQN Components ---
# ===================================================================

class QNetwork(nn.Module):
    """The neural network that approximates the Q-function."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory:
    """A buffer to store and sample past experiences."""
    def __init__(self, capacity):
        self.memory = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """The main agent that interacts with the environment."""
    def __init__(self):
        self.main_network = QNetwork(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
        self.target_network = QNetwork(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval() # Target network is only for evaluation
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(ACTION_SPACE_SIZE)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.main_network(state_tensor)
                return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        experiences = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        predicted_q_values = self.main_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = F.mse_loss(predicted_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
    return -dist / 10.0

# ===================================================================
# --- 4. VISUALIZATION AND INTEGRATED TRAINING LOOP ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.2)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Learning Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200)
ax.set_aspect('equal'); ax.grid(True)
title = ax.set_title("2DOF Arm with DQN Learning")
ax.legend()

# --- Global variables to manage the training state ---
agent = DQNAgent()
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

    # 1. Choose action (with exploration)
    action = agent.choose_action(thetas_vis)
    
    # 2. Take action and get outcome
    thetas_next = thetas_vis + ACTIONS[action]
    thetas_next[0] = np.clip(thetas_next[0], 0, np.pi)
    thetas_next[1] = np.clip(thetas_next[1], -np.pi, np.pi)
    
    reward = get_reward(thetas_next)
    done = reward > 50
    
    # 3. Store experience in memory and learn from a batch
    agent.memory.push(thetas_vis, action, reward, thetas_next, done)
    agent.learn()
    
    # Update state for the next step
    thetas_vis = thetas_next
    
    # 4. Update visualization
    update_plot(thetas_vis)
    title.set_text(f"Episode: {current_episode}/{NUM_EPISODES} | Epsilon: {agent.epsilon:.3f}")

    # 5. Handle end of episode
    current_step += 1
    if done or current_step >= MAX_STEPS_PER_EPISODE:
        if done:
            print(f"Goal reached in episode {current_episode}!")
        
        agent.decay_epsilon()
        if current_episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.target_network.load_state_dict(agent.main_network.state_dict())
            
        # Start next episode
        current_episode += 1
        current_step = 0
        thetas_vis = np.copy(INITIAL_THETAS)

def start_training_visualization(event):
    """Callback for the 'Start Training' button."""
    global agent, thetas_vis, current_episode, current_step, timer
    # Reset everything to start a new training run
    agent = DQNAgent()
    thetas_vis = np.copy(INITIAL_THETAS)
    current_episode = 1
    current_step = 0
    print("--- Starting New DQN Training ---")
    
    if timer:
        timer.stop()
    # A shorter interval makes the visualization feel faster
    timer = fig.canvas.new_timer(interval=1) 
    timer.add_callback(training_step_and_visualize, None)
    timer.start()

ax_run = plt.axes([0.65, 0.05, 0.25, 0.075])
btn_run = Button(ax_run, 'Start/Reset Training')
btn_run.on_clicked(start_training_visualization)

plt.show()

