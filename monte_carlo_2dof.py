import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import random

# ===================================================================
# --- 1. System Parameters (Unchanged) ---
# ===================================================================

L1 = 75.0
L2 = 75.0
target_x, target_y = -100.0, 20.0
goal = (target_x, target_y)
theta1 = np.radians(100)
theta2 = np.radians(30)
goal_threshold = 2.0

# ===================================================================
# --- MCTS Controller Parameters (<<< NEW >>>) ---
# ===================================================================
# Number of MCTS iterations to run at each single time step. This is the "thinking" budget.
NUM_ITERATIONS = 300
# C: The exploration constant in the UCB1 formula. Higher C encourages more exploration.
EXPLORATION_CONSTANT = 40.0
# How far into the future the random rollouts go.
ROLLOUT_HORIZON = 15

# ===================================================================
# --- 2. The MCTS Node Class (<<< NEW >>>) ---
# ===================================================================

class MCTSNode:
    """A node in the Monte Carlo Search Tree."""
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # The action that led to this state
        self.children = []
        self.visits = 0
        self.value = 0.0  # Total cost from simulations

    def is_fully_expanded(self, num_actions):
        return len(self.children) == num_actions

    def select_best_child_ucb(self):
        """Selects the best child using the UCB1 formula for minimizing cost."""
        best_score = float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                # If a child has not been visited, it's an immediate priority
                return child

            # UCB1 formula adapted for cost minimization (we want low average cost)
            exploit_score = child.value / child.visits
            explore_score = math.sqrt(math.log(self.visits) / child.visits)
            
            # We subtract the exploration term because we want to explore nodes
            # with fewer visits, which makes their score lower (better).
            score = exploit_score - EXPLORATION_CONSTANT * explore_score
            
            if score < best_score:
                best_score = score
                best_child = child
        return best_child

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
# --- 4. The MCTS Optimizer (<<< THE NEW BRAIN >>>) ---
# ===================================================================

def mcts_optimizer(current_state):
    """
    Finds the best action by building and searching a Monte Carlo Tree.
    """
    # Define the discrete set of possible actions
    control_step = 0.05
    possible_d_thetas = [-control_step, 0, control_step]
    actions = []
    for dt1 in possible_d_thetas:
        for dt2 in possible_d_thetas:
            if dt1 == 0 and dt2 == 0: continue
            actions.append(np.array([dt1, dt2]))

    root = MCTSNode(state=current_state)

    for _ in range(NUM_ITERATIONS):
        # --- 1. Selection ---
        node = root
        while node.is_fully_expanded(len(actions)) and node.children:
            node = node.select_best_child_ucb()

        # --- 2. Expansion ---
        if not node.is_fully_expanded(len(actions)):
            untried_actions = [a for a in actions if not any(np.array_equal(c.action, a) for c in node.children)]
            action_to_try = random.choice(untried_actions)
            next_state = node.state + action_to_try
            child_node = MCTSNode(state=next_state, parent=node, action=action_to_try)
            node.children.append(child_node)
            node = child_node # Move to the new node for rollout

        # --- 3. Simulation (Rollout) ---
        rollout_state = node.state
        total_rollout_cost = 0
        for _ in range(ROLLOUT_HORIZON):
            random_action = random.choice(actions)
            rollout_state = rollout_state + random_action
            total_rollout_cost += distance_from_goal(rollout_state[0], rollout_state[1])
        
        # --- 4. Backpropagation ---
        while node is not None:
            node.visits += 1
            node.value += total_rollout_cost
            node = node.parent

    # After all iterations, choose the action that leads to the most visited child
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action

# ===================================================================
# --- 5. VISUALIZATION AND MAIN LOOP (Mostly Unchanged) ---
# ===================================================================

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.3)
arm_line, = plt.plot([], [], 'b-o', linewidth=3, markersize=5, label='Robotic Arm')
target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8, label='Target')
end_effector_path, = plt.plot([], [], 'g:', linewidth=1, label='Path')
path_history = []
ax.set_xlim(-200, 200); ax.set_ylim(-200, 200); ax.set_aspect('equal')
ax.grid(True); ax.set_title("2DOF Arm with Monte Carlo Tree Search"); ax.legend()
ax_theta1 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.1, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))

def update_plot():
    x0, y0 = 0, 0; x1 = L1 * np.cos(theta1); y1 = L1 * np.sin(theta1)
    x2, y2 = forward_kinematics(theta1, theta2)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    path_history.append((x2, y2)); path_x, path_y = zip(*path_history)
    end_effector_path.set_data(path_x, path_y)
    slider1.set_val(np.degrees(theta1)); slider2.set_val(np.degrees(theta2))
    fig.canvas.draw_idle()

def mcts_control_step(event):
    global theta1, theta2
    current_state = np.array([theta1, theta2])
    current_dist = distance_from_goal(current_state[0], current_state[1])
    print(f"Distance to Goal: {current_dist:.2f}")

    if current_dist < goal_threshold:
        print("Goal Reached!"); timer.stop(); return

    # Call the MCTS planner to get the single best action
    best_action = mcts_optimizer(current_state)
    dtheta1, dtheta2 = best_action
    
    theta1 += dtheta1; theta2 += dtheta2
    update_plot()

# --- Setup and start the animation timer ---
timer = fig.canvas.new_timer(interval=50) # The interval should be > thinking time
timer.add_callback(mcts_control_step, None)
ax_start = plt.axes([0.8, 0.025, 0.1, 0.04])
btn_start = Button(ax_start, 'Start')
btn_start.on_clicked(lambda event: timer.start())

update_plot()
plt.show()