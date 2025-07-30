import numpy as np
import math
import sys
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Link lengths
L1 = 75
L2 = 75

target_x, target_y = 20, 80

# Initial joint angles (radians)
theta1 = np.radians(100)
theta2 = np.radians(30)
error = 1000

# Initial angular velocities (radians/frame)
theta1_vel = 0.0
theta2_vel = 0.0

# --- Setup figure ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.4)
arm_line, = plt.plot([], [], '-o', linewidth=3, markersize=5)
ax.set_xlim(-200, 200)
ax.set_ylim(0, 200)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("2DOF Robotic Arm with Velocity Control")

# --- Sliders (read-only) ---
ax_theta1 = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.2, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))

# --- Velocity control buttons ---
btn_size = [0.1, 0.04]
ax_inc1 = plt.axes([0.25, 0.1, *btn_size])
ax_dec1 = plt.axes([0.36, 0.1, *btn_size])
ax_inc2 = plt.axes([0.55, 0.1, *btn_size])
ax_dec2 = plt.axes([0.66, 0.1, *btn_size])

btn_inc1 = Button(ax_inc1, '↑ θ1')
btn_dec1 = Button(ax_dec1, '↓ θ1')
btn_inc2 = Button(ax_inc2, '↑ θ2')
btn_dec2 = Button(ax_dec2, '↓ θ2')

# Discretization
angle_step = 10
angles = np.arange(0, 180, angle_step)  # 0 to 170 degrees
n_angles = len(angles)

# Define state space: all combinations of (theta1, theta2)
states = list(product(angles, angles))  # 18 x 18 = 324 states

# Define action space: change in angles
# Actions = [d_theta1, d_theta2]
angle_changes = [-angle_step, 0, angle_step]
actions = list(product(angle_changes, angle_changes))  # 9 actions

# Print sample
print(f"Total states: {len(states)}")
print(f"Sample state: {states[0]}")
print(f"Actions (angle changes): {actions}")

# --- Update function ---
def apply_action(state, action):
    global theta1, theta2
    theta1, theta2 = state
    d_theta1, d_theta2 = action

    # Apply the action
    new_theta1 = np.clip(theta1 + d_theta1, 0, 170)
    new_theta2 = np.clip(theta2 + d_theta2, 0, 170)

    theta1, theta2 = new_theta1, new_theta2
    return (new_theta1, new_theta2)

def compute_reward(error, threshold=10):
    distance = error
    # Option 1: shaped reward (negative distance)
    reward = -distance

    # Option 2: bonus for being very close
    if distance < threshold:
        reward += 100  # big bonus for reaching target

    return reward

def value_iteration(states, actions, gamma=0.9, iterations=10):
    global error
    V = {s: 0 for s in states}         # Value function
    policy = {s: actions[0] for s in states}  # Initial dummy policy

    for itr in range(iterations):
        new_V = {}
        index = 0
        for state in states:
            
            index += 1
            max_value = float('-inf')
            best_action = None

            for action in actions:
                next_state = apply_action(state, action)
                reward = compute_reward(error)
                value = reward + gamma * V[next_state]

                if value > max_value:
                    max_value = value
                    best_action = action
                
                print(itr, index, state, action, reward, value)
                update_plot()
                # plt.pause(0.001)
                # plt.plot(block=False)


            new_V[state] = max_value
            policy[state] = best_action
        V = new_V

    return V, policy

def update_plot():
    global theta1, theta2, theta1_vel, theta2_vel, error
    
    # Joint positions
    x0, y0 = 0, 0
    x1 = L1 * np.cos(np.radians(theta1))
    y1 = L1 * np.sin(np.radians(theta1))
    x2 = x1 + L2 * np.cos(np.radians(theta1) + np.radians(theta2))
    y2 = y1 + L2 * np.sin(np.radians(theta1) + np.radians(theta2))
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8)

    # calculate error by distance
    error = ((target_y - y2)**2 + (target_x - x2)**2)**(1/2)

    # calculate IK - finding D
    # D = ((L1**2 + L2**2) - (target_x**2 + target_y**2))/(2*L1*L2)
    # print("Validity score: ", D)
    # theta_2 = math.acos(-D)
    # print(math.acos(-D))
    # theta_1 = math.atan2(target_y,target_x) - math.atan2((L2*math.sin(theta_2)), (L1+(L2*math.cos(theta_2))))
    # print("target: ", theta_2, theta_1)
    # print("current: ", theta2, theta1)

    # error_theta1 = round(theta_1 - theta1, 2)
    # error_theta2 = round(theta_2 - theta2, 2)

    # print("theta error: ", error_theta1, error_theta2)
    # print("xy error: ", error)

    # theta1_vel = np.sign(error_theta1)*0.006
    # theta2_vel = np.sign(error_theta2)*0.006

    slider1.set_val(theta1)
    slider2.set_val(theta2)
    fig.canvas.draw_idle()

# --- Animation loop using timer ---
def update_angles(event):
    global theta1, theta2
    theta1 += theta1_vel
    theta2 += theta2_vel
    # update_plot()

# timer = fig.canvas.new_timer(interval=50)  # ~20 FPS
# timer.add_callback(update_angles, None)
# timer.add_callback(value_iteration(states, actions), None)
# timer.start()

# timer2 = fig.canvas.new_timer(interval=5)  # ~20 FPS
# timer2.add_callback(value_iteration(states, actions), None)
# timer2.start()

# --- Button callbacks ---
def inc_theta1(event):
    global theta1_vel
    theta1_vel += 0.01

def dec_theta1(event):
    global theta1_vel
    theta1_vel -= 0.01

def inc_theta2(event):
    global theta2_vel
    theta2_vel += 0.01

def dec_theta2(event):
    global theta2_vel
    theta2_vel -= 0.01

btn_inc1.on_clicked(inc_theta1)
btn_dec1.on_clicked(dec_theta1)
btn_inc2.on_clicked(inc_theta2)
btn_dec2.on_clicked(dec_theta2)


def simulate_policy(start_state, policy):
    global theta1, theta2, error
    # path = [start_state]
    state = theta1, theta2

    for itr in range(100):
        action = policy[state]
        next_state = apply_action(state, action)
        # path.append(next_state

        theta1, theta2 = next_state
        # Check if target reached
        # dist = np.linalg.norm(np.array(end_effector) - np.array(target_xy))
        if error < 2:
            print(error)
            plt.plot()

        state = next_state

        print(itr, state)
        update_plot()
        print(".")
        plt.pause(1)
        print("..")
        plt.plot(block=False)
        print("done")

    return 0

# --- Initial draw ---

try:
    # update_plot()
    # plt.show(block=False)
    value, policy = value_iteration(states, actions)
    path = simulate_policy((0, 0), policy)
    
except Exception as e:
    print(e)
    sys.exit()