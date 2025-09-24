import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# --- Link lengths ---
L1, L2, L3 = 50, 50, 50

# --- Target position ---
target_x, target_y = -100, 20

# --- Initial joint angles (radians) ---
theta1 = np.radians(100)
theta2 = np.radians(30)
theta3 = np.radians(-20)

# --- Initial angular velocities (radians/frame) ---
theta1_vel, theta2_vel, theta3_vel = 0.0, 0.0, 0.0

# --- Setup figure ---
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.5)
arm_line, = plt.plot([], [], '-o', linewidth=3, markersize=5)
ax.set_xlim(-220, 220)
ax.set_ylim(-50, 220)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("3DOF Robotic Arm with Velocity Control")

# --- Sliders (read-only) ---
ax_theta1 = plt.axes([0.25, 0.35, 0.65, 0.03])
ax_theta2 = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_theta3 = plt.axes([0.25, 0.25, 0.65, 0.03])

slider1 = Slider(ax_theta1, 'Theta1 (°)', -180, 180, valinit=np.degrees(theta1))
slider2 = Slider(ax_theta2, 'Theta2 (°)', -180, 180, valinit=np.degrees(theta2))
slider3 = Slider(ax_theta3, 'Theta3 (°)', -180, 180, valinit=np.degrees(theta3))

# --- Velocity control buttons ---
btn_size = [0.1, 0.04]
ax_inc1 = plt.axes([0.25, 0.15, *btn_size])
ax_dec1 = plt.axes([0.36, 0.15, *btn_size])
ax_inc2 = plt.axes([0.55, 0.15, *btn_size])
ax_dec2 = plt.axes([0.66, 0.15, *btn_size])
ax_inc3 = plt.axes([0.25, 0.08, *btn_size])
ax_dec3 = plt.axes([0.36, 0.08, *btn_size])

btn_inc1 = Button(ax_inc1, '↑ θ1')
btn_dec1 = Button(ax_dec1, '↓ θ1')
btn_inc2 = Button(ax_inc2, '↑ θ2')
btn_dec2 = Button(ax_dec2, '↓ θ2')
btn_inc3 = Button(ax_inc3, '↑ θ3')
btn_dec3 = Button(ax_dec3, '↓ θ3')

# --- Forward Kinematics ---
def forward_kinematics(t1, t2, t3):
    x1 = L1 * np.cos(t1)
    y1 = L1 * np.sin(t1)
    x2 = x1 + L2 * np.cos(t1 + t2)
    y2 = y1 + L2 * np.sin(t1 + t2)
    x3 = x2 + L3 * np.cos(t1 + t2 + t3)
    y3 = y2 + L3 * np.sin(t1 + t2 + t3)
    return (x1, y1), (x2, y2), (x3, y3)

# --- Inverse Kinematics ---
def inverse_kinematics(x, y, phi=45):
    """
    Solve IK for 3DOF planar arm.
    phi = desired end effector orientation.
    """
    phi = theta1 + theta2 + theta3
    
    # Virtual target for 2DOF
    x_prime = x - L3 * np.cos(phi)
    y_prime = y - L3 * np.sin(phi)

    # Distance check
    D = (x_prime**2 + y_prime**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1:
        return None  # unreachable

    theta2_sol = np.arccos(D)
    theta1_sol = np.arctan2(y_prime, x_prime) - np.arctan2(L2*np.sin(theta2_sol), L1+L2*np.cos(theta2_sol))
    theta3_sol = phi - (theta1_sol + theta2_sol)

    return theta1_sol, theta2_sol, theta3_sol

# --- Update function ---
def update_plot():
    global theta1, theta2, theta3, theta1_vel, theta2_vel, theta3_vel
    e1, e2, e3 = 0,0,0

    # Forward kinematics
    (x1, y1), (x2, y2), (x3, y3) = forward_kinematics(theta1, theta2, theta3)

    arm_line.set_data([0, x1, x2, x3], [0, y1, y2, y3])
    ax.plot(target_x, target_y, 'ro', markersize=8)

    # Inverse kinematics (fix phi = 0)
    sol = inverse_kinematics(target_x, target_y, phi=0)
    print(sol)
    if sol:
        t1_target, t2_target, t3_target = sol

        # Error in joint space
        e1 = t1_target - theta1
        e2 = t2_target - theta2
        e3 = t3_target - theta3

        # Simple velocity update (sign controller)
        theta1_vel = np.sign(e1) * 0.006
        theta2_vel = np.sign(e2) * 0.006
        theta3_vel = np.sign(e3) * 0.006

    print("theta error: ", e1, e2, e3)
    # calculate error by distance
    error = ((target_y - y3)**2 + (target_x - x3)**2)**(1/2)
    print("xy error: ", error)

    # Update sliders
    slider1.set_val(np.degrees(theta1))
    slider2.set_val(np.degrees(theta2))
    slider3.set_val(np.degrees(theta3))

    fig.canvas.draw_idle()

# --- Animation loop using timer ---
def update_angles(event):
    global theta1, theta2, theta3
    theta1 += theta1_vel
    theta2 += theta2_vel
    theta3 += theta3_vel
    update_plot()

timer = fig.canvas.new_timer(interval=50)  # ~20 FPS
timer.add_callback(update_angles, None)
timer.start()

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

def inc_theta3(event):
    global theta3_vel
    theta3_vel += 0.01
def dec_theta3(event):
    global theta3_vel
    theta3_vel -= 0.01

btn_inc1.on_clicked(inc_theta1)
btn_dec1.on_clicked(dec_theta1)
btn_inc2.on_clicked(inc_theta2)
btn_dec2.on_clicked(dec_theta2)
btn_inc3.on_clicked(inc_theta3)
btn_dec3.on_clicked(dec_theta3)

# --- Initial draw ---
try:
    update_plot()
    plt.show()
except:
    sys.exit()
