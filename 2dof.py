import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Link lengths
L1 = 75
L2 = 75

target_x, target_y = -100, 20

# Initial joint angles (radians)
theta1 = np.radians(100)
theta2 = np.radians(30)

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

# --- Update function ---
def update_plot():
    global theta1, theta2, theta1_vel, theta2_vel
    # Joint positions
    x0, y0 = 0, 0
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    arm_line.set_data([x0, x1, x2], [y0, y1, y2])
    target_dot, = ax.plot(target_x, target_y, 'ro', markersize=8)

    # calculate error by distance
    error = ((target_y - y2)**2 + (target_x - x2)**2)**(1/2)

    # calculate IK - finding D
    D = ((L1**2 + L2**2) - (target_x**2 + target_y**2))/(2*L1*L2)
    print("Validity score: ", D)
    theta_2 = math.acos(-D)
    # print(math.acos(-D))
    theta_1 = math.atan2(target_y,target_x) - math.atan2((L2*math.sin(theta_2)), (L1+(L2*math.cos(theta_2))))
    print("target: ", theta_2, theta_1)
    print("current: ", theta2, theta1)

    error_theta1 = round(theta_1 - theta1, 2)
    error_theta2 = round(theta_2 - theta2, 2)

    print("theta error: ", error_theta1, error_theta2)
    print("xy error: ", error)

    theta1_vel = np.sign(error_theta1)*0.006
    theta2_vel = np.sign(error_theta2)*0.006

    slider1.set_val(np.degrees(theta1))
    slider2.set_val(np.degrees(theta2))
    fig.canvas.draw_idle()

# --- Animation loop using timer ---
def update_angles(event):
    global theta1, theta2
    theta1 += theta1_vel
    theta2 += theta2_vel
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

btn_inc1.on_clicked(inc_theta1)
btn_dec1.on_clicked(dec_theta1)
btn_inc2.on_clicked(inc_theta2)
btn_dec2.on_clicked(dec_theta2)

# --- Initial draw ---
update_plot()
plt.show()