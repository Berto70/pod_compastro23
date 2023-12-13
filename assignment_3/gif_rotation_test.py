import numpy as np
from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.mplot3d import Axes3D

path = '/home/bertinelli/pod_compastro23/Fireworks/fireworks_test'
intr = np.load(path + '/data/ass_3/pos_i.npy', allow_pickle=True)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

# Calculate the number of extra frames to add at the beginning and the end
pause_duration = 0.5  # pause duration in seconds start
pause_duration_end = 1  # pause duration in seconds end
frame_rate = 15  # frame rate of the animation
extra_frames = pause_duration * frame_rate
extra_frames_end = pause_duration_end * frame_rate

# Create an array of frame indices
frames = np.concatenate([
    np.full(int(extra_frames), 0),  # initial frame (pause
    np.arange(0, len(intr)-1, 100),  # original frames
    np.full(int(extra_frames_end), len(intr)-2)  # extra frames at the end
])

def update_pos(frame): 
    ax.clear()

    # Increase the step size to speed up the lines
    step_size = 2
    x1 = intr[:frame * step_size, 0, 0]
    x2 = intr[:frame * step_size, 1, 0]
    x3 = intr[:frame * step_size, 2, 0]
    y1 = intr[:frame * step_size, 0, 1]
    y2 = intr[:frame * step_size, 1, 1]
    y3 = intr[:frame * step_size, 2, 1]
    z1 = intr[:frame * step_size, 0, 2]
    z2 = intr[:frame * step_size, 1, 2]
    z3 = intr[:frame * step_size, 2, 2]

    ax.scatter(intr[0,0, 0], intr[0,0, 1], intr[0,0, 2], color="tab:red", s=100, marker="x", zorder=10, label="Initial Position")
    ax.scatter(intr[0,1, 0], intr[0,1,1], intr[0,1,2], color="tab:red", s=100, marker="x", zorder=10)
    ax.scatter(intr[0,2,0], intr[0,2,1], intr[0,2,2], color="tab:red", s=100, marker="x", zorder=10)
    ax.plot(x1, y1, z1, color="tab:blue", label="Body 1", alpha=0.8)
    ax.plot(x2, y2, z2, color="tab:orange", label="Body 2", alpha=0.8)
    ax.plot(x3, y3, z3, color="tab:green", label="Body 3", alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax.set_xlim(-3, 6)
    ax.set_ylim(-0.5, 2)
    ax.set_zlim(-3, 6)

    ax.view_init(30, frame)

print("Starting Position Animation")

gif_pos = animation.FuncAnimation(fig=fig, func=update_pos, frames=frames, interval=10)

gif_pos.save("/home/bertinelli/pod_compastro23/assignment_3/animations/rotating_random.gif", writer="pillow")

print("Position Animation Saved")