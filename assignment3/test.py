import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Assuming you have defined your particles, evolve function, and other necessary modules

# Sample data or placeholder values
positions = np.random.random((40, 1, 2))  # Placeholder positions for demonstration

fig, ax = plt.subplots()
scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], c="b", s=5, label="Star 1")
ax.legend()

def update(frame):
    x = positions[frame, :, 0]
    y = positions[frame, :, 1]
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return (scat,)

ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30, blit=True)
plt.show()
