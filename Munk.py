# Munk's sound speed profile
import numpy as np
import matplotlib.pyplot as plt

# parameters
c0 = 1500		# sound speed at sound channel axis [m/s]
eps = 0.00737   # coeff
B = 1000        # scale depth
z0 = 1000       # depth of sound channel axis
z_max = 5000    # max depth of calc
z_step = 1      # depth step for calc

def c(z):
    eta = 2 * (z - z0) / B
    return c0 * (1 + eps * (eta - 1 + np.exp(-eta)))

z = np.arange(0, z_max, z_step)
y = c(z)

fig, ax = plt.subplots()
ax.invert_yaxis()
ax.plot(y, z)
ax.set_xlabel('sound speed [m/s]')
ax.set_ylabel('depth [m]')
plt.show()
