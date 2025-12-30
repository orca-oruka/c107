# detection area of a pair of buoys as monostatic or bistatic
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# parameters
R1 = 1  #monostatic detection range
R2 = 1.5  #bistatic detection range
d_max = 5   #half of max distance between two buoys, equals to x-value of the buoy position
step = 0.05 

# monostatic
def f1(x):
    return np.sqrt(R1**2 - (x - d)**2)

# bistatic
def f2(x):
    return np.sqrt(np.sqrt(4 * (d**2) * (x**2) + R2**4) - x**2 - d**2)

dist = np.zeros(int(d_max/step)+1)
mono = np.zeros(int(d_max/step)+1)
bi = np.zeros(int(d_max/step)+1)
d = 0
g1 = 0
h1 = d + R1
g2 = 0
h2 = np.sqrt(d**2 + R2**2)
dist[0] = 0
mono[0] = 4 * integrate.quad(f1, g1, h1)[0]
bi[0] = 4 * integrate.quad(f2, g2, h2)[0]
for i in range(int(d_max/step)):
    d = d + step
    g1 = max(0, d - R1)
    h1 = d + R1
    if d <= R2:
        g2 = 0
    else:
        g2 = np.sqrt(d**2 - R2**2)
    h2 = np.sqrt(d**2 + R2**2)
    dist[i+1] = 2 * d
    mono[i+1] = 4 * integrate.quad(f1, g1, h1)[0]
    bi[i+1] = 4 * integrate.quad(f2, g2, h2)[0]

plt.xlabel('buoy distance')
plt.ylabel('detection area')
plt.plot(dist, mono, color='red', label='monostatic')
plt.plot(dist, bi, color='blue', label='bistatic')
plt.legend()
plt.show()
