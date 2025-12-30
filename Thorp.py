# Attenuation coefficient by Thorp (1961).

import numpy as np
import matplotlib.pyplot as plt

def att(f):
    y_att = f**2 *((3.01 * 10**(-4)) + ((43.7)/(4100 + f**2)) + ((0.109)/(1 + f**2)))
    return np.log10(y_att)

def water(f):
    y_wat = (f**2) *3.01 * 10**(-4)
    return np.log10(y_wat)

def magnesium_sulfate(f):
    y_magsul = (f**2) *43.7/(4100 + f**2)
    return np.log10(y_magsul)

def borate(f):
    y_bor = (f**2) * 0.109/(1 + f**2)
    return np.log10(y_bor)

x = np.arange(-2, 5, 0.01)
f = 10**(x)
y1 = att(f)
y2 = water(f)
y3 = magnesium_sulfate(f)
y4 = borate(f)

fig, ax = plt.subplots()

ax.plot(x, y1, label='attenuation', color='black')
ax.plot(x, y2, ':', label='water', color='blue')
ax.plot(x, y3, ':', label='magnesium sulfate', color='red')
ax.plot(x, y4, ':', label='borate', color='green')
plt.legend()
ax.set_xlabel('Log10( f [kHz])')
ax.set_ylabel('Log10( Attenuation coefficient [dB/km])')
plt.show()
