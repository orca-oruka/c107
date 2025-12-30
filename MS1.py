# ランダム配置ソノブイフィールドのマルチスタティック探知性能評価.
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# parameters
det_range = 2       # detection range
s_quant = int(4)    # sources quantity
r_quant = int(9)    # receivers quantity
buoy_side = 10      # side length of buoy field
tgt_side = 12       # side length of detection area
db = 0.1            # direct blast effect
trial_1 = int(500)  # number of trial, reproducing the buoy field
trial_2 = int(500)  # number of trial, reproducing detection/nondetection point

count = 0
P = []
P_ave = []
P_max = 0
s_max = []
r_max = []
P_min = 1
s_min = []
r_min = []

# producing sonobuoy field
for i in range(trial_1):
    source = []
    for j in range(s_quant):
        x = buoy_side * (random() - 0.5)
        y = buoy_side * (random() - 0.5)
        source.append([x,y])
    receiver = []
    for j in range(r_quant):
        x = buoy_side * (random() - 0.5)
        y = buoy_side * (random() - 0.5)
        receiver.append([x,y])
    det = 0

    # producing detection points
    for j in range(trial_2):
        target = []
        x = tgt_side * (random() - 0.5)
        y = tgt_side * (random() - 0.5)

        # bistatic pairs
        for k in range(s_quant):

            for l in range(r_quant):
                R1 = np.sqrt((source[k][0] - x)**2 + (source[k][1] - y)**2)
                R2 = np.sqrt((receiver[l][0] - x)**2 + (receiver[l][1] - y)**2)
                R3 = np.sqrt((source[k][0] - receiver[l][0])**2 + (source[k][1] - receiver[l][1])**2)
                
                # detection
                if R1*R2 <= det_range**2:
                    if R1 + R2 - R3 >= db:
                        det = det + 1
                        target = 1
                
                if target == 1:
                    break
            
            if target == 1:
                break
        
    P.append(det / trial_2)
    P_ave.append(np.sum(P) /(i + 1))

    if P[i] > P_max:
        P_max = P[i]
        s_max = source
        r_max = receiver
    elif P[i] < P_min:
        P_min = P[i]
        s_min = source
        r_min = receiver

# plot probability of detection (PoD) and the average
c1,c2 = 'lime','black'
print(P_ave[trial_1 - 1])
print(np.std(P))
left = np.arange(0,trial_1)
height = P
plt.bar(left, height, align='center', color=c1)
plt.plot(P_ave, color=c2)
plt.show()

# plot sonobuoy arrangement for max/min PoD 
c3,c4 = 'red','blue'
l1,l2 = 'source','receiver'
plt.title('Max probability of detection')
ax = plt.axes()
r1 = patches.Rectangle(xy=(-buoy_side/2, -buoy_side/2), width=buoy_side, height=buoy_side, fill=True)
r2 = patches.Rectangle(xy=(-tgt_side/2, -tgt_side/2), width=tgt_side, height=tgt_side, fill=False)
ax.add_patch(r1)
ax.add_patch(r2)
for i in range(s_quant):
    plt.plot(s_max[i][0],s_max[i][1], marker='.', markersize=20, color=c3, label=l1)
for i in range(r_quant):
    plt.plot(r_max[i][0],r_max[i][1], marker='^', markersize=10, color=c4, label=l2)
plt.show()

plt.title('Min probability of detection')
ax = plt.axes()
r1 = patches.Rectangle(xy=(-buoy_side/2, -buoy_side/2), width=buoy_side, height=buoy_side, fill=True)
r2 = patches.Rectangle(xy=(-tgt_side/2, -tgt_side/2), width=tgt_side, height=tgt_side, fill=False)
ax.add_patch(r1)
ax.add_patch(r2)
for i in range(s_quant):
    plt.plot(s_min[i][0],s_min[i][1], marker='.', markersize=20, color=c3, label=l1)
for i in range(r_quant):
    plt.plot(r_min[i][0],r_min[i][1], marker='^', markersize=10, color=c4, label=l2)
plt.show()
