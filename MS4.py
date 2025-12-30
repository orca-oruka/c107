# 正六角形の頂点に受波器, 中心に音源を配置したマルチスタティックソノブイフィールド.
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

# parameters
det_range = 3       # detection range
s_quant = 1         # sources quantity 
r_quant = 6         # receivers quantity
rec_loc_diff = 0.1  # location movement step of receivers
rec_dist = 15       # upper limitation of source-receiver distance
tgt_side = 20       # side length of detection area
tgt_diff = 0.1      # distance of detection points
db = 0.1            # direct blast effect
trial = int(10000)  # number of trial, producing detection point

det = []
P = []
P_max = 0

# initial sonobuoy field
source = [0.0,0.0]
receiver = [[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0]]

# sonobuoy field expansion
for i in range(int(rec_dist/rec_loc_diff) + 1):
    receiver[0] = [receiver[0][0] + rec_loc_diff, receiver[0][1]]
    receiver[1] = [receiver[1][0] + rec_loc_diff / 2, receiver[1][1] + rec_loc_diff * np.sqrt(3) / 2]
    receiver[2] = [receiver[2][0] - rec_loc_diff / 2, receiver[2][1] + rec_loc_diff * np.sqrt(3) / 2]
    receiver[3] = [receiver[3][0] - rec_loc_diff, receiver[3][1]]
    receiver[4] = [receiver[1][0] - rec_loc_diff / 2, receiver[1][1] - rec_loc_diff * np.sqrt(3) / 2]
    receiver[5] = [receiver[1][0] + rec_loc_diff / 2, receiver[1][1] - rec_loc_diff * np.sqrt(3) / 2]
    det.append(0)

	# detection points
    x = -tgt_side/2
    for j in range(int(tgt_side/tgt_diff)):
        target = 0
        x = x + tgt_diff
        y = -tgt_side/2
        for k in range(int(tgt_side/tgt_diff)):
            y = y + tgt_diff

			#detection
            for l in range(len(receiver)):
                R1 = np.sqrt((source[0] - x)**2 + (source[1] - y)**2)
                R2 = np.sqrt((receiver[l][0] - x)**2 + (receiver[l][1] - y)**2)
                R3 = np.sqrt((source[0] - receiver[l][0])**2 + (source[1] - receiver[l][1])**2)

                if R1*R2 <= det_range**2 and R1 + R2 - R3 >= db:
                    det[i] = det[i] + 1
                    target = 1
                    break

                if target == 1:
                    break
            if target == 1:
                break
            
    P.append(det[i] / trial)

    if P[i] > P_max:
        P_max = P[i]

# plot
c1 = 'lime'
print(P_max)
print(np.std(P))
plt.xlabel('source-receiver distance')
plt.ylabel('detection probability')
left = np.arange(0, rec_dist + rec_loc_diff, rec_loc_diff)
plt.bar(left, P, align='center', color=c1)
plt.show()
