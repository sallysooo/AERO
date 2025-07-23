import numpy as np


y_seq = np.array([0,0,0,1,0,0,0,0,1,0]) # 10 packets
window_size = 4
stride = 2

all_indices = np.arange(window_size, len(y_seq) + 1, stride)
labels = [y_seq[i - window_size : i].max() for i in all_indices]
print(labels) # [1,1,0,1]

# window 1: [0,0,0,1] → 1
# window 2: [0,1,0,0] → 1
# window 3: [0,0,0,0] → 0
# window 4: [0,0,0,1] → 1
# [np.int64(1), np.int64(1), np.int64(0), np.int64(1)]


