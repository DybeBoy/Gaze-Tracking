import numpy as np

data = np.load('data/training/straight/data.npz', allow_pickle=True)

print(data["head_rots"])