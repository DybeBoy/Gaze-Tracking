import os
import numpy as np

saved_models = os.listdir("saved_models")
length = len(saved_models)

print("\n")
for i in range(length):
    data = np.load(f"saved_model_extra/head_scale{i}.npy")
    
    head_scale = float(data[0]) if len(data) > 0 else 0
    val_loss = float(data[1]) if len(data) > 1 else 0
    
    print(f"Model {i}: Head scale = {head_scale}, Validation loss = {val_loss}") 