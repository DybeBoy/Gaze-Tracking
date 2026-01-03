import numpy as np

# Load the npz file
data = np.load('data/validation/straight/data.npz', allow_pickle=True)

image_ids = data["image_ids"]
print(image_ids[100])

# Update image_ids
for i, img in enumerate(image_ids):
    image_ids[i] = f"straight_{img}"

# Save the updated metadata back to the file
np.savez(
    'data/validation/straight/data.npz',
    image_ids=image_ids,
    head_rots=data["head_rots"],
    head_pos=data["head_pos"],
    head_depth=data["head_depth"],
    labels=data["labels"]
)

print(image_ids[100])
print(np.load('data/validation/straight/data.npz', allow_pickle=True)["image_ids"][100])