import numpy as np
import os
from pathlib import Path
import cv2
import random
#exit()
"""
    Please älä vittu runnaa tätä skriptii kun se paskoo datan jos se tallentaa tän paskan. 
    joten ainakin kato miten menee enne ku teet mitään paskaa.
"""

data_dir = Path("data/training") # vaihtoon vaan täältä kiitos jooko vovsruigffe5jihnfoe5jtfui7gh5inguieh5igviueheee
npz_files = list(data_dir.rglob("data.npz"))

print(f"Found {len(npz_files)} data.npz files")


# Loop through each file
for npz_path in npz_files:
    print(f"\nProcessing: {npz_path}")
    """
    # Load the npz file
    data = np.load(npz_path)
    print(data.files)
    image_ids = data["image_ids"]
    head_rots = data["head_rots"]
    head_pos = data["head_pos"]
    head_depths = data["head_depths"]
    for i in range(len(head_depths)):
        head_depths[i] = head_depths[i] / 1080 * 10000000.0
    labels = data["labels"]

    # Save the modified data back to the npz file
    np.savez(
        f"{npz_path.parent}/data.npz",
        image_ids=image_ids,
        head_rots=head_rots,
        head_pos=head_pos,
        head_depths=head_depths,
        labels=labels
    )
    """

    data = np.load(npz_path)

    image_ids = data["image_ids"]
    head_rots = data["head_rots"]
    head_pos = data["head_pos"]
    head_depths = data["head_depths"]
    labels = data["labels"]

    print(head_depths[0])