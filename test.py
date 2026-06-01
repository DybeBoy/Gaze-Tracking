import os
import numpy as np
from library import *

dir_path = "files/training/sesh4"

try:
    folders = []
    for f in os.listdir(dir_path):
        if not os.path.isfile(os.path.join(dir_path, f)):
            folders.append(dir_path + "/" + f)
    print(folders)

    files = []
    for i in folders:
        for f in os.listdir(i):
            if os.path.isfile(os.path.join(i, f)):
                files.append(i + "/" + f)
    print(files)

    data = 0

    if files == []:
        print("using dir_path")
        data = np.load(dir_path + "/data.npz", allow_pickle=True)
    else:
        print("using files")
        with np.load(files[0]) as tmp:
            keys = list(tmp.keys())

        acc = {k: [] for k in keys}
        for f in files:
            with np.load(f) as data:
                for k in keys:
                    acc[k].append(data[k])

        data = {k: np.concatenate(acc[k], axis=0) for k in keys}

    print(len(data["image_ids"]))

except FileNotFoundError:
    os.makedirs(dir_path + "/images", exist_ok=True)
    print("folder made")