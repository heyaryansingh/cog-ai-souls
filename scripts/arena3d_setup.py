"""One-time setup for Arena3D: download the CMU mocap retarget (476 MB) and build the open move library.

    pip install mujoco h5py mediapipe pillow
    python scripts/arena3d_setup.py            # -> data/mocap/cmu_2020.h5, checkpoints/arena3d_units.pkl
Set CMU_H5 to use a copy elsewhere. Pose model files go to checkpoints/pose/.
"""
import os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5 = os.environ.get("CMU_H5", os.path.join(ROOT, "data", "mocap", "cmu_2020.h5"))
URL = "https://storage.googleapis.com/dm_control/cmu_2020_dfe3e9e0.h5"
POSE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"

if not os.path.exists(H5):
    os.makedirs(os.path.dirname(H5), exist_ok=True)
    print("downloading CMU mocap ->", H5)
    urllib.request.urlretrieve(URL, H5)
os.makedirs(os.path.join(ROOT, "checkpoints", "pose"), exist_ok=True)
for v in ("lite", "full"):
    p = os.path.join(ROOT, "checkpoints", "pose", f"pose_landmarker_{v}.task")
    if not os.path.exists(p):
        urllib.request.urlretrieve(POSE.format(v=v), p)
os.environ["CMU_H5"] = H5
from arena3d import library as L, mocap
mocap.H5 = H5
U = L.build_library(mocap.load(path=H5))
L.save(U, os.path.join(ROOT, "checkpoints", "arena3d_units.pkl"))
print(len(U), "units")
