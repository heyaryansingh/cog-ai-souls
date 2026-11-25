"""Motion-capture clips (dm_control CMU 2020 retarget, 836 clips, 3.3 h, 30 ms frames) as qpos tracks."""
from __future__ import annotations

import os

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5 = os.environ.get("CMU_H5") or next((p for p in [os.path.join(_ROOT, "data", "mocap", "cmu_2020.h5"),
                                                      "/home/user/mocap/cmu_2020.h5"] if os.path.exists(p)),
                                         os.path.join(_ROOT, "data", "mocap", "cmu_2020.h5"))
DT = 0.03


class Clip:
    def __init__(self, name, pos, quat, joints):
        self.name, self.pos, self.quat, self.joints = name, pos, quat, joints   # (T,3) (T,4) (T,56)

    def __len__(self):
        return len(self.pos)

    def qpos(self, t, scale=1.0, origin=np.zeros(3), yaw=0.0):
        """Root placed relative to frame 0, rotated by yaw about z and scaled; joints unchanged."""
        t = min(max(int(t), 0), len(self) - 1)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        d = self.pos[t] - self.pos[0] * np.array([1, 1, 0])
        p = origin + Rz @ (d * scale)
        q = _qmul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), self.quat[t])
        return np.concatenate([p, q, self.joints[t]])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def load(names=None, path=None):
    path = path or H5
    out = {}
    with h5py.File(path, "r") as f:
        for k in (names or list(f.keys())):
            w = f[k]["walkers/walker_0"]
            out[k] = Clip(k, w["position"][:].T, w["quaternion"][:].T, w["joints"][:].T)
    return out
