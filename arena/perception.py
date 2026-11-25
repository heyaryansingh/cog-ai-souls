"""Perception stage 1: from raw keypoints to boss-centric pose features, then unsupervised motion
syllables (MoSeq-style sticky HMM over pose + velocity), scored against the true animation segments.
"""
from __future__ import annotations

import numpy as np

from .skeleton import J, N_JOINTS


def boss_centric(kp_flat: np.ndarray) -> np.ndarray:
    """(T, 51) keypoints in the player's lock-on frame -> (T, 51) pose in the boss's own frame:
    centred on the pelvis (xy), rotated so the hip line's normal points along +x. Dropped joints (all
    zeros) are filled from the previous frame."""
    kp = kp_flat.reshape(-1, N_JOINTS, 3).copy()
    for t in range(1, len(kp)):
        miss = np.all(kp[t] == 0, axis=1)
        kp[t, miss] = kp[t - 1, miss]
    pel = kp[:, J["pelvis"], :2]
    left = kp[:, J["l_hip"], :2] - kp[:, J["r_hip"], :2]
    fwd_ang = np.arctan2(-left[:, 0], left[:, 1])         # forward = left rotated by -90 degrees
    c, s = np.cos(-fwd_ang), np.sin(-fwd_ang)
    xy = kp[:, :, :2] - pel[:, None]
    x = c[:, None] * xy[..., 0] - s[:, None] * xy[..., 1]
    y = s[:, None] * xy[..., 0] + c[:, None] * xy[..., 1]
    return np.stack([x, y, kp[..., 2]], -1).reshape(len(kp), -1)


def features(pose: np.ndarray, pca=None, smooth: int = 3):
    """Smoothed pose + velocity; PCA fitted on the first call if `pca` is None."""
    from sklearn.decomposition import PCA
    if smooth > 1:
        k = np.ones(smooth) / smooth
        pose = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, pose)
    if pca is None:
        pca = PCA(n_components=12).fit(pose)
    z = pca.transform(pose)
    dz = np.vstack([np.zeros((1, z.shape[1])), np.diff(z, axis=0)])
    return np.hstack([z, 3.0 * dz]), pca
