"""Per-frame timeline from souls gameplay video: player HP bar fill (top-left HUD, red), stamina bar fill (green),
and boss-region motion energy (frame difference in the screen centre). HP drops = hit frames; motion rises = move
onsets. Output: npz with arrays per frame."""
import sys, numpy as np, subprocess
path, out = sys.argv[1], sys.argv[2]
W, H = 640, 360                     # decode at half resolution
cmd = ["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
hp, st, mot, prev = [], [], [], None
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.int16)
    row = f[16:19, 45:330]                                      # HP bar band (half-res coords)
    red = (row[..., 0] > 110) & (row[..., 1] < 80) & (row[..., 2] < 80)
    hp.append(red.any(0).sum())
    row = f[29:32, 45:330]
    grn = (row[..., 1] > 90) & (row[..., 1] > row[..., 0] + 20)
    st.append(grn.any(0).sum())
    c = f[60:300, 120:520].mean(2)
    mot.append(0.0 if prev is None else float(np.abs(c - prev).mean())); prev = c
np.savez(out, hp=np.array(hp), st=np.array(st), mot=np.array(mot))
print(out, len(hp), "frames")
