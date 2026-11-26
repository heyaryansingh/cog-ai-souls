"""Filmstrip for manual move annotation: frames [a, b) every `step`, cropped around the screen centre, labelled."""
import sys, subprocess, numpy as np
from PIL import Image, ImageDraw
path, a, b, step, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
fps = float(sys.argv[6]) if len(sys.argv) > 6 else 30.0
W, H = 320, 180
cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{a / fps:.3f}", "-i", path, "-frames:v", str(b - a),
       "-vf", f"crop=960:540:160:60,scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
raw = subprocess.run(cmd, capture_output=True).stdout
fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
sel = list(range(0, len(fr), step)); cols = 6; rows = (len(sel) + cols - 1) // cols
sheet = Image.new("RGB", (cols * W, rows * H))
for n, i in enumerate(sel):
    im = Image.fromarray(fr[i]); d = ImageDraw.Draw(im); d.rectangle((0, 0, 60, 14), fill=(0, 0, 0)); d.text((2, 1), str(a + i), fill=(255, 255, 0))
    sheet.paste(im, ((n % cols) * W, (n // cols) * H))
sheet.save(out); print(out, len(sel))
