import argparse
import time
import numpy as np
from PIL import Image
from seam.carve import carve

p = argparse.ArgumentParser(description="Content-aware image resizing.")
p.add_argument("input")
p.add_argument("output")
p.add_argument("--seams", type=int, required=True, help="number of vertical seams to remove")
p.add_argument("--max-size", type=int, default=0, help="downscale long edge before carving")
a = p.parse_args()

im = Image.open(a.input).convert("RGB")
if a.max_size:
    im.thumbnail((a.max_size, a.max_size))
img = np.array(im, dtype=np.float64)

if a.seams >= img.shape[1]:
    p.error(f"--seams must be less than image width ({img.shape[1]})")

t = time.perf_counter()
out = carve(img, a.seams)
elapsed = time.perf_counter() - t

Image.fromarray(out.astype(np.uint8)).save(a.output)
print(f"{img.shape[1]} -> {out.shape[1]} px, {a.seams} seams, {elapsed:.1f}s")