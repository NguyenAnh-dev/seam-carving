import time
import numpy as np
from PIL import Image
from seam.energy import energy
from seam.carve import find_seam, find_seam_logic

img = np.array(Image.open("photo.jpg").convert("RGB"), dtype=np.float64)
cost = energy(img)
print("image:", img.shape)

t = time.perf_counter()
find_seam(cost)
fast = (time.perf_counter() - t) * 1000

t = time.perf_counter()
find_seam_logic(cost.tolist())
slow = (time.perf_counter() - t) * 1000

print(f"naive:      {slow:.1f}ms")
print(f"vectorized: {fast:.1f}ms")
print(f"speedup:    {slow/fast:.1f}x")