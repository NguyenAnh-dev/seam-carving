import time
import numpy as np
from PIL import Image
from seam.carve import carve

img = np.array(Image.open("photo.jpg").convert("RGB"), dtype=np.float64)
print("input:", img.shape)

t = time.perf_counter()
out = carve(img, 70)
elapsed = time.perf_counter() - t

print("output:", out.shape, f"{elapsed*1000:.1f}ms")
Image.fromarray(out.astype(np.uint8)).save("resized.jpg")