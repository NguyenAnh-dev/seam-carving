from PIL import Image
import numpy as np

original = Image.open("photo.jpg")
carved = Image.open("out.jpg")
squashed = original.resize((carved.width, original.height))

im = Image.open("photo.jpg").convert("RGB")
im.thumbnail((500, 500))
img = np.array(im, dtype=np.float64)
gap = 10
total_w = original.width + squashed.width + carved.width + gap * 2
canvas = Image.new("RGB", (total_w, original.height), "white")

canvas.paste(original, (0, 0))
canvas.paste(squashed, (original.width + gap, 0))
canvas.paste(carved, (original.width + squashed.width + gap * 2, 0))
canvas.save("comparison.jpg")
print("saved comparison.jpg")
print(original.size, squashed.size, carved.size)