import gradio as gr
import numpy as np
from PIL import Image
from seam.carve import carve

def resize(image, seams):
    im = Image.fromarray(image).convert("RGB")
    im.thumbnail((600, 600))
    img = np.array(im, dtype=np.float64)
    seams = min(int(seams), img.shape[1] - 1)
    return carve(img, seams).astype(np.uint8)

gr.Interface(
    fn=resize,
    inputs=[gr.Image(), gr.Slider(10, 150, value=80, step=10, label="Seams to remove")],
    outputs=gr.Image(),
    title="Seam carving",
    description="Content-aware resizing. Removes low-energy vertical seams so the subject keeps its proportions.",
).launch()