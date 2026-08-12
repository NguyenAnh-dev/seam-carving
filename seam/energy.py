import numpy as np


def energy(img):
    """img: (h, w, 3) float array -> (h, w) float cost grid."""
    gray = np.mean(img, axis=2)

    dx = np.abs(np.diff(gray, axis=1))
    dx = np.pad(dx, ((0, 0), (0, 1)))

    dy = np.abs(np.diff(gray, axis=0))
    dy = np.pad(dy, ((0, 1), (0, 0)))

    return dx + dy