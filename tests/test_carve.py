import numpy as np
import pytest
from seam.energy import energy
from seam.carve import find_seam, find_seam_logic, remove_seam, carve


def cost(grid):
    return np.array(grid, dtype=np.float64)


def test_diagonal():
    g = cost([[1, 9, 9],
              [9, 1, 9],
              [9, 9, 1]])
    assert list(find_seam(g)) == [0, 1, 2]


def test_straight_column():
    g = cost([[9, 1, 9]] * 3)
    assert list(find_seam(g)) == [1, 1, 1]


def test_zigzag():
    g = cost([[9, 1, 9],
              [1, 9, 9],
              [9, 1, 9]])
    assert list(find_seam(g)) == [1, 0, 1]


def test_single_column():
    assert list(find_seam(cost([[5], [7]]))) == [0, 0]


def test_single_row():
    assert list(find_seam(cost([[4, 2, 8]]))) == [1]


def test_right_edge():
    g = cost([[9, 9, 1]] * 3)
    assert list(find_seam(g)) == [2, 2, 2]


def test_seam_is_connected():
    rng = np.random.default_rng(0)
    g = rng.random((40, 30))
    s = find_seam(g)
    assert len(s) == 40
    assert all(0 <= c < 30 for c in s)
    assert all(abs(int(s[i]) - int(s[i - 1])) <= 1 for i in range(1, 40))


def test_matches_naive():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 50, size=(20, 15)).astype(np.float64)
    assert list(find_seam(g)) == find_seam_logic(g.tolist())


def test_remove_seam_shape():
    img = np.zeros((10, 8, 3))
    out = remove_seam(img, np.zeros(10, dtype=np.int32))
    assert out.shape == (10, 7, 3)


def test_remove_seam_removes_right_pixels():
    img = np.zeros((3, 4, 3))
    img[:, 2] = 255
    out = remove_seam(img, np.full(3, 2, dtype=np.int32))
    assert not (out == 255).any()


def test_carve_reduces_width():
    img = np.random.default_rng(2).random((30, 25, 3)) * 255
    assert carve(img, 5).shape == (30, 20, 3)


def test_energy_shape():
    img = np.random.default_rng(3).random((12, 9, 3)) * 255
    assert energy(img).shape == (12, 9)