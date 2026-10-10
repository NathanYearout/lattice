"""Procedural building blocks. Every function is vectorized and seeded, so a whole world is a few lines and the same seed
always gives the same world: ``land = gen.terrain(256, 128, seed=3)``, ``w.tree.spawn(len(p), pos=land.place(p))``."""

import numpy as np
import pygfx as gfx


def rgb(*hexes):
    """Hex colours as an (n, 3) float array (sRGB 0..1), ready for a ``tint`` column."""
    return np.array([gfx.Color(h).rgb for h in hexes], np.float32)


def palette(n, *hexes, seed=0, jitter=0.0):
    """``n`` tints drawn from the given colours, optionally jittered in brightness for a hand-made feel."""
    rng = np.random.default_rng(seed)
    return rgb(*hexes)[rng.integers(len(hexes), size=n)] * (1 + rng.uniform(-jitter, jitter, (n, 1))).astype(np.float32)


def noise(x, z, seed=0, octaves=4, scale=1.0):
    """Smooth fractal value noise in [-1, 1] at any array of points (fbm with smoothstep interpolation)."""
    x, z = np.asarray(x, float) / scale, np.asarray(z, float) / scale
    total, amp, norm = np.zeros(np.broadcast(x, z).shape), 1.0, 0.0
    for o in range(octaves):
        xi, zi = np.floor(x), np.floor(z)
        fx, fz = x - xi, z - zi
        fx, fz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
        h = lambda a, b: (np.sin(a * 127.1 + b * 311.7 + (seed + o * 17) * 74.7) * 43758.5453) % 1.0 * 2 - 1
        top = h(xi, zi) + (h(xi + 1, zi) - h(xi, zi)) * fx
        bottom = h(xi, zi + 1) + (h(xi + 1, zi + 1) - h(xi, zi + 1)) * fx
        total, norm, amp = total + (top + (bottom - top) * fz) * amp, norm + amp, amp * 0.5
        x, z = x * 2.03, z * 2.03
    return total / norm


class Terrain:
    """A heightfield mesh with vertex colours by height, plus ``height(x, z)`` / ``place(points)`` to stand things on it."""

    def __init__(self, heights, size, colors):
        self.heights, self.size = heights, size
        res = heights.shape[0]
        g = np.linspace(-size / 2, size / 2, res)
        x, z = np.meshgrid(g, g)
        pos = np.stack([x, heights, z], -1).reshape(-1, 3).astype(np.float32)
        i = np.arange(res * res).reshape(res, res)[:-1, :-1].ravel()
        quads = np.stack([i, i + res, i + 1, i + 1, i + res, i + res + 1], -1).reshape(-1, 3)
        self.geometry = gfx.Geometry(positions=pos, indices=quads.astype(np.uint32), colors=colors.reshape(-1, 4).astype(np.float32),
                                     texcoords=((pos[:, [0, 2]] / size) + 0.5).astype(np.float32))
        self.geometry.normals = gfx.Buffer(_normals(pos, quads))

    def height(self, x, z):
        res, h = self.heights.shape[0], self.heights
        u = np.clip((np.asarray(x) / self.size + 0.5) * (res - 1), 0, res - 1.001)
        v = np.clip((np.asarray(z) / self.size + 0.5) * (res - 1), 0, res - 1.001)
        i, j, fu, fv = v.astype(int), u.astype(int), u % 1, v % 1
        return (h[i, j] * (1 - fu) + h[i, j + 1] * fu) * (1 - fv) + (h[i + 1, j] * (1 - fu) + h[i + 1, j + 1] * fu) * fv

    def place(self, points, offset=0.0):
        """Drop (n, 2) xz or (n, 3) points onto the surface."""
        p = np.asarray(points, np.float32)
        x, z = p[:, 0], p[:, -1]
        return np.c_[x, self.height(x, z) + offset, z].astype(np.float32)


def _normals(pos, faces):
    n = np.zeros_like(pos)
    fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
    for c in range(3):
        np.add.at(n, faces[:, c], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return (n * np.sign(n[:, 1:2] + 1e-9)).astype(np.float32)  # face up


def terrain(size=200.0, res=128, height=18.0, seed=0, scale=60.0, ramp=("#2b4a2f", "#4f7a3a", "#8a8a5c", "#d8d8d0"), flat=None):
    """Rolling hills from fractal noise, coloured by a height ramp (low to high). ``flat(x, z)`` may return 0..1 to flatten
    areas (roads, plazas, a city footprint)."""
    g = np.linspace(-size / 2, size / 2, res)
    x, z = np.meshgrid(g, g)
    h = noise(x, z, seed, 5, scale) * height
    h = h * (1 - flat(x, z)) if flat else h
    t = np.clip((h - h.min()) / (np.ptp(h) + 1e-9), 0, 1) * (len(ramp) - 1)
    c = rgb(*ramp) ** 2.2  # vertex colours are linear
    i = np.minimum(t.astype(int), len(ramp) - 2)
    col = c[i] + (c[i + 1] - c[i]) * (t - i)[..., None]
    return Terrain(h, size, np.concatenate([col, np.ones((*h.shape, 1))], -1))


def scatter(n, size, seed=0, where=None, min_dist=0.0):
    """``n`` random xz points in a ``size`` square (centred), kept where ``where(x, z)`` is true, thinned to ``min_dist``."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-size / 2, size / 2, (n * 3 if where else n, 2))
    p = p[where(p[:, 0], p[:, 1])] if where else p
    if min_dist:  # grid-hash thinning: keep one point per cell
        _, keep = np.unique(np.floor(p / min_dist).astype(int), axis=0, return_index=True)
        p = p[np.sort(keep)]
    return p[:n].astype(np.float32)


def lots(blocks=8, block=40.0, street=10.0, per_side=4, seed=0, jitter=0.15):
    """A street grid of city blocks split into building lots. Returns (centres (n, 2), footprints (n, 2))."""
    rng = np.random.default_rng(seed)
    cell = block / per_side
    b = (np.arange(blocks) - (blocks - 1) / 2) * (block + street)
    l = (np.arange(per_side) - (per_side - 1) / 2) * cell
    bx, bz, lx, lz = (a.ravel() for a in np.meshgrid(b, b, l, l, indexing="ij"))
    centres = np.c_[bx + lx, bz + lz] + rng.uniform(-jitter, jitter, (len(bx), 2)) * cell
    return centres.astype(np.float32), (cell * rng.uniform(0.6, 0.92, (len(bx), 2))).astype(np.float32)
