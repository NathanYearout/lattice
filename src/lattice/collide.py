"""Collision queries over whole kinds at once. Every instance is an oriented box (its geometry's bounds, posed by the
instance's pos/rot/scale) or a sphere. scipy's KD-trees find candidate pairs; NumPy does the exact tests in bulk."""

from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree


class Hit(NamedTuple):
    kind: str
    index: int
    dist: float
    point: np.ndarray
    normal: np.ndarray


def rotations(q):
    """(n, 4) xyzw quaternions -> (n, 3, 3) rotation matrices."""
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def boxes(k):
    """World centres, rotations and half extents of every instance's oriented box."""
    lo, hi = k.bounds
    r = rotations(k.rot)
    return k.pos + np.einsum("nij,nj->ni", r, (lo + hi) / 2 * k.scale), r, np.abs((hi - lo) / 2 * k.scale)


def radii(k, radius=None):
    """Per-instance sphere radius: given (scalar or array), or the largest half extent of its box."""
    return np.broadcast_to(np.asarray(radius, float), (k.n,)) if radius is not None else boxes(k)[2].max(1)


def raycast(kinds, origin, direction, max_dist=np.inf):
    """Nearest oriented-box hit along a ray across ``kinds`` (name -> Kind), or None."""
    o, d = np.asarray(origin, float), np.asarray(direction, float) / np.linalg.norm(direction)
    best = None
    for name, k in kinds.items():
        if not k.n:
            continue
        lo, hi = k.bounds
        r, s = rotations(k.rot), np.where(np.abs(k.scale) < 1e-9, 1e-9, k.scale)
        ol = np.einsum("nji,nj->ni", r, o - k.pos) / s  # the ray in each instance's unscaled local space
        dl = np.einsum("nji,j->ni", r, d) / s
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - ol) / dl, (hi - ol) / dl
        near, far = np.nanmax(np.minimum(t1, t2), 1), np.nanmin(np.maximum(t1, t2), 1)
        t = np.where(near > 0, near, 0.0)
        ok = (far >= t) & (t <= max_dist)
        if ok.any() and (best is None or t[ok].min() < best.dist):
            i = int(np.flatnonzero(ok)[t[ok].argmin()])
            axis = int(np.nanargmax(np.minimum(t1[i], t2[i])))
            n = np.zeros(3)
            n[axis] = -np.sign(dl[i, axis]) or 1.0
            n = r[i] @ (n / s[i])
            best = Hit(name, i, float(t[i]), o + d * t[i], n / np.linalg.norm(n))
    return best


def overlaps(a, b, ra=None, rb=None):
    """Index pairs (i, j) where sphere i of kind ``a`` touches sphere j of kind ``b``. With a is b, each pair once."""
    empty = np.zeros(0, int)
    if not a.n or not b.n:
        return empty, empty
    ra, rb = radii(a, ra), radii(b, rb)
    m = cKDTree(a.pos).sparse_distance_matrix(cKDTree(b.pos), ra.max() + rb.max(), output_type="ndarray")
    i, j, dist = m["i"], m["j"], m["v"]
    keep = dist < ra[i] + rb[j]
    keep &= (i < j) if a is b else True
    return i[keep].astype(int), j[keep].astype(int)


def push_out(points, radius, walls, iterations=2):
    """Slide spheres out of every oriented box of ``walls``. Returns (new points, contact normals: zero where free)."""
    p, normal = np.array(points, float).reshape(-1, 3), np.zeros((len(points), 3))
    radius = np.broadcast_to(np.asarray(radius, float), (len(p),))
    if not walls.n or not len(p):
        return p, normal
    c, r, h = boxes(walls)
    tree, reach = cKDTree(c), np.linalg.norm(h, axis=1)
    for _ in range(iterations):
        m = cKDTree(p).sparse_distance_matrix(tree, radius.max() + reach.max(), output_type="ndarray")
        near = m["v"] < radius[m["i"]] + reach[m["j"]]
        i, j = m["i"][near], m["j"][near]
        local = np.einsum("nji,nj->ni", r[j], p[i] - c[j])
        closest = np.clip(local, -h[j], h[j])
        away = local - closest
        dist = np.linalg.norm(away, axis=1)
        inside = dist < 1e-9  # centre inside the box: leave along the axis of least penetration
        axis = np.argmin(h[j] - np.abs(local), 1)
        exit = np.zeros_like(local)
        exit[np.arange(len(i)), axis] = np.where(local[np.arange(len(i)), axis] < 0, -1, 1)
        direction = np.where(inside[:, None], exit, away / np.maximum(dist, 1e-9)[:, None])
        depth = np.where(inside, h[j][np.arange(len(i)), axis] - np.abs(local[np.arange(len(i)), axis]), 0) + radius[i] - dist
        hit = depth > 0
        push = np.einsum("nij,nj->ni", r[j], direction * np.maximum(depth, 0)[:, None])
        if not hit.any():
            break
        np.add.at(p, i[hit], push[hit])
        np.add.at(normal, i[hit], push[hit])
    return p, normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
