"""Galaxy: 30,000 tumbling, orbiting cubes, simulated by a few lines of NumPy per frame.

Drag to orbit, scroll to zoom. Try editing ARMS, TWIST or a colour and hit save. The galaxy keeps spinning.
"""

import numpy as np
import pygfx as gfx

ARMS, TWIST, RADIUS = 3, 2.6, 40.0


def glow(color, strength=0.6):
    return gfx.MeshStandardMaterial(color=color, emissive=color, emissive_intensity=strength, roughness=0.4)


def scene(w):
    w.background.material = gfx.BackgroundMaterial("#020208", "#0b0620")
    w.sun.cast_shadow = False
    columns = dict(r=0.0, a=0.0, axis=(0.0, 1, 0), spin=1.0)
    w.kind("core", gfx.box_geometry(), material=glow("#ffcf7a", 1.2), **columns)
    w.kind("arm", gfx.box_geometry(), material=glow("#6fb7ff"), **columns)
    w.kind("dust", gfx.octahedron_geometry(), material=glow("#ff6fb5", 0.9), **columns)


def setup(w):
    rng = np.random.default_rng(7)
    for name, n, size in (("core", 4_000, 0.35), ("arm", 18_000, 0.25), ("dust", 8_000, 0.18)):
        r = rng.exponential(5.0, n) if name == "core" else rng.uniform(4, RADIUS, n)
        arm = rng.integers(0, ARMS, n) * 2 * np.pi / ARMS
        a = arm + TWIST * np.log1p(r) + rng.normal(0, 0.35 if name != "dust" else 0.6, n)
        axis = rng.normal(size=(n, 3))
        w.kinds[name].spawn(n, r=r, a=a, axis=axis / np.linalg.norm(axis, axis=1, keepdims=True),
                            spin=rng.uniform(-3, 3, n), scale=size * rng.uniform(0.5, 1.5, (n, 1)),
                            pos=np.c_[np.zeros(n), rng.normal(0, 0.6 + 2 / (1 + r / 4), n), np.zeros(n)])
    w.camera.local.position = (0, 38, 62)
    w.camera.look_at((0, 0, 0))
    w.orbit()
    update(w)  # place everything now, so the galaxy looks right in the editor before pressing Play


def update(w):
    for k in w.kinds.values():
        k.a += w.dt * 2.5 / np.sqrt(k.r + 1)  # inner stars orbit faster: the arms wind up
        k.pos[:, 0], k.pos[:, 2] = k.r * np.cos(k.a), k.r * np.sin(k.a)
        half = (w.t * k.spin / 2)[:, None]
        k.rot = np.c_[k.axis * np.sin(half), np.cos(half)]
    w.hud = f"**{sum(map(len, w.kinds.values())):,} entities** · {len(w.kinds)} draw calls"
