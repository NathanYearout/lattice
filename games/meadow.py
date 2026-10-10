"""Meadow: a lush valley grown from one seed. Change STYLE to restyle the whole game in one word (see fx.STYLES).

Rolling hills, a pond, a village, 4,000 trees, 12,000 flowers and drifting clouds, in about 60 lines and 10 draw calls.
"""

import numpy as np
import pygfx as gfx

from lattice import fx, gen

SEED, SIZE, STYLE = 4, 320.0, "cel"  # realistic · source · cel · ps2 · ps1 · n64 · vhs · noir · gameboy · synthwave
POND = (30.0, 40.0, 26.0)  # x, z, radius


def scene(w):
    w.background.material = gfx.BackgroundMaterial("#fff3d6", "#7fb6ff")
    w.sun.intensity, w.ambient.intensity, w.sun.color = 3.2, 0.55, "#fff1d8"
    w.sun.local.position = (60, 90, 30)
    w.sun.shadow.camera.width = w.sun.shadow.camera.height = 260
    w.camera.fov, w.camera.depth_range = 50, (0.5, 1500)
    pond = lambda x, z: np.clip(1.4 - np.hypot(x - POND[0], z - POND[1]) / POND[2], 0, 1)
    w.land = gen.terrain(SIZE, 160, 22, SEED, 90, ("#3f7d34", "#6aa84f", "#a5c262", "#c9b98a"), flat=pond)
    w.kind("ground", w.land.geometry, material=w.material("standard", color_mode="vertex", roughness=1))
    w.kind("water", gfx.cylinder_geometry(1, 1, 0.2, 48), material=w.material("standard", color="#3d8fd6", roughness=0.1, opacity=0.85))
    w.kind("trunk", gfx.cylinder_geometry(0.25, 0.35, 2.4, 6), "#6b4a2f")
    w.kind("crown", gfx.cone_geometry(1.6, 4.5, 7), "#ffffff", sway=0.0)
    w.kind("bush", gfx.sphere_geometry(1, 8, 6), "#ffffff")
    w.kind("flower", gfx.sphere_geometry(0.18, 5, 4), "#ffffff")
    w.kind("house", gfx.box_geometry(), "#f2e6d0")
    w.kind("roof", gfx.cone_geometry(1, 1, 4), "#c0503a")
    w.kind("cloud", gfx.sphere_geometry(1, 10, 8), material=w.material("basic", color="#ffffff", opacity=0.95), shadow=False,
           drift=0.0)
    fx.style(w, STYLE)


def setup(w):
    rng, up = np.random.default_rng(SEED), (-0.7071, 0, 0, 0.7071)  # pygfx builds cones/cylinders along +z: stand them up
    w.ground.spawn()
    w.water.spawn(pos=(POND[0], w.land.height(*POND[:2]) + 0.3, POND[1]), scale=(POND[2] * 0.75, POND[2] * 0.75, 1), rot=up)
    dry = lambda x, z: np.hypot(x - POND[0], z - POND[1]) > POND[2] * 0.9

    trees = w.land.place(gen.scatter(4000, SIZE * 0.95, SEED, lambda x, z: dry(x, z) & (gen.noise(x, z, SEED + 9, 3, 50) > -0.05), 2.2))
    size = rng.uniform(0.7, 1.5, (len(trees), 1))
    w.trunk.spawn(len(trees), pos=trees + (0, 1.1, 0) * size, scale=np.c_[size, size, size], rot=up)
    w.crown.spawn(len(trees), pos=trees + (0, 4.2, 0) * size, scale=np.c_[size, size, size * rng.uniform(0.9, 1.3, (len(trees), 1))],
                  rot=up, tint=gen.palette(len(trees), "#2f6b2a", "#3d8a36", "#5a9e3a", "#2b5e3b", seed=SEED, jitter=0.15),
                  sway=rng.uniform(0, 6.3, len(trees)))
    bushes = w.land.place(gen.scatter(1500, SIZE * 0.9, SEED + 1, dry))
    w.bush.spawn(len(bushes), pos=bushes, scale=rng.uniform(0.6, 1.4, (len(bushes), 1)) * (1.2, 0.8, 1.2),
                 tint=gen.palette(len(bushes), "#4f8f3a", "#6fae45", seed=SEED, jitter=0.2))
    flowers = w.land.place(gen.scatter(12000, SIZE * 0.9, SEED + 2, lambda x, z: dry(x, z) & (gen.noise(x, z, SEED + 3, 2, 25) > 0.2)), 0.2)
    w.flower.spawn(len(flowers), pos=flowers, tint=gen.palette(len(flowers), "#ffffff", "#ffd23f", "#ff6fb5", "#b18cff", seed=SEED))

    homes = w.land.place(gen.scatter(18, 70, SEED + 4, lambda x, z: np.hypot(x + 40, z + 30) < 36, 9) - (40, 30))
    s = rng.uniform(4, 6.5, (len(homes), 1))
    w.house.spawn(len(homes), pos=homes + (0, 1.5, 0) * s / 2, scale=s * (1, 0.6, 0.8))
    w.roof.spawn(len(homes), pos=homes + (0, 1.0, 0) * s * 0.9, scale=s * (0.85, 0.85, 0.5), rot=(-0.6533, 0.2706, 0.2706, 0.6533),
                 tint=gen.palette(len(homes), "#c0503a", "#4a6fa5", "#8a5a44", seed=SEED))

    puffs = np.repeat(rng.uniform((-SIZE / 2, 70, -SIZE / 2), (SIZE / 2, 95, SIZE / 2), (22, 3)), 6, axis=0)
    w.cloud.spawn(len(puffs), pos=puffs + rng.normal(0, (6, 1.5, 4), puffs.shape), scale=rng.uniform(4, 9, (len(puffs), 1)) * (1.4, 0.7, 1),
                  drift=np.repeat(rng.uniform(2, 5, 22), 6))
    w.angle = 0.9
    update(w, 0.0)


def update(w, dt=None):
    dt = w.dt if dt is None else dt
    w.angle += dt * 0.05
    eye = np.array([np.cos(w.angle) * 150, 75, np.sin(w.angle) * 150])
    w.camera.local.position = eye
    w.camera.look_at((0, 4, 0))
    c = w.crown  # wind: every tree leans a little, out of phase
    lean = 0.04 * np.sin(w.t * 1.3 + c.sway)
    c.rot = np.c_[np.full(c.n, -0.7071), np.zeros(c.n), lean, np.full(c.n, 0.7071)]
    cl = w.cloud
    cl.pos[:, 0] = (cl.pos[:, 0] + cl.drift * dt + SIZE / 2) % SIZE - SIZE / 2
