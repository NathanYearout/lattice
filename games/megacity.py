"""Neo-Tokyo: a rain-soaked neon megacity at 2 a.m. Thousands of towers, signs, cars and raindrops; a dozen draw calls.

Drag with the left mouse button to look around · W/S to fly in and out · A/D to circle · Q/E to rise and fall.
"""

import numpy as np
import pygfx as gfx

from lattice import fx, gen

SEED, BLOCKS, BLOCK, STREET = 7, 14, 44.0, 12.0
NEON = ("#ff2a6d", "#05d9e8", "#ffd319", "#ff6c11", "#c23bff", "#39ff9f")
FLOORS = (14, 28, 48, 80, 130, 200)  # one kind per height band, so window textures never stretch far
SPAN = BLOCKS * (BLOCK + STREET) / 2


def windows(seed=SEED):
    """A 16x32 tile of window cells, randomly lit warm, cool or dark: the emissive map that makes towers glow."""
    rng = np.random.default_rng(seed)
    lit = rng.random((32, 16)) < 0.55
    hue = gen.rgb("#ffd9a0", "#bfe4ff", "#ffffff", "#ff9ad5")[rng.integers(4, size=(32, 16))] * rng.uniform(0.35, 1, (32, 16, 1))
    cell = np.zeros((32, 16, 4, 4, 3), np.float32)
    cell[:, :, 1:4, 0:3] = (hue * lit[..., None])[:, :, None, None]
    img = cell.transpose(0, 2, 1, 3, 4).reshape(128, 64, 3)
    return (np.dstack([img, np.ones(img.shape[:2])]) * 255).astype(np.uint8)


def tower_geometry(height, seed):
    """A box whose side UVs repeat once per 3.5 m floor; roofs sample a dark texel."""
    g = gfx.box_geometry(1, height, 1)
    uv, n = g.texcoords.data, g.normals.data
    side = np.abs(n[:, 1]) < 0.5
    uv[side] *= (12 / 16, height / 3.5 / 32)
    uv[side] += np.random.default_rng(seed).random(2)
    uv[~side] = 0
    g.positions.data[:, 1] += height / 2  # pivot at the base
    return g


def scene(w):
    fx.retro(w.renderer, 0.45)
    w.renderer.effect_passes = [fx.bloom(0.09), fx.fog("#3a1450", 260), fx.grade(1.25, 1.08, "#ffe8ff"), fx.chromatic(0.6),
                                fx.vignette(0.7), fx.noise(0.03)]
    w.background.material = gfx.BackgroundMaterial("#ff3d7f", "#2a0f4a", "#05040d", "#05040d")
    w.sun.intensity, w.sun.cast_shadow, w.ambient.intensity = 0.25, False, 0.12
    w.sun.color, w.ambient.color = "#6a7dff", "#4b3aff"
    w.camera.fov, w.camera.depth_range = 55, (1, 4000)

    glass = w.texture(windows(), filter="nearest")
    for i, h in enumerate(FLOORS):
        w.kind(f"tower{i}", tower_geometry(h, i), material=w.material("standard", color="#20222e", emissive="#ffffff", emissive_map=glass,
                                                                      emissive_intensity=4.5, roughness=0.4, metalness=0.5))
    w.kind("ground", gfx.box_geometry(), "#0c0c14", material="phong")
    w.kind("road", gfx.box_geometry(), material=w.material("phong", color="#15151f", specular="#6a5a8a", shininess=90))
    w.kind("sign", gfx.box_geometry(), material=w.material("basic", color="#ffffff"), phase=0.0)
    w.kind("beacon", gfx.sphere_geometry(0.9, 8, 6), material=w.material("basic", color="#ffffff"), phase=0.0)
    w.kind("car", gfx.box_geometry(1.8, 0.5, 0.9), material=w.material("basic", color="#ffffff"), speed=0.0, axis=0.0)
    w.kind("rain", gfx.box_geometry(0.04, 1.6, 0.04), material=w.material("basic", color="#b9c8ff", opacity=0.35))
    w.kind("tower_spire", gfx.cone_geometry(1, 1, 4), material=w.material("basic", color="#ffffff"))
    w.land = gen.terrain(3600, 160, 260, SEED, 500, ("#07060f", "#120d26", "#1d1638", "#5b4a7a"),
                         flat=lambda x, z: np.clip(1.6 - np.hypot(x, z) / (SPAN * 1.1), 0, 1))
    w.kind("mountains", w.land.geometry, material=w.material("phong", color_mode="vertex", shininess=5))


def setup(w):
    rng = np.random.default_rng(SEED)
    w.mountains.spawn(pos=(0, -6, 0))
    w.ground.spawn(pos=(0, -0.5, 0), scale=(SPAN * 2.4, 1, SPAN * 2.4))

    # towers: tallest in the core, a lognormal skyline everywhere else
    centre, size = gen.lots(BLOCKS, BLOCK, STREET, 4, SEED)
    core = np.exp(-(np.hypot(*centre.T) / (SPAN * 0.45)) ** 2)
    height = np.clip(rng.lognormal(3.0, 0.55, len(centre)) * (0.6 + 2.4 * core) + 6 * gen.noise(*centre.T, SEED, 2, 120), 8, 260)
    band = np.digitize(height, (np.array(FLOORS[:-1]) + np.array(FLOORS[1:])) / 2)
    body = gen.palette(len(centre), "#ffd9b0", "#cfe6ff", "#ffffff", "#ffb3e6", "#b3fff0", seed=SEED, jitter=0.25)
    for i, h in enumerate(FLOORS):
        m = band == i
        w.kinds[f"tower{i}"].spawn(int(m.sum()), pos=np.c_[centre[m, 0], np.zeros(m.sum()), centre[m, 1]],
                                    scale=np.c_[size[m, 0], height[m] / h, size[m, 1]], tint=body[m])
    w.tops = np.c_[centre[:, 0], height, centre[:, 1]]

    # roads between blocks
    lines = (np.arange(BLOCKS + 1) - BLOCKS / 2) * (BLOCK + STREET)
    for axis in (0, 1):
        pos = np.zeros((len(lines), 3))
        pos[:, 2 - 2 * axis], pos[:, 1] = lines, 0.02
        w.road.spawn(len(lines), pos=pos, scale=(SPAN * 2, 0.04, STREET) if axis == 0 else (STREET, 0.04, SPAN * 2))

    # neon: vertical blade signs on facades, more of them low down where the streets are
    n = 2600
    which = rng.integers(len(centre), size=n)
    face = rng.integers(4, size=n)
    out = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)[face]
    y = rng.uniform(0.1, 1, n) * np.minimum(height[which] * 0.8, 40) + 3
    tall = rng.random(n) < 0.6
    sx, sy = np.where(tall, 0.5, rng.uniform(4, 9, n)), np.where(tall, rng.uniform(5, 14, n), rng.uniform(1.2, 2.5, n))
    off = out * (size[which] / 2 + 0.6)
    w.sign.spawn(n, pos=np.c_[centre[which, 0] + off[:, 0], y, centre[which, 1] + off[:, 1]],
                 scale=np.c_[np.where(face < 2, 0.3, sx), sy, np.where(face < 2, sx, 0.3)],
                 tint=gen.palette(n, *NEON, seed=SEED + 1) * 1.6, phase=rng.uniform(0, 100, n))

    # aircraft warning lights on every tower over 60 m, and a red lattice spire as the landmark
    tall = w.tops[w.tops[:, 1] > 60]
    w.beacon.spawn(len(tall), pos=tall + (0, 1.2, 0), tint=(1.6, 0.1, 0.1), phase=rng.uniform(0, 6, len(tall)))
    for k, (r, h, y) in enumerate(((18, 120, 0), (10, 110, 110), (5, 90, 210))):
        w.tower_spire.spawn(pos=(BLOCK + STREET / 2, y + h / 2, -(BLOCK + STREET) * 2.5), scale=(r, r, h),
                            rot=(-0.7071, 0, 0, 0.7071), tint=(1.8, 0.35, 0.12) if k % 2 == 0 else (1.6, 1.4, 1.3))

    # traffic: rivers of headlights one way, tail lights the other
    n = 2400
    axis, line = rng.integers(2, size=n), lines[rng.integers(len(lines), size=n)]
    lane = rng.choice((-1, 1), n) * rng.uniform(1.2, 4.5, n)
    t = rng.uniform(-SPAN, SPAN, n)
    pos = np.where(axis[:, None] == 0, np.c_[t, np.full(n, 0.6), line + lane], np.c_[line + lane, np.full(n, 0.6), t])
    w.car.spawn(n, pos=pos, axis=axis, speed=np.sign(lane) * rng.uniform(14, 26, n),
                rot=np.where(axis[:, None] == 0, (0, 0, 0, 1), (0, 0.7071, 0, 0.7071)),
                tint=np.where(lane[:, None] > 0, (1.8, 1.6, 1.3), (1.8, 0.08, 0.05)))

    w.rain.spawn(5000, pos=rng.uniform((-70, 0, -70), (70, 140, 70), (5000, 3)))
    w.yaw, w.pitch, w.radius, w.height = 0.6, -0.12, SPAN * 1.25, 120.0
    update(w, dt=0)


def drone(w, dt):
    """A slow cinematic orbit you can steer."""
    keys = lambda a, b: (a in w.keys) - (b in w.keys)
    w.yaw += (0.03 + 0.6 * keys("d", "a")) * dt
    if 1 in w.buttons:
        w.yaw -= w.mouse_delta[0] * 0.004
        w.pitch = float(np.clip(w.pitch - w.mouse_delta[1] * 0.003, -0.8, 0.3))
    w.radius = float(np.clip(w.radius * (1 - 0.8 * keys("w", "s") * dt), 60, SPAN * 2.2))
    w.height = float(np.clip(w.height + 60 * keys("e", "q") * dt, 8, 600))
    eye = np.array([np.sin(w.yaw) * w.radius, w.height, np.cos(w.yaw) * w.radius])
    w.camera.local.position = eye
    w.camera.look_at(eye + (-np.sin(w.yaw), np.tan(w.pitch) - eye[1] / w.radius * 0.4, -np.cos(w.yaw)))


def update(w, dt=None):
    dt = w.dt if dt is None else dt
    drone(w, dt)

    c = w.car  # traffic along both street axes, wrapping at the city edge
    along = c.axis.astype(int) * 2
    c.pos[np.arange(c.n), along] = (c.pos[np.arange(c.n), along] + c.speed * dt + SPAN) % (2 * SPAN) - SPAN

    s = w.sign  # a few signs buzz and stutter like failing tubes
    s.tint = gen.palette(s.n, *NEON, seed=SEED + 1) * 1.6 * np.where(np.sin(w.t * 23 + s.phase * 7) > 0.97, 0.15, 1.0)[:, None]
    b = w.beacon
    b.tint = np.c_[(np.sin(w.t * 2.2 + b.phase) > 0.6) * 1.8 + 0.08, np.full(b.n, 0.05), np.full(b.n, 0.05)]

    r, eye = w.rain, w.camera.local.position  # rain lives in a box that follows the camera
    r.pos[:, 1] -= 70 * dt
    r.pos[:, [0, 2]] = (r.pos[:, [0, 2]] - eye[[0, 2]] + 70) % 140 - 70 + eye[[0, 2]]
    r.pos[:, 1] = (r.pos[:, 1] - eye[1] + 70) % 140 - 70 + eye[1]

    clock = 2 * 60 + 13 + int(w.t / 4)
    w.hud = f"**NEO-TOKYO**  ·  {clock // 60:02d}:{clock % 60:02d}  ·  {sum(map(len, w.kinds.values())):,} lights in the rain"
    w.play(RAIN, volume=0.35, loop=True, key="rain")


def _rain_sound(seconds=4.0, rate=44100):
    """Brown-ish noise with a soft low hum: rain on concrete under a city that never sleeps. Loops seamlessly."""
    rng = np.random.default_rng(1)
    white = rng.normal(0, 1, (int(seconds * rate), 2))
    spectrum = np.fft.rfft(white, axis=0) / np.sqrt(np.arange(1, white.shape[0] // 2 + 2))[:, None] ** 0.8
    noise = np.fft.irfft(spectrum, n=white.shape[0], axis=0)
    t = np.arange(white.shape[0]) / rate
    hum = 0.15 * np.sin(2 * np.pi * 55 * t)[:, None]
    return (noise / np.abs(noise).max() * 0.6 + hum) * 0.5


RAIN = _rain_sound()
