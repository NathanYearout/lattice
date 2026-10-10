"""Fluorescent: a PS2-era liminal horror test game.

You are somewhere in an office that doesn't end. The lights hum. Find the EXIT.
Something is in here with you. It only moves when you aren't looking.

Click to capture the mouse · WASD to walk · mouse / arrow keys to look · Shift to run · Esc to free the mouse · F5 to restart
"""

import numpy as np
import pygfx as gfx
import pylinalg as la

GRID, CELL, HEIGHT, RADIUS = 16, 4.0, 3.0, 0.3
FOG, SEED = "#1e1b0f", 13
HALF = GRID * CELL / 2


def solid(color, **kw):
    return gfx.MeshStandardMaterial(color=color, roughness=1.0, **kw)


def scene(w):
    r = w.renderer  # the PS2 look: low internal resolution, nearest upscale, depth fog, film grain
    r.pixel_ratio, r.pixel_filter, r.ppaa = 0.3, "nearest", "none"
    fog = gfx.Color(*gfx.Color(FOG).to_physical())  # effect passes work in linear colour
    r.effect_passes = [gfx.renderers.wgpu.FogPass(fog, power=28), gfx.renderers.wgpu.NoisePass(0.04)]
    w.text.font_size = 40
    w.sun.intensity, w.sun.cast_shadow, w.ambient.intensity = 0.0, False, 0.3
    w.camera.fov, w.camera.depth_range = 72, (0.05, 80)
    w.kind("floor", gfx.box_geometry(), material=solid("#5c4f2a"))
    w.kind("ceiling", gfx.box_geometry(), material=solid("#cfc59a"))
    w.kind("wall", gfx.box_geometry(), material=solid("#c2b072"))
    w.kind("lamp", gfx.box_geometry(), material=gfx.MeshBasicMaterial(color="#fff8d6"), phase=0.0, dead=0.0)
    w.kind("socket", gfx.box_geometry(), material=solid("#3a3628"))
    w.kind("door", gfx.box_geometry(), material=solid("#2c2418"))
    w.kind("sign", gfx.box_geometry(), material=gfx.MeshBasicMaterial(color="#3dff6a"))
    w.kind("it", gfx.cylinder_geometry(0.22, 0.12, 2.3, radial_segments=6), material=gfx.MeshBasicMaterial(color="#050403"))
    if not hasattr(w, "torch"):
        w.torch = gfx.PointLight("#ffe7a3", intensity=6, distance=14, decay=1.6)
        w.scene.add(w.torch)


def setup(w):
    rng = np.random.default_rng(SEED)
    cells = (np.indices((GRID, GRID)).reshape(2, -1).T + 0.5) * CELL - HALF  # cell centres (x, z)
    w.floor.spawn(pos=(0, -0.05, 0), scale=(GRID * CELL, 0.1, GRID * CELL))
    w.ceiling.spawn(pos=(0, HEIGHT + 0.05, 0), scale=(GRID * CELL, 0.1, GRID * CELL))

    # Walls: each cell randomly closes its east / south edge (with doorway gaps), plus the outer boundary.
    east, south = rng.random(len(cells)) < 0.42, rng.random(len(cells)) < 0.42
    span = CELL * rng.choice([1.0, 0.55], len(cells))  # some walls are short: openings & awkward nooks
    for mask, offset, size in ((east, (CELL / 2, 0), lambda s: np.c_[np.full_like(s, 0.2), np.full_like(s, HEIGHT), s]),
                               (south, (0, CELL / 2), lambda s: np.c_[s, np.full_like(s, HEIGHT), np.full_like(s, 0.2)])):
        xz = cells[mask] + offset
        w.wall.spawn(mask.sum(), pos=np.c_[xz[:, 0], np.full(mask.sum(), HEIGHT / 2), xz[:, 1]], scale=size(span[mask]))
    for x, z, sx, sz in ((HALF, 0, 0.2, 2 * HALF), (-HALF, 0, 0.2, 2 * HALF), (0, HALF, 2 * HALF, 0.2), (0, -HALF, 2 * HALF, 0.2)):
        w.wall.spawn(pos=(x, HEIGHT / 2, z), scale=(sx, HEIGHT, sz))
    pillars = cells[rng.random(len(cells)) < 0.18] + CELL / 2
    w.wall.spawn(len(pillars), pos=np.c_[pillars[:, 0], np.full(len(pillars), HEIGHT / 2), pillars[:, 1]], scale=(0.6, HEIGHT, 0.6))

    # Ceiling panels: one per cell, a few dead, all of them flickering a little.
    lamp_pos = np.c_[cells[:, 0], np.full(len(cells), HEIGHT - 0.01), cells[:, 1]]
    w.socket.spawn(len(cells), pos=lamp_pos + (0, 0.005, 0), scale=(1.3, 0.02, 0.7))
    w.lamp.spawn(len(cells), pos=lamp_pos, scale=(1.2, 0.02, 0.6), phase=rng.uniform(0, 100, len(cells)),
                 dead=(rng.random(len(cells)) < 0.22).astype(np.float32))

    # The exit sits in the far corner; you start in the opposite one. It starts somewhere in between.
    w.door.spawn(pos=(HALF - 0.12, 1.1, HALF - CELL / 2), scale=(0.1, 2.2, 1.1))
    w.sign.spawn(pos=(HALF - 0.15, 2.45, HALF - CELL / 2), scale=(0.06, 0.25, 0.7))
    w.it.spawn(pos=(rng.uniform(-6, 6), 1.15, rng.uniform(-6, 6)), rot=(0.7071, 0, 0, 0.7071))  # cylinders are built along z
    w.me, w.yaw, w.pitch, w.state, w.clock = np.array([-HALF + CELL / 2, 1.6, -HALF + CELL / 2]), -2.36, 0.0, "alive", 0.0


def look(w):
    dx, dy = w.mouse_delta
    keys = lambda a, b: (a in w.keys) - (b in w.keys)
    w.yaw -= dx * 0.0025 + keys("right", "left") * 2.2 * w.dt
    w.pitch = np.clip(w.pitch - dy * 0.0025 + keys("up", "down") * 1.5 * w.dt, -1.3, 1.3)
    if w.clicked:
        w.capture_mouse()
    if "escape" in w.pressed:
        w.capture_mouse(False)
    return np.array([-np.sin(w.yaw), 0, -np.cos(w.yaw)]), np.array([np.cos(w.yaw), 0, -np.sin(w.yaw)])


def walk(w, forward, right):
    keys = lambda a, b: (a in w.keys) - (b in w.keys)
    move = forward * keys("w", "s") + right * keys("d", "a")
    speed = 5.5 if "shift" in w.keys else 3.0
    w.me += move / max(np.linalg.norm(move), 1) * speed * w.dt
    walls = w.wall  # push out of every wall at once: closest point on each AABB, in the xz plane
    nearest = np.clip(w.me, walls.pos - walls.scale / 2, walls.pos + walls.scale / 2)
    away = (w.me - nearest) * (1, 0, 1)
    dist = np.linalg.norm(away, axis=1, keepdims=True)
    hit = dist[:, 0] < RADIUS
    w.me += (away[hit] / np.maximum(dist[hit], 1e-6) * (RADIUS - dist[hit])).sum(0)
    bob = np.sin(w.clock * 9) * 0.035 * min(np.linalg.norm(move), 1)
    w.camera.local.position = w.torch.local.position = w.me + (0, bob, 0)
    w.camera.local.rotation = la.quat_mul(la.quat_from_axis_angle((0, 1, 0), w.yaw), la.quat_from_axis_angle((1, 0, 0), w.pitch))


def stalk(w, forward):
    it = w.it
    to_me = (w.me - it.pos[0]) * (1, 0, 1)
    dist = np.linalg.norm(to_me)
    seen = np.dot(-to_me / dist, forward) > np.cos(np.radians(48))
    if not seen:  # it walks through walls. it does not need doors.
        it.pos[0] += to_me / dist * min(dist, (1.3 + w.clock / 90) * w.dt)
        if dist > 22:  # never let it fall too far behind
            it.pos[0, [0, 2]] = (w.me - forward * 14)[[0, 2]]
    if dist < 0.8:
        w.state = "caught"
    return dist


def lights(w, dread):
    lamp = w.lamp  # flicker gets worse the closer it is
    buzz = np.sin(w.t * 37 + lamp.phase) * np.sin(w.t * 5.3 + lamp.phase * 2)
    off = (lamp.dead > 0) | (buzz > 0.97 - dread * 0.6)
    lamp.scale = np.where(off[:, None], 0, (1.2, 0.02, 0.6))
    w.torch.intensity = 6 * (1 - dread * 0.7) * (0.25 if np.sin(w.t * 23) > 0.9 - dread else 1)


def update(w):
    if w.state != "alive":
        w.hud = ("## YOU WERE NOT ALONE" if w.state == "caught" else f"## YOU FOUND THE EXIT\n\n{w.clock:.0f} seconds") + "\n\nF5 to wake up again"
        w.torch.intensity = 0 if w.state == "caught" else 6
        return
    w.clock += w.dt
    forward, right = look(w)
    walk(w, forward, right)
    dist = stalk(w, forward)
    lights(w, float(np.clip(1 - dist / 18, 0, 1)))
    if np.linalg.norm((w.me - w.door.pos[0])[[0, 2]]) < 1.4:
        w.state = "escaped"
    w.hud = "*the lights hum. find the exit.*" if w.clock < 6 else ""
