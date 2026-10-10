"""Arena: a top-down collision playground. WASD to move, click to fire a laser at the cursor.

Walls are solid (``w.push_out``), gems are collected on touch and chasers hurt (``w.overlaps``), and the laser stops at the
first thing it hits (``w.raycast``). Every query runs over whole kinds at once.
"""

import numpy as np
import pygfx as gfx

from lattice import fx, gen

SIZE, SEED = 40.0, 2


def scene(w):
    w.background.material = gfx.BackgroundMaterial("#1b1530", "#3a2b66")
    w.sun.local.position, w.sun.intensity, w.ambient.intensity = (10, 30, 8), 2.6, 0.6
    w.camera.fov = 45
    w.kind("floor", gfx.box_geometry(), "#2a2440")
    w.kind("wall", gfx.box_geometry(), "#8f7cff")
    w.kind("player", gfx.sphere_geometry(0.5, 24, 16), "#ffd23f")
    w.kind("gem", gfx.octahedron_geometry(0.35), material="basic", shadow=False, spin=0.0)
    w.kind("chaser", gfx.box_geometry(0.9, 0.9, 0.9), "#ff4f6d", speed=0.0)
    w.kind("laser", gfx.box_geometry(), material="basic", shadow=False, life=0.0)
    fx.style(w, "cel")


def setup(w):
    rng = np.random.default_rng(SEED)
    w.floor.spawn(pos=(0, -0.5, 0), scale=(SIZE, 1, SIZE))
    edge = SIZE / 2
    w.wall.spawn(4, pos=[(0, 1, edge), (0, 1, -edge), (edge, 1, 0), (-edge, 1, 0)], scale=[(SIZE, 2, 1), (SIZE, 2, 1), (1, 2, SIZE), (1, 2, SIZE)])
    blocks = gen.scatter(26, SIZE - 8, SEED, lambda x, z: np.hypot(x, z) > 5, 5)
    angle = rng.uniform(0, np.pi, len(blocks)) / 2
    w.wall.spawn(len(blocks), pos=np.c_[blocks[:, 0], np.ones(len(blocks)), blocks[:, 1]],
                 scale=np.c_[rng.uniform(1, 6, len(blocks)), np.full(len(blocks), 2), rng.uniform(1, 2, len(blocks))],
                 rot=np.c_[np.zeros(len(blocks)), np.sin(angle), np.zeros(len(blocks)), np.cos(angle)])
    w.player.spawn(pos=(0, 0.5, 0))
    w.score, w.hp, w.cooldown = 0, 5, 0.0
    spawn_gems(w, 60, rng)
    update(w, dt=0.0)


def spawn_gems(w, n, rng):
    p = gen.scatter(n * 2, SIZE - 4, int(rng.integers(1 << 30)))
    p, _ = w.push_out(np.c_[p[:, 0], np.full(len(p), 0.8), p[:, 1]], "wall", 0.6)  # never inside a wall
    w.gem.spawn(len(p[:n]), pos=p[:n], spin=rng.uniform(1, 3, len(p[:n])), tint=gen.palette(n, "#39ff9f", "#05d9e8", "#ffd319", seed=w.score))


def update(w, dt=None):
    dt, rng = (w.dt if dt is None else dt), np.random.default_rng()
    me = w.player
    keys = lambda a, b: (a in w.keys) - (b in w.keys)
    move = np.array([keys("d", "a"), 0, keys("s", "w")], float)
    me.pos += move / max(np.linalg.norm(move), 1) * 9 * dt
    w.push_out(me, "wall", 0.5)

    g = w.gem  # spin, bob and collect
    a = w.t * g.spin
    g.rot = np.c_[np.zeros(g.n), np.sin(a / 2), np.zeros(g.n), np.cos(a / 2)]
    i, j = w.overlaps("player", "gem")
    w.score += len(j)
    g.kill(j)
    if g.n < 20:
        spawn_gems(w, 40, rng)

    c = w.chaser  # chasers home in, push each other apart and slide along walls
    if c.n < 3 + w.score // 15 and rng.random() < dt:
        corner = rng.choice([-1, 1], 2) * (SIZE / 2 - 2)
        c.spawn(pos=(corner[0], 0.45, corner[1]), speed=rng.uniform(2.5, 4.5))
    to_me = (me.pos[0] - c.pos) * (1, 0, 1)
    c.pos += to_me / np.maximum(np.linalg.norm(to_me, axis=1, keepdims=True), 1e-6) * c.speed[:, None] * dt
    a, b = w.overlaps("chaser", "chaser", 0.5, 0.5)
    np.add.at(c.pos, a, (c.pos[a] - c.pos[b]) * 0.1), np.add.at(c.pos, b, (c.pos[b] - c.pos[a]) * 0.1)
    w.push_out(c, "wall", 0.45)
    _, hurt = w.overlaps("player", "chaser")
    w.hp -= len(hurt)
    c.kill(hurt)

    w.cooldown -= dt  # laser: from the player towards the cursor, stopped by the first wall or chaser
    if 1 in w.buttons and w.cooldown <= 0 and w.renderer is not None:
        w.cooldown = 0.12
        aim = (w.mouse_ground(0.5) - me.pos[0]) * (1, 0, 1)
        aim /= max(np.linalg.norm(aim), 1e-6)
        hit = w.raycast(me.pos[0] + aim * 0.6, aim, ["wall", "chaser"], 60)
        length = hit.dist if hit else 60
        if hit and hit.kind == "chaser":
            c.kill([hit.index])
            w.score += 3
        yaw = np.arctan2(aim[0], aim[2]) / 2
        w.laser.spawn(pos=me.pos[0] + aim * (0.6 + length / 2), scale=(0.12, 0.12, length), rot=(0, np.sin(yaw), 0, np.cos(yaw)),
                      tint=(2.5, 0.6, 2.2), life=0.1)
    lz = w.laser
    lz.life -= dt
    lz.scale[:, :2] *= 0.85
    lz.kill(lz.life <= 0)

    w.camera.local.position = me.pos[0] + (0, 26, 14)
    w.camera.look_at(me.pos[0])
    if w.hp <= 0:
        w.hud = f"## CAUGHT\n\nscore {w.score} · F5 to try again"
        c.speed = 0.0
    else:
        w.hud = f"**ARENA**  ·  score {w.score}  ·  hp {w.hp}/5"
