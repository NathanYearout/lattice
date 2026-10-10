import numpy as np
import pygfx as gfx

from lattice import World


def test_restore_empties_kinds_that_were_empty_at_snapshot():
    w = World(gfx.Scene(), gfx.PerspectiveCamera(), None)
    w.kind("enemy", gfx.box_geometry(), hp=3.0)
    w.score = 1
    snap = w.snapshot()  # like pressing Play: no enemies yet
    w.enemy.spawn(5, hp=np.arange(5))
    w.score = 99
    w.restore(snap)  # like pressing Stop
    assert w.enemy.n == 0 and w.score == 1


def test_columns_are_live_views_and_kill_compacts():
    w = World(gfx.Scene(), gfx.PerspectiveCamera(), None)
    w.kind("ball", gfx.sphere_geometry(), hp=1.0)
    w.ball.spawn(200, pos=np.random.rand(200, 3), hp=np.arange(200))  # grows past the initial capacity
    w.ball.pos += (0, 1, 0)
    w.ball.kill(w.ball.hp % 2 == 0)
    assert w.ball.n == 100 and (w.ball.hp % 2 == 1).all() and (w.ball.pos[:, 1] >= 1).all()
