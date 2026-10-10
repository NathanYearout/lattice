import numpy as np
import pygfx as gfx
import pytest

from lattice import World


@pytest.fixture
def w():
    return World(gfx.Scene(), gfx.PerspectiveCamera(), None)


def test_raycast_hits_nearest_box_with_normal(w):
    w.kind("box", gfx.box_geometry())
    w.box.spawn(2, pos=[(0, 0, -5), (0, 0, -10)], scale=[(2, 2, 2), (1, 1, 1)])
    hit = w.raycast((0, 0, 0), (0, 0, -1))
    assert (hit.kind, hit.index) == ("box", 0)
    assert hit.dist == pytest.approx(4.0)
    np.testing.assert_allclose(hit.normal, (0, 0, 1), atol=1e-6)
    assert w.raycast((0, 0, 0), (0, 0, 1)) is None
    assert w.raycast((0, 0, 0), (0, 0, -1), max_dist=3) is None


def test_raycast_respects_rotation(w):
    w.kind("wall", gfx.box_geometry())
    a = np.radians(45) / 2
    w.wall.spawn(pos=(0, 0, -5), scale=(4, 1, 0.2), rot=(0, np.sin(a), 0, np.cos(a)))
    hit = w.raycast((0, 0, 0), (0, 0, -1))
    assert hit is not None and 4.5 < hit.dist < 5.0
    assert hit.normal[2] > 0.6 and abs(hit.normal[0]) > 0.6  # tilted face


def test_overlaps_pairs_and_self(w):
    w.kind("a", gfx.sphere_geometry(0.5))
    w.kind("b", gfx.sphere_geometry(0.5))
    w.a.spawn(3, pos=[(0, 0, 0), (10, 0, 0), (0.6, 0, 0)])
    w.b.spawn(2, pos=[(0.8, 0, 0), (50, 0, 0)])
    i, j = w.overlaps("a", "b")
    assert sorted(zip(i.tolist(), j.tolist())) == [(0, 0), (2, 0)]
    i, j = w.overlaps("a", "a")
    assert list(zip(i.tolist(), j.tolist())) == [(0, 2)]
    assert len(w.overlaps("a", "b", radius_a=0.01, radius_b=0.01)[0]) == 0


def test_push_out_slides_sphere_out_of_walls(w):
    w.kind("wall", gfx.box_geometry())
    w.wall.spawn(pos=(0, 0, 0), scale=(2, 2, 2))
    p, n = w.push_out((1.2, 0, 0), "wall", radius=0.5)  # 0.3 inside the +x face
    np.testing.assert_allclose(p, (1.5, 0, 0), atol=1e-6)
    np.testing.assert_allclose(n, (1, 0, 0), atol=1e-6)
    p, n = w.push_out((0.9, 0, 0.1), "wall", radius=0.5)  # centre inside: least-penetration exit
    assert p[0] == pytest.approx(1.5) and n[0] == pytest.approx(1)
    p, n = w.push_out((5, 0, 0), "wall", radius=0.5)
    np.testing.assert_allclose(p, (5, 0, 0)) and np.testing.assert_allclose(n, 0)


def test_push_out_moves_a_kind_in_place(w):
    w.kind("floor", gfx.box_geometry())
    w.kind("ball", gfx.sphere_geometry(0.5))
    w.floor.spawn(pos=(0, -0.5, 0), scale=(20, 1, 20))
    w.ball.spawn(2, pos=[(0, 0.2, 0), (3, 4, 0)])
    _, normal = w.push_out("ball", "floor", radius=0.5)
    np.testing.assert_allclose(w.ball.pos[:, 1], (0.5, 4.0), atol=1e-6)
    assert normal[0, 1] == pytest.approx(1) and not normal[1].any()


def test_empty_kinds_are_fine(w):
    w.kind("a", gfx.box_geometry())
    w.kind("b", gfx.box_geometry())
    assert w.raycast((0, 0, 0), (1, 0, 0)) is None
    assert len(w.overlaps("a", "b")[0]) == 0 and len(w.overlaps("a", "a")[0]) == 0
    assert w.push_out("a", "b")[0].shape == (0, 3)
    w.b.spawn()
    assert w.push_out("a", "b")[0].shape == (0, 3)
