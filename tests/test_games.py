"""Every example game loads, plays headlessly without errors, and round-trips through a playable screenshot."""

from pathlib import Path

import numpy as np
import pygfx as gfx
import pytest
from rendercanvas.offscreen import RenderCanvas

from lattice import audio, shot
from lattice.runtime import Runtime

GAMES = sorted((Path(__file__).parents[1] / "games").glob("*.py"))
audio.mixer.enabled = False


@pytest.mark.parametrize("game", GAMES, ids=lambda p: p.stem)
def test_game_plays_and_resumes_from_a_shot(game, tmp_path, monkeypatch):
    monkeypatch.setattr(shot, "HOME", tmp_path / "shots")
    errors = []
    rt = Runtime(game, gfx.WgpuRenderer(RenderCanvas(size=(320, 180))), log=lambda lvl, msg: errors.append(msg) if lvl == "error" else None)
    for _ in range(10):
        rt.step(1 / 60)
        rt.render()
    assert not errors, errors[0]
    assert sum(map(len, rt.world.kinds.values())) > 0

    png = shot.save(rt, tmp_path / "shot.png", (320, 180))
    assert shot.is_shot(png)
    before = {n: k.pos.copy() for n, k in rt.world.kinds.items()}
    resumed = Runtime(shot.unpack(png), gfx.WgpuRenderer(RenderCanvas(size=(320, 180))), log=lambda *a: None)
    shot.resume(resumed)
    for name, pos in before.items():
        np.testing.assert_allclose(resumed.world.kinds[name].pos, pos, atol=1e-5)
    rt.close(), resumed.close()
