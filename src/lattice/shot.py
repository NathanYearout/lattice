"""Playable screenshots: a PNG with a zip appended. Image viewers show the frame; Lattice (or any unzip tool) finds the game,
the assets it used and the exact world state, and resumes play from that moment."""

import io
import json
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np
import pygfx as gfx
import wgpu
from PIL import Image

LOOK = ("pixel_ratio", "pixel_filter", "ppaa", "effect_passes")
HOME = Path.home() / ".lattice" / "shots"


def grab(rt, size=(1280, 720)):
    """Render the game camera, with the game's look and HUD, into an offscreen texture and read it back as RGBA."""
    tex = gfx.Texture(dim=2, size=(*size, 1), format="rgba8unorm", usage=wgpu.TextureUsage.COPY_SRC)
    r = gfx.WgpuRenderer(tex)
    for attr in LOOK:
        setattr(r, attr, getattr(rt.world.renderer, attr))
    rt.render(renderer=r, consume_input=False)
    data = r.device.queue.read_texture({"texture": tex._wgpu_object, "origin": (0, 0, 0), "mip_level": 0},
                                       {"offset": 0, "bytes_per_row": size[0] * 4, "rows_per_image": size[1]}, (*size, 1))
    return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)


def plain(value):  # game state worth keeping: data, not live engine objects like lights or meshes
    try:
        return isinstance(value, (int, float, str, bytes, type(None), np.ndarray, np.generic, list, tuple, dict, set)) and bool(pickle.dumps(value))
    except Exception:
        return False


def save(rt, path=None, size=(1280, 720)):
    w, root = rt.world, rt.path.parent
    path = Path(path or Path.home() / "Pictures" / f"{rt.path.stem}-{time.strftime('%Y%m%d-%H%M%S')}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    cam = w.camera
    meta = dict(game=rt.path.name, t=w.t, hud=w.hud, camera=[list(map(float, cam.local.position)), list(map(float, cam.local.rotation)), cam.fov])
    out = io.BytesIO()
    Image.fromarray(grab(rt, size)).convert("RGB").save(out, "PNG")
    with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as z:  # appending to a non-zip writes a zip after IEND
        z.writestr("lattice.json", json.dumps(meta))
        for f in {rt.path, *w.files}:
            if f.is_file() and f.is_relative_to(root):
                z.write(f, f.relative_to(root).as_posix())
        moment = io.BytesIO()
        w.save(moment)
        z.writestr("moment.npz", moment.getvalue())
        z.writestr("state.pkl", pickle.dumps({k: v for k, v in w.snapshot()[1].items() if plain(v)}))
    path.write_bytes(out.getvalue())
    return path


def is_shot(path):
    return Path(path).suffix.lower() == ".png" and zipfile.is_zipfile(path)


def unpack(path):
    """Extract a shot into ~/.lattice/shots/<name>/ and return the game file inside."""
    with zipfile.ZipFile(path) as z:
        folder = HOME / Path(path).stem
        z.extractall(folder)
        return folder / json.loads(z.read("lattice.json"))["game"]


def resume(rt):
    """Put a freshly loaded shot's world back into the captured moment."""
    w, folder = rt.world, rt.path.parent
    meta = json.loads((folder / "lattice.json").read_text())
    w.load(folder / "moment.npz")
    vars(w).update(pickle.loads((folder / "state.pkl").read_bytes()))
    (pos, rot, fov), w.t, w.hud = meta["camera"], meta["t"], meta["hud"]
    w.camera.local.position, w.camera.local.rotation, w.camera.fov = pos, rot, fov
