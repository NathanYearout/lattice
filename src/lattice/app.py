"""Runtime: opens a window, hot-reloads the game file on save, feeds input, and renders with pygfx/WebGPU."""

import argparse
import importlib.util
import threading
import time
import traceback
from pathlib import Path

import pygfx as gfx
import watchfiles

from .world import World

KEYS = {" ": "space", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right"}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(path, snapshot=None, frames=180, size=(1280, 720)):
    path = Path(path).resolve()
    if snapshot:
        from rendercanvas.offscreen import RenderCanvas, loop
    else:
        from rendercanvas.glfw import RenderCanvas, loop
    canvas = RenderCanvas(size=size, title=f"lattice · {path.stem} · $fps fps", update_mode="continuous", max_fps=240)
    renderer = gfx.WgpuRenderer(canvas)
    scene, camera = gfx.Scene(), gfx.PerspectiveCamera(60, depth_range=(0.1, 2000))
    sun = gfx.DirectionalLight(intensity=2.5, cast_shadow=True)
    sun.local.position = (30, 60, 20)
    sun.shadow.camera.width = sun.shadow.camera.height = 80
    sun.shadow.bias = 0.001
    scene.add(gfx.Background(None, gfx.BackgroundMaterial("#e8eef5", "#6d9ccf")), gfx.AmbientLight(intensity=0.7), sun, camera)
    text = gfx.Text(markdown="", font_size=18, screen_space=True, anchor="top-left",
                    material=gfx.TextMaterial(color="#fff", outline_color="#000", outline_thickness=0.3))
    text.local.position = (16, 16, 0)
    hud, hud_camera = gfx.Scene().add(text), gfx.ScreenCoordsCamera(invert_y=True)

    w = World(scene, camera, renderer)
    w.sun, w.background = sun, scene.children[0]
    state = dict(game=None, error=None, last=time.perf_counter())
    changed = threading.Event()
    threading.Thread(target=lambda: [changed.set() for _ in watchfiles.watch(path.parent)], daemon=True).start()

    def call(hook, *args):
        if fn := getattr(state["game"], hook, None):
            try:
                fn(*args)
            except Exception:
                state["error"] = f"**{hook}() failed** (fix and save to resume)\n\n" + traceback.format_exc(limit=-3)
                print(state["error"])

    def reload(reset=False):
        try:
            state["game"], state["error"] = load(path), None
        except Exception:
            state["error"] = "**load failed**\n\n" + traceback.format_exc(limit=-3)
            return print(state["error"])
        call("scene", w)  # appearance: re-run on every save
        if reset:  # initial state: only on start / F5
            [k.clear() for k in w.kinds.values()]
            call("setup", w)
        print(f"[lattice] {'started' if reset else 'reloaded'} {path.name}")

    def on_event(e):
        kind, key = e["event_type"], KEYS.get(k := str(e.get("key", "")).lower(), k)
        if kind == "key_down":
            w.keys.add(key), w.pressed.add(key)
            reload(reset=True) if key == "f5" else None
        elif kind == "key_up":
            w.keys.discard(key)
        elif kind == "pointer_down":
            w.buttons.add(e["button"]), w.clicked.add(e["button"])
        elif kind == "pointer_up":
            w.buttons.discard(e["button"])
        if "x" in e:
            w.mouse = (e["x"], e["y"])

    def frame():
        now = time.perf_counter()
        w.dt = 1 / 60 if snapshot else min(now - state["last"], 1 / 20)
        w.t, state["last"] = w.t + w.dt, now
        if changed.is_set():
            changed.clear(), reload()
        if not state["error"]:
            call("update", w)
        for k in w.kinds.values():
            k.sync()
        text.set_markdown(state["error"] or w.hud)
        renderer.render(scene, camera, flush=False)
        renderer.render(hud, hud_camera)
        w.pressed.clear(), w.clicked.clear()

    canvas.add_event_handler(on_event, "key_down", "key_up", "pointer_down", "pointer_up", "pointer_move")
    canvas.request_draw(frame)
    reload(reset=True)
    if not snapshot:
        return loop.run()
    from PIL import Image

    image = [canvas.draw() for _ in range(frames)][-1]
    Image.fromarray(image).convert("RGB").save(snapshot)
    print(f"[lattice] wrote {snapshot}")


def main():
    p = argparse.ArgumentParser(prog="lattice", description="Run a lattice game file with live reload.")
    p.add_argument("game", help="path to a game .py file")
    p.add_argument("--snapshot", metavar="PNG", help="render offscreen and save the last frame instead of opening a window")
    p.add_argument("--frames", type=int, default=180, help="frames to simulate before a snapshot (default 180)")
    a = p.parse_args()
    run(a.game, a.snapshot, a.frames)
