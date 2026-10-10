"""Command line: ``lattice`` opens the editor, ``lattice play game.py`` runs a game in a plain window."""

import argparse
import time

import pygfx as gfx

from .runtime import Runtime


def play(path, snapshot=None, frames=180, size=(1280, 720)):
    if snapshot:
        from rendercanvas.offscreen import RenderCanvas, loop
    else:
        from rendercanvas.glfw import RenderCanvas, loop
    canvas = RenderCanvas(size=size, title="lattice · $fps fps", update_mode="continuous", max_fps=240)
    rt = Runtime(path, gfx.WgpuRenderer(canvas))
    last = [time.perf_counter()]

    def on_event(e):
        rt.feed(e)
        if e["event_type"] == "key_down" and e["key"] == "F5":
            rt.reload(reset=True)

    def frame():
        now = time.perf_counter()
        rt.poll()
        rt.step(1 / 60 if snapshot else min(now - last[0], 1 / 20))
        rt.render()
        last[0] = now

    canvas.add_event_handler(on_event, "key_down", "key_up", "pointer_down", "pointer_up", "pointer_move")
    canvas.request_draw(frame)
    if not snapshot:
        return loop.run()
    from PIL import Image

    image = [canvas.draw() for _ in range(frames)][-1]
    Image.fromarray(image).convert("RGB").save(snapshot)
    print(f"[lattice] wrote {snapshot}")


def main():
    p = argparse.ArgumentParser(prog="lattice", description="Lattice: a data-oriented 3D game engine in Python.")
    sub = p.add_subparsers(dest="command")
    e = sub.add_parser("edit", help="open the editor (default)")
    e.add_argument("game", nargs="?", help="game .py file to open")
    e.add_argument("--snapshot", metavar="PNG", help="render the editor offscreen and save a screenshot")
    r = sub.add_parser("play", help="run a game in a window with live reload")
    r.add_argument("game", help="path to a game .py file")
    r.add_argument("--snapshot", metavar="PNG", help="render offscreen and save the last frame instead")
    r.add_argument("--frames", type=int, default=180, help="frames to simulate before a snapshot (default 180)")
    a = p.parse_args()
    if a.command == "play":
        return play(a.game, a.snapshot, a.frames)
    from .editor import Editor

    Editor(getattr(a, "game", None), getattr(a, "snapshot", None)).run()
