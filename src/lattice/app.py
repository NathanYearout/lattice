"""Command line: ``lattice`` opens the editor, ``lattice play game.py`` runs a game in a plain window."""

import argparse
import os
import time

import pygfx as gfx

from . import audio, shot
from .runtime import Runtime


def style_window(canvas, color=0x0D0B0B):
    """Windows 11: tint the title bar and border to the theme, and paint exposed areas black instead of white."""
    if os.name != "nt" or not (window := getattr(canvas, "_window", None)):
        return
    import ctypes

    import glfw

    hwnd, dwm, c = glfw.get_win32_window(window), ctypes.windll.dwmapi, ctypes.c_int(color)  # COLORREF is 0x00BBGGRR
    for attribute, value in ((20, ctypes.c_int(1)), (35, c), (34, c)):  # dark mode, caption colour, border colour
        dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(value), 4)
    ctypes.windll.user32.SetClassLongPtrW(hwnd, -10, ctypes.windll.gdi32.GetStockObject(4))  # GCLP_HBRBACKGROUND = black


def play(path, snapshot=None, frames=180, size=(1280, 720), title="lattice · $fps fps"):
    if snapshot:
        from rendercanvas.offscreen import RenderCanvas, loop
    else:
        from rendercanvas.glfw import RenderCanvas, loop
    canvas = RenderCanvas(size=size, title=title, update_mode="continuous", max_fps=240)
    style_window(canvas, 0)
    audio.mixer.enabled = not snapshot
    playable = shot.is_shot(path)
    rt = Runtime(shot.unpack(path) if playable else path, gfx.WgpuRenderer(canvas), canvas=canvas)
    shot.resume(rt) if playable else None
    last = [time.perf_counter()]

    def on_event(e):
        rt.feed(e)
        if e["event_type"] == "key_down" and e["key"] == "F5":
            rt.reload(reset=True)
        if e["event_type"] == "key_down" and e["key"] == "F12":
            print(f"[lattice] playable screenshot: {shot.save(rt, size=tuple(map(int, canvas.get_physical_size())))}")

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
    [canvas.draw() for _ in range(frames)]
    print(f"[lattice] wrote {shot.save(rt, snapshot, size)} (a playable screenshot: `lattice play` it)")


def main():
    p = argparse.ArgumentParser(prog="lattice", description="Lattice: a data-oriented 3D game engine in Python.")
    sub = p.add_subparsers(dest="command")
    e = sub.add_parser("edit", help="open the editor (default)")
    e.add_argument("game", nargs="?", help="game .py file (or playable screenshot .png) to open")
    e.add_argument("--snapshot", metavar="PNG", help="render the editor offscreen and save a screenshot")
    r = sub.add_parser("play", help="run a game in a window with live reload")
    r.add_argument("game", help="a game .py file, or a playable screenshot .png to resume")
    r.add_argument("--snapshot", metavar="PNG", help="render offscreen and save the last frame as a playable screenshot")
    r.add_argument("--frames", type=int, default=180, help="frames to simulate before a snapshot (default 180)")
    b = sub.add_parser("build", help="export a game as a standalone executable (no Python needed to play)")
    b.add_argument("game", help="path to a game .py file")
    b.add_argument("--out", default="dist", help="output folder (default dist/)")
    b.add_argument("--console", action="store_true", help="keep a console window for print() and errors")
    a = p.parse_args()
    if a.command == "play":
        return play(a.game, a.snapshot, a.frames)
    if a.command == "build":
        from .build import build

        return build(a.game, a.out, a.console)
    from .editor import Editor

    Editor(getattr(a, "game", None), getattr(a, "snapshot", None)).run()
