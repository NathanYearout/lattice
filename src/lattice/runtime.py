"""A loaded game: its world, its module, hot reload and the default scene dressing. Shared by player and editor."""

import importlib.util
import threading
import traceback
from pathlib import Path

import pygfx as gfx
import watchfiles

from .world import World

KEYS = {" ": "space", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right"}


class Runtime:
    def __init__(self, path, renderer, log=lambda level, msg: print(msg)):
        self.path, self.log, self.game, self.error = Path(path).resolve(), log, None, None
        scene, camera = gfx.Scene(), gfx.PerspectiveCamera(60, depth_range=(0.1, 2000))
        sun = gfx.DirectionalLight(intensity=2.5, cast_shadow=True)
        sun.local.position = (30, 60, 20)
        sun.shadow.camera.width = sun.shadow.camera.height = 80
        sun.shadow.bias = 0.001
        background, ambient = gfx.Background(None, gfx.BackgroundMaterial("#e8eef5", "#6d9ccf")), gfx.AmbientLight(intensity=0.7)
        scene.add(background, ambient, sun, camera)
        self.text = gfx.Text(markdown="", font_size=18, screen_space=True, anchor="top-left",
                             material=gfx.TextMaterial(color="#fff", outline_color="#000", outline_thickness=0.3))
        self.text.local.position = (16, 16, 0)
        self.hud, self.hud_camera = gfx.Scene().add(self.text), gfx.ScreenCoordsCamera(invert_y=True)

        self.world = w = World(scene, camera, renderer)
        w.sun, w.background, w.ambient, w.text = sun, background, ambient, self.text
        w._engine = set(vars(w)) | {"_engine"}  # everything else on `w` is game state (snapshotted on Play)
        self.changed, self._stop = threading.Event(), threading.Event()
        watch = lambda: [self.changed.set() for _ in watchfiles.watch(self.path.parent, stop_event=self._stop)]
        threading.Thread(target=watch, daemon=True).start()
        self.reload(reset=True)

    @property
    def scene_file(self):
        return self.path.with_suffix(".scene.npz")

    def call(self, hook, *args):
        if fn := getattr(self.game, hook, None):
            try:
                fn(*args)
            except Exception:
                self.error = f"**{hook}() failed** (fix and save to resume)\n\n" + traceback.format_exc(limit=-3)
                self.log("error", self.error)

    def reload(self, reset=False):
        try:
            spec = importlib.util.spec_from_file_location(self.path.stem, self.path)
            game = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(game)
            self.game, self.error = game, None
        except Exception:
            self.error = "**load failed**\n\n" + traceback.format_exc(limit=-3)
            return self.log("error", self.error)
        self.call("scene", self.world)  # appearance: re-run on every save
        if reset:  # initial state: only on start / restart
            [k.clear() for k in self.world.kinds.values()]
            self.call("setup", self.world)
            self.world.load(self.scene_file) if self.scene_file.exists() else None
        self.log("info", f"{'Loaded' if reset else 'Reloaded'} {self.path.name}")

    def poll(self):
        if self.changed.is_set():
            self.changed.clear()
            self.reload()

    def step(self, dt):
        w = self.world
        w.dt, w.t = dt, w.t + dt
        if not self.error:
            self.call("update", w)

    def render(self, camera=None, hud=True, renderer=None):
        w, r = self.world, renderer or self.world.renderer
        for k in w.kinds.values():
            k.sync()
        self.text.set_markdown((self.error or w.hud) if hud else "")
        r.render(w.scene, camera or w.camera, flush=False)
        r.render(self.hud, self.hud_camera)
        w.pressed.clear(), w.clicked.clear()
        w.mouse_delta = (0.0, 0.0)

    def feed(self, e, offset=(0, 0)):
        """Translate a rendercanvas event into world input; ``offset`` maps window to viewport coordinates."""
        w, kind = self.world, e["event_type"]
        key = KEYS.get(k := str(e.get("key", "")).lower(), k)
        if kind == "key_down":
            w.keys.add(key), w.pressed.add(key)
        elif kind == "key_up":
            w.keys.discard(key)
        elif kind == "pointer_down":
            w.buttons.add(e["button"]), w.clicked.add(e["button"])
        elif kind == "pointer_up":
            w.buttons.discard(e["button"])
        if "x" in e:
            x, y = e["x"] - offset[0], e["y"] - offset[1]
            dx, dy = w.mouse_delta
            w.mouse_delta, w.mouse = (dx + x - w.mouse[0], dy + y - w.mouse[1]), (x, y)

    def close(self):
        self._stop.set()
        for k in self.world.kinds.values():
            self.world.scene.remove(k.obj)
