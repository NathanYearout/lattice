"""The Lattice editor: Dear ImGui (imgui-bundle) docked around a pygfx viewport, sharing one WebGPU device."""

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pygfx as gfx
import pylinalg as la
from imgui_bundle import imgui, imguizmo
from imgui_bundle import imgui_color_text_edit as ed
from imgui_bundle import portable_file_dialogs as pfd
from wgpu.utils.imgui import ImguiRenderer

from .. import audio, fx, shot
from .. import icons as I
from ..app import style_window
from ..world import MATERIALS, convert
from ..runtime import Runtime
from . import theme
from .theme import Fonts, hex4

gz = imguizmo.im_guizmo
PANEL = imgui.WindowFlags_.no_focus_on_appearing  # only the viewport may grab focus
HOME = Path.home() / ".lattice"
EXAMPLES = Path(__file__).resolve().parents[3] / "games"
TEMPLATE = '''"""{name}: made with Lattice. Edit this file and save; the world updates instantly."""

import numpy as np
import pygfx as gfx


def scene(w):  # what things look like: re-run on every save
    w.kind("ground", gfx.box_geometry(), "#2a2a32")
    w.kind("cube", gfx.box_geometry(), "#ffb347", spin=0.0)


def setup(w):  # the starting state: run once (and on Restart)
    w.ground.spawn(pos=(0, -0.5, 0), scale=(40, 1, 40))
    xz = np.mgrid[-4:5, -4:5].reshape(2, -1).T * 2.0
    w.cube.spawn(len(xz), pos=np.c_[xz[:, 0], np.full(len(xz), 0.5), xz[:, 1]], spin=np.random.uniform(-2, 2, len(xz)))
    w.camera.local.position = (12, 10, 14)
    w.camera.look_at((0, 0, 0))


def update(w):  # every frame while playing
    c = w.cube
    c.rot = np.c_[np.zeros((c.n, 1)), np.sin(w.t * c.spin / 2)[:, None], np.zeros((c.n, 1)), np.cos(w.t * c.spin / 2)[:, None]]
    c.pos[:, 1] = 0.5 + np.abs(np.sin(w.t * 2 + c.pos[:, 0]))
'''


def u32(h, a=1.0):
    return imgui.get_color_u32(hex4(h, a))


def icon_button(icon, tip, active=False, size=30, color=None):
    """Square icon button with tooltip; ``active`` paints it in the accent like a toggled tool."""
    imgui.push_style_color(imgui.Col_.text, hex4(color or (theme.ACCENT if active else theme.TEXT)))
    imgui.push_style_color(imgui.Col_.button, hex4(theme.ACTIVE if active else theme.RAISED, 1 if active else 0))
    clicked = imgui.button(f"{icon}##{tip}", imgui.ImVec2(size, size))
    imgui.pop_style_color(2)
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_short):
        imgui.set_tooltip(tip)
    return clicked


def matrix(pos, rot, scale):
    return la.mat_compose(pos, rot / np.linalg.norm(rot), scale)


class Console:
    def __init__(self):
        self.lines, self.filters = [], {"info": True, "warn": True, "error": True}

    def log(self, level, msg):
        for i, line in enumerate(str(msg).rstrip().splitlines() or [""]):
            self.lines.append((level, time.strftime("%H:%M:%S") if i == 0 else "", line))
        del self.lines[:-2000]

    def write(self, text):  # stands in for sys.stdout so game print() lands here
        if text.strip():
            self.log("info", text)

    def flush(self):
        pass


class Editor:
    def __init__(self, path=None, snapshot=None):
        if snapshot:
            from rendercanvas.offscreen import RenderCanvas
        else:
            from rendercanvas.glfw import RenderCanvas
        self.snapshot = snapshot
        self.canvas = RenderCanvas(size=(1600, 940), title="Lattice", update_mode="continuous", max_fps=144)
        style_window(self.canvas)
        self.gui = ImguiRenderer(gfx.renderers.wgpu.get_shared().device, self.canvas)
        HOME.mkdir(exist_ok=True)
        imgui.get_io().set_ini_filename(str(HOME / "layout.ini"))
        theme.apply()
        self.console, self.toasts = Console(), []
        sys.stdout = self.console
        self.rt = self.renderer = None
        self.effects, self.game_lighting = False, False
        self.size, self.mode, self.view = (0, 0), "edit", "scene"
        self.selected, self.op, self.world_space, self.snap, self.show_grid = None, gz.OPERATION.translate, True, False, True
        self.undo, self.redo, self.play_state, self.was_using = [], [], None, False
        self.cam = gfx.PerspectiveCamera(60, depth_range=(0.05, 2000))
        self.cam_pos, self.yaw, self.pitch, self.speed = np.array([14.0, 10, 18]), 0.66, -0.4, 8.0
        self.grid = gfx.Grid(None, gfx.GridMaterial(major_step=(4, 4), minor_step=(1, 1), major_thickness=1.5, minor_thickness=0.6,
                                                    major_color="#3a3a44", minor_color="#24242b", axis_color=theme.ACCENT_DIM,
                                                    axis_thickness=2), orientation="xz")
        self.box = gfx.BoxHelper(thickness=2, color=theme.ACCENT)
        self.scripts, self.script_tab, self.frame_ms, self.layout_done = {}, None, np.zeros(120, np.float32), (HOME / "layout.ini").exists()
        recent = json.loads((HOME / "recent.json").read_text()) if (HOME / "recent.json").exists() else []
        self.recent = list(dict.fromkeys(p for p in recent if HOME not in Path(p).parents))  # unique; no unpacked shots
        self.last, self.focus_viewport = time.perf_counter(), True
        self.ensure_target((1280, 720))
        self.canvas.add_event_handler(self.on_event, "key_down", "key_up", "pointer_down", "pointer_up", "pointer_move", order=-100)
        if window := getattr(self.canvas, "_window", None):  # drop a game or a playable screenshot onto the window
            import glfw

            glfw.set_drop_callback(window, lambda _, paths: setattr(self, "dropped", paths[0]))
        self.gui.set_gui(self.frame)
        self.open(path) if path else None

    def run(self, frames=60):
        self.canvas.request_draw(self.gui.render)
        if not self.snapshot:
            from rendercanvas.glfw import loop

            return loop.run()
        from PIL import Image

        Image.fromarray([self.canvas.draw() for _ in range(frames)][-1]).convert("RGB").save(self.snapshot)

    # ------------------------------------------------------------------ project
    def open(self, path):
        source, playable = Path(path).resolve(), shot.is_shot(path)
        path = shot.unpack(source) if playable else source
        if self.rt:
            self.stop() if self.mode != "edit" else None
            self.rt.close()
        size, self.size, self.renderer = self.size, (0, 0), None
        self.ensure_target(size)  # fresh renderers: no look leaks from the previous game
        self.rt, self.selected, self.undo, self.redo = Runtime(path, self.renderer, self.console.log, self.canvas), None, [], []
        self.rt.world.scene.add(self.grid, self.box)
        cam = self.rt.world.camera  # the Scene view starts where the game camera looks from
        f = la.vec_transform_quat((0, 0, -1), cam.local.rotation)
        self.cam_pos, self.yaw, self.pitch = np.array(cam.local.position, float), float(np.arctan2(-f[0], -f[2])), float(np.arcsin(f[1]))
        self.recent = [str(source)] + [p for p in self.recent if p != str(source) and HOME not in Path(p).parents][:7]
        (HOME / "recent.json").write_text(json.dumps(self.recent))
        self.canvas.set_title(f"Lattice · {path.name}")
        self.edit_script(path)
        self.focus_viewport = True
        self.toast(I.FOLDER_OPEN, f"Opened {path.name}")
        if playable:  # straight back into the captured moment; Stop returns to it
            shot.resume(self.rt)
            self.play()
            self.toast(I.CAMERA, "Resumed from a playable screenshot")

    def new_game(self):
        if target := pfd.save_file("New game", str(Path.cwd() / "my_game.py"), ["Python", "*.py"]).result():
            target = Path(target).with_suffix(".py")
            target.write_text(TEMPLATE.format(name=target.stem.replace("_", " ").title()))
            self.open(target)

    def open_dialog(self):
        if files := pfd.open_file("Open game", str(EXAMPLES if EXAMPLES.exists() else Path.cwd()), ["Games", "*.py *.png"]).result():
            self.open(files[0])

    def toast(self, icon, text, color=theme.ACCENT):
        self.toasts.append((icon, text, color, time.perf_counter()))

    def log_toast(self, level, msg):
        self.console.log(level, msg)

    # ------------------------------------------------------------------ play / undo
    def play(self):
        if self.mode == "edit":
            self.play_state, self.mode, self.view = self.rt.world.snapshot(), "play", "game"
            self.focus_viewport = True
            self.toast(I.PLAY, "Playing: changes revert on Stop")
        else:
            self.mode = "paused" if self.mode == "play" else "play"

    def stop(self):
        if self.mode != "edit":
            self.rt.world.restore(self.play_state)
            self.mode, self.view = "edit", "scene"
            w = self.rt.world
            w.keys.clear(), w.buttons.clear(), w.capture_mouse(False), audio.mixer.stop_all()

    def screenshot(self):
        if self.rt:
            self.grid.visible = self.box.visible = False
            path = shot.save(self.rt, size=self.size)
            self.toast(I.CAMERA, f"Playable screenshot ï¿½ {path.name}", theme.OK)
            self.console.log("info", f"Saved {path}  (open it in Lattice to play from this moment)")

    def checkpoint(self):
        if self.rt and self.mode == "edit":
            self.undo.append(self.rt.world.snapshot())
            del self.undo[:-64]
            self.redo.clear()

    def undo_redo(self, back=True):
        src, dst = (self.undo, self.redo) if back else (self.redo, self.undo)
        if src and self.mode == "edit":
            dst.append(self.rt.world.snapshot())
            self.rt.world.restore(src.pop())
            self.selected = None

    def selection(self):
        if self.rt and self.selected:
            name, i = self.selected
            if (k := self.rt.world.kinds.get(name)) is not None and i < k.n:
                return name, k, i
        self.selected = None

    def duplicate(self):
        if sel := self.selection():
            self.checkpoint()
            name, k, i = sel
            idx = k.spawn(**{c: col[i].copy() for c, col in k._cols.items() if k.n > i})
            k.pos[idx[0]] += (0.5, 0, 0.5)
            self.selected = (name, int(idx[0]))

    def delete(self):
        if sel := self.selection():
            self.checkpoint()
            sel[1].kill([sel[2]])
            self.selected = None

    # ------------------------------------------------------------------ input
    def on_event(self, e):
        """Raw canvas events go to the game only while it is playing and the viewport has the mouse/keyboard."""
        if self.rt and self.mode == "play" and getattr(self, "viewport_focused", False):
            self.rt.feed(e, getattr(self, "viewport_origin", (0, 0)))
        elif self.rt and e["event_type"] == "key_down" and str(e.get("key")).lower() == "escape":
            self.rt.world.capture_mouse(False)

    def shortcuts(self):
        io = imgui.get_io()
        ctrl, typing = io.key_ctrl, io.want_text_input
        pressed = lambda key: imgui.is_key_pressed(key, False)
        if ctrl and pressed(imgui.Key.p):
            self.play() if self.mode != "play" else self.stop()
        if ctrl and pressed(imgui.Key.s):
            self.save_script() if self.script_focused else self.save_scene()
        if ctrl and pressed(imgui.Key.n):
            self.new_game()
        if ctrl and pressed(imgui.Key.o):
            self.open_dialog()
        if pressed(imgui.Key.f12):
            self.screenshot()
        if typing or self.mode != "edit":
            return
        if ctrl and pressed(imgui.Key.z):
            self.undo_redo(True)
        if ctrl and pressed(imgui.Key.y):
            self.undo_redo(False)
        if ctrl and pressed(imgui.Key.d):
            self.duplicate()
        if not ctrl and self.viewport_hovered and not imgui.is_mouse_down(imgui.MouseButton_.right):
            for key, op in ((imgui.Key.w, gz.OPERATION.translate), (imgui.Key.e, gz.OPERATION.rotate), (imgui.Key.r, gz.OPERATION.scale)):
                self.op = op if pressed(key) else self.op
            if pressed(imgui.Key.f):
                self.frame_selected()
        if pressed(imgui.Key.delete):
            self.delete()

    def save_scene(self):
        if self.rt and self.mode == "edit":
            self.rt.world.save(self.rt.scene_file)
            self.toast(I.SAVE, f"Scene saved · {self.rt.scene_file.name}", theme.OK)

    # ------------------------------------------------------------------ frame
    def frame(self):
        now = time.perf_counter()
        dt, self.last = min(now - self.last, 1 / 20), now
        self.frame_ms = np.roll(self.frame_ms, -1)
        self.frame_ms[-1] = dt * 1000
        gz.begin_frame()
        self.viewport_hovered = getattr(self, "viewport_hovered", False)
        self.script_focused = getattr(self, "script_focused", False)
        if path := getattr(self, "dropped", None):
            self.dropped = None
            self.open(path)
        audio.mixer.paused = self.mode != "play"
        if self.rt:
            self.rt.poll()
            if self.mode == "play":
                self.rt.step(dt)
        self.shortcuts()
        self.dockspace()
        self.viewport_window(dt)
        self.hierarchy_window()
        self.inspector_window()
        self.project_window()
        self.console_window()
        self.script_window()
        self.draw_toasts()

    def dockspace(self):
        vp = imgui.get_main_viewport()
        imgui.set_next_window_pos(vp.work_pos)
        imgui.set_next_window_size(vp.work_size)
        imgui.set_next_window_viewport(vp.id_)
        flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_docking
                 | imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_nav_focus | imgui.WindowFlags_.menu_bar)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0)
        imgui.push_style_color(imgui.Col_.window_bg, hex4(theme.BG))
        imgui.begin("Root", None, flags)
        imgui.pop_style_var(2)
        imgui.pop_style_color()
        self.menu_bar()
        self.toolbar()
        dock_id = imgui.get_id("MainDock")
        if not self.layout_done:
            self.default_layout(dock_id, imgui.get_content_region_avail())
        imgui.dock_space(dock_id, imgui.ImVec2(0, -26))
        self.status_bar()
        imgui.end()

    def default_layout(self, dock_id, size):
        dock = imgui.internal
        dock.dock_builder_remove_node(dock_id)
        dock.dock_builder_add_node(dock_id, imgui.DockNodeFlags_.none | (1 << 10))  # ImGuiDockNodeFlags_DockSpace
        dock.dock_builder_set_node_size(dock_id, size)
        _, left, rest = dock.dock_builder_split_node_py(dock_id, imgui.Dir.left, 0.17)
        _, right, center = dock.dock_builder_split_node_py(rest, imgui.Dir.right, 0.25)
        _, bottom, center = dock.dock_builder_split_node_py(center, imgui.Dir.down, 0.28)
        for name, node in (("###Hierarchy", left), ("###Inspector", right), ("###Project", bottom), ("###Console", bottom),
                           ("###Script", center), ("###Viewport", center)):
            dock.dock_builder_dock_window(name, node)
        dock.dock_builder_finish(dock_id)
        self.layout_done = True

    def menu_bar(self):
        if not imgui.begin_menu_bar():
            return
        imgui.push_font(Fonts.bold, 0)
        imgui.text_colored(hex4(theme.ACCENT), f"{I.BOXES}  LATTICE")
        imgui.pop_font()
        imgui.dummy(imgui.ImVec2(8, 0))
        item = lambda label, shortcut="", enabled=True: imgui.menu_item(label, shortcut, False, enabled)[0]
        if imgui.begin_menu("File"):
            if item(f"{I.FILE_PLUS}  New Game", "Ctrl+N"):
                self.new_game()
            if item(f"{I.FOLDER_OPEN}  Open Game", "Ctrl+O"):
                self.open_dialog()
            if imgui.begin_menu(f"{I.HOUSE}  Recent", bool(self.recent)):
                for p in self.recent:
                    if item(f"{Path(p).name}   {Path(p).parent.name}###{p}"):
                        self.open(p)
                imgui.end_menu()
            imgui.separator()
            if item(f"{I.SAVE}  Save Scene", "Ctrl+S", bool(self.rt)):
                self.save_scene()
            if item(f"{I.CAMERA}  Playable Screenshot", "F12", bool(self.rt)):
                self.screenshot()
            if item(f"{I.REFRESH_CW}  Restart Game", "", bool(self.rt)):
                self.stop(), self.rt.reload(reset=True)
            imgui.separator()
            if item(f"{I.X}  Exit"):
                self.canvas.close()
            imgui.end_menu()
        if imgui.begin_menu("Edit"):
            if item(f"{I.UNDO_2}  Undo", "Ctrl+Z", bool(self.undo)):
                self.undo_redo(True)
            if item(f"{I.REDO_2}  Redo", "Ctrl+Y", bool(self.redo)):
                self.undo_redo(False)
            imgui.separator()
            if item(f"{I.COPY}  Duplicate", "Ctrl+D", bool(self.selected)):
                self.duplicate()
            if item(f"{I.TRASH_2}  Delete", "Del", bool(self.selected)):
                self.delete()
            imgui.end_menu()
        if imgui.begin_menu("View"):
            if item(f"{I.GRID_3X3}  Grid", "", True):
                self.show_grid = not self.show_grid
            if item(f"{I.LAYOUT_DASHBOARD}  Reset Layout"):
                self.layout_done = False
            imgui.end_menu()
        if imgui.begin_menu("Help"):
            if item(f"{I.BOOK_OPEN}  Documentation"):
                os.startfile(Path(__file__).resolve().parents[3] / "README.md") if os.name == "nt" else None
            if item(f"{I.KEYBOARD}  Shortcuts"):
                self.toast(I.KEYBOARD, "W/E/R move·rotate·scale  F frame  RMB+WASD fly  Ctrl+P play")
            imgui.end_menu()
        imgui.end_menu_bar()

    def toolbar(self):
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(4, 0))
        imgui.set_cursor_pos(imgui.ImVec2(8, imgui.get_cursor_pos().y + 6))
        tools = ((I.MOVE, "Move (W)", gz.OPERATION.translate), (I.ROTATE_3D, "Rotate (E)", gz.OPERATION.rotate),
                 (I.SCALING, "Scale (R)", gz.OPERATION.scale))
        for icon, tip, op in tools:
            if icon_button(icon, tip, self.op == op):
                self.op = op
            imgui.same_line()
        imgui.dummy(imgui.ImVec2(8, 0)), imgui.same_line()
        if icon_button(I.GLOBE if self.world_space else I.BOX, "World space" if self.world_space else "Local space", False):
            self.world_space = not self.world_space
        imgui.same_line()
        if icon_button(I.MAGNET, "Snap to grid", self.snap):
            self.snap = not self.snap
        imgui.same_line()
        if icon_button(I.GRID_3X3, "Show grid", self.show_grid):
            self.show_grid = not self.show_grid
        # centred transport controls
        imgui.same_line(imgui.get_window_width() / 2 - 51)
        playing = self.mode != "edit"
        if icon_button(I.PLAY, "Play (Ctrl+P)", playing, color=theme.ACCENT if playing else None) and self.rt:
            self.play() if not playing else self.stop()
        imgui.same_line()
        if icon_button(I.PAUSE, "Pause", self.mode == "paused") and self.rt and playing:
            self.play()
        imgui.same_line()
        if icon_button(I.STEP_FORWARD, "Step one frame") and self.rt and playing:
            self.mode = "paused"
            self.rt.step(1 / 60)
        imgui.same_line(), imgui.dummy(imgui.ImVec2(8, 0)), imgui.same_line()
        if icon_button(I.CAMERA, "Playable screenshot (F12): a PNG that resumes this exact moment") and self.rt:
            self.screenshot()
        # stats on the right
        if self.rt:
            w = self.rt.world
            stats = f"{I.ACTIVITY} {1000 / max(self.frame_ms.mean(), 1e-3):.0f} fps   {I.BOXES} {sum(map(len, w.kinds.values())):,}   {I.LAYERS} {len(w.kinds)} draws"
            imgui.same_line(imgui.get_window_width() - imgui.calc_text_size(stats).x - 16)
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 6)
            imgui.text_colored(hex4(theme.DIM), stats)
        imgui.pop_style_var()
        imgui.dummy(imgui.ImVec2(0, 6))

    def status_bar(self):
        imgui.set_cursor_pos_x(12)
        if not self.rt:
            return imgui.text_colored(hex4(theme.DIM), "No game open")
        color, label = {"edit": (theme.DIM, "EDIT"), "play": (theme.OK, "PLAYING"), "paused": (theme.WARN, "PAUSED")}[self.mode]
        imgui.text_colored(hex4(color), f"{I.ZAP} {label}")
        imgui.same_line(), imgui.text_colored(hex4(theme.FAINT), "·"), imgui.same_line()
        imgui.text_colored(hex4(theme.DIM), f"{I.FILE_CODE} {self.rt.path}")
        if self.rt.error:
            imgui.same_line(), imgui.text_colored(hex4(theme.ERROR), f"   {I.CIRCLE_ALERT} {self.rt.error.splitlines()[0].strip('*')}")

    # ------------------------------------------------------------------ viewport
    LOOK = ("pixel_ratio", "pixel_filter", "ppaa", "effect_passes")

    def ensure_target(self, size, settle=0.12):
        """One render target per view: Game is exactly what players see, Scene is a clean, lit editing view.

        While the window is being resized the last frame is stretched; targets are rebuilt once the size settles."""
        size, now = (max(int(size[0]), 16), max(int(size[1]), 16)), time.perf_counter()
        if size == self.size:
            return
        if self.renderer and not self.snapshot and (size != getattr(self, "_pending", None) or now - self._pending_t < settle):
            if size != getattr(self, "_pending", None):
                self._pending, self._pending_t = size, now
            return
        old, self.size = self.renderer, size
        self.targets = {v: gfx.Texture(dim=2, size=(*size, 1), format="rgba8unorm") for v in ("scene", "game")}
        self.renderers = {v: gfx.WgpuRenderer(t) for v, t in self.targets.items()}
        self.renderer, self.texs = self.renderers["game"], {}
        if old:  # carry the game's look (pixel ratio, filters, post effects) across resizes
            for attr in self.LOOK:
                setattr(self.renderer, attr, getattr(old, attr))
        if self.rt:
            self.rt.world.renderer = self.renderer

    def viewport_window(self, dt):
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
        if self.focus_viewport:
            imgui.set_next_window_focus()
            self.focus_viewport = False
        visible = imgui.begin(f"{I.CLAPPERBOARD}  Viewport###Viewport")[0]
        imgui.pop_style_var()
        self.viewport_hovered = self.viewport_focused = False
        if visible and not self.rt:
            self.welcome()
        elif visible:
            self.ensure_target(imgui.get_content_region_avail())
            origin = imgui.get_cursor_screen_pos()
            self.viewport_origin = (origin.x, origin.y)
            scene_view = self.view == "scene"
            renderer, ambient = self.renderers[self.view], self.rt.world.ambient
            if scene_view:
                self.editor_camera(dt)
                look = {a: getattr(self.renderer, a) for a in self.LOOK} if self.effects else dict(zip(self.LOOK, (1, "linear", "default", ())))
                for attr, value in look.items():  # only touch the renderer when the look actually changes
                    if getattr(renderer, attr) != value:
                        setattr(renderer, attr, value)
            self.grid.visible = scene_view and self.show_grid
            for k in self.rt.world.kinds.values():
                k.material.pick_write = True
            self.update_box(scene_view)
            game_ambient = ambient.intensity
            ambient.intensity = max(game_ambient, 1.4) if scene_view and not self.game_lighting else game_ambient
            self.rt.render(self.cam if scene_view else None, hud=not scene_view, renderer=renderer)
            ambient.intensity = game_ambient
            if self.view not in self.texs:
                self.texs[self.view] = self.gui.backend.register_texture(self.targets[self.view]._wgpu_default_view)
            imgui.image(self.texs[self.view], imgui.get_content_region_avail())
            self.viewport_hovered = imgui.is_item_hovered()
            self.viewport_focused = self.viewport_hovered or imgui.is_window_focused()
            if scene_view:
                self.gizmo(origin)
                self.pick(origin)
            self.viewport_overlay(origin)
        imgui.end()

    def viewport_overlay(self, origin):
        imgui.set_cursor_screen_pos(imgui.ImVec2(origin.x + self.size[0] - 214, origin.y + 8))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, imgui.ImVec2(2, 0))
        for view, icon, label in (("scene", I.ORBIT, "Scene"), ("game", I.GAMEPAD_2, "Game")):
            active = self.view == view
            imgui.push_style_color(imgui.Col_.button, hex4(theme.ACTIVE if active else theme.BG, 0.92))
            imgui.push_style_color(imgui.Col_.text, hex4(theme.ACCENT if active else theme.DIM))
            if imgui.button(f"{icon} {label}"):
                self.view = view
            imgui.pop_style_color(2)
            imgui.same_line()
        if self.view == "scene":
            for flag, icon, tip in (("effects", I.SPARKLES, "Preview post effects & pixel look"), ("game_lighting", I.SUN, "Preview game lighting")):
                imgui.same_line()
                if icon_button(icon, tip, getattr(self, flag), size=24):
                    setattr(self, flag, not getattr(self, flag))
        imgui.pop_style_var()
        if self.view == "scene":
            hint = "RMB + WASD fly · MMB pan · Wheel zoom · F frame"
            sz = imgui.calc_text_size(hint)
            imgui.get_window_draw_list().add_text(imgui.ImVec2(origin.x + self.size[0] - sz.x - 12, origin.y + self.size[1] - sz.y - 10),
                                                  u32(theme.DIM, 0.8), hint)

    def editor_camera(self, dt):
        io = imgui.get_io()
        forward = np.array([-np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), -np.cos(self.yaw) * np.cos(self.pitch)])
        right = np.array([np.cos(self.yaw), 0, -np.sin(self.yaw)])
        up = np.cross(right, forward)
        hovered = imgui.is_window_hovered()
        if imgui.is_mouse_down(imgui.MouseButton_.right) and (hovered or self.flying):
            self.flying = True
            self.yaw -= io.mouse_delta.x * 0.004
            self.pitch = float(np.clip(self.pitch - io.mouse_delta.y * 0.004, -1.5, 1.5))
            keys = lambda a, b: imgui.is_key_down(a) - imgui.is_key_down(b)
            move = forward * keys(imgui.Key.w, imgui.Key.s) + right * keys(imgui.Key.d, imgui.Key.a) + up * keys(imgui.Key.e, imgui.Key.q)
            self.speed = float(np.clip(self.speed * (1.2 ** io.mouse_wheel), 0.5, 200))
            self.cam_pos += move * self.speed * dt * (3 if io.key_shift else 1)
        else:
            self.flying = False
            if hovered and imgui.is_mouse_down(imgui.MouseButton_.middle):
                self.cam_pos += (-right * io.mouse_delta.x + up * io.mouse_delta.y) * 0.002 * self.speed
            elif hovered and io.mouse_wheel:
                self.cam_pos += forward * io.mouse_wheel * self.speed * 0.25
        self.cam.local.position = self.cam_pos
        self.cam.local.rotation = la.quat_mul(la.quat_from_axis_angle((0, 1, 0), self.yaw), la.quat_from_axis_angle((1, 0, 0), self.pitch))
        self.cam.fov = self.rt.world.camera.fov

    def frame_selected(self):
        if sel := self.selection():
            _, k, i = sel
            forward = np.array([-np.sin(self.yaw) * np.cos(self.pitch), np.sin(self.pitch), -np.cos(self.yaw) * np.cos(self.pitch)])
            self.cam_pos = k.pos[i] - forward * max(4.0, float(np.abs(k.scale[i]).max()) * 3)

    def update_box(self, scene_view):
        sel = self.selection()
        self.box.visible = bool(sel) and scene_view
        if sel:
            _, k, i = sel
            lo, hi = k.geometry.positions.data.min(0), k.geometry.positions.data.max(0)
            local = la.mat_compose((lo + hi) / 2, (0, 0, 0, 1), np.maximum(hi - lo, 1e-3) * 1.02)
            self.box.local.matrix = matrix(k.pos[i], k.rot[i], k.scale[i]) @ local

    def gizmo(self, origin):
        sel = self.selection()
        using = False
        if sel and self.mode == "edit":
            _, k, i = sel
            gz.set_drawlist()
            gz.set_rect(origin.x, origin.y, *self.size)
            view, proj = (gz.Matrix16(m.T.flatten().tolist()) for m in (self.cam.view_matrix, self.cam.projection_matrix))
            model = gz.Matrix16(matrix(k.pos[i], k.rot[i], k.scale[i]).T.flatten().tolist())
            snap = gz.Matrix3([15.0] * 3 if self.op == gz.OPERATION.rotate else [0.25] * 3 if self.op == gz.OPERATION.scale else [1.0] * 3)
            mode = gz.MODE.world if self.world_space else gz.MODE.local
            if gz.manipulate(view, proj, self.op, mode, model, None, snap if self.snap else None):
                t, q, s = la.mat_decompose(np.array(model.values).reshape(4, 4).T)
                k.pos[i], k.rot[i], k.scale[i] = t, q, s
            using = gz.is_using()
            if using and not self.was_using:
                self.checkpoint()
        self.was_using = using

    def pick(self, origin):
        io = imgui.get_io()
        if not (self.viewport_hovered and imgui.is_mouse_clicked(imgui.MouseButton_.left)) or gz.is_over() or gz.is_using():
            return
        info = self.renderers["scene"].get_pick_info((io.mouse_pos.x - origin.x, io.mouse_pos.y - origin.y))
        hit = info.get("world_object")
        self.selected = next(((n, int(info.get("instance_index") or 0)) for n, k in self.rt.world.kinds.items() if k.obj is hit), None)

    def welcome(self):
        avail, start = imgui.get_content_region_avail(), imgui.get_cursor_screen_pos()
        dl = imgui.get_window_draw_list()
        dl.add_rect_filled(start, imgui.ImVec2(start.x + avail.x, start.y + avail.y), u32(theme.BG))
        for i in range(0, int(avail.y), 4):  # faint CRT scanlines
            dl.add_line(imgui.ImVec2(start.x, start.y + i), imgui.ImVec2(start.x + avail.x, start.y + i), u32("#ffffff", 0.012))
        width = min(640, avail.x - 32)
        imgui.set_cursor_screen_pos(imgui.ImVec2(start.x + (avail.x - width) / 2, start.y + max(avail.y * 0.1, 24)))
        imgui.begin_group()
        imgui.push_font(Fonts.title, 44)
        imgui.text_colored(hex4(theme.ACCENT), "LATTICE")
        imgui.pop_font()
        imgui.text_colored(hex4(theme.DIM), "Games are Python files. Worlds are arrays. Everything is live.")
        imgui.dummy(imgui.ImVec2(0, 16))
        card = imgui.ImVec2(width / 2 - 4, 64)
        if imgui.button(f"{I.FILE_PLUS}   New Game", card):
            self.new_game()
        imgui.same_line()
        if imgui.button(f"{I.FOLDER_OPEN}   Open Game", card):
            self.open_dialog()
        imgui.dummy(imgui.ImVec2(0, 16))
        imgui.separator_text("Examples")
        examples = (("megacity.py", I.BUILDING_2, "Neo-Tokyo", "Rain, neon and 13,000 lights."),
                    ("meadow.py", I.MOUNTAIN, "Meadow", "A lush valley from one seed."),
                    ("liminal.py", I.GHOST, "Fluorescent", "PS2-era liminal horror."),
                    ("galaxy.py", I.SPARKLES, "Galaxy", "30,000 entities, 3 draw calls."))
        for i, (file, icon, title, blurb) in enumerate(e for e in examples if (EXAMPLES / e[0]).exists()):
            imgui.same_line() if i % 2 else None
            if self.card(file, icon, title, blurb, width / 2 - 4):
                self.open(EXAMPLES / file)
        if self.recent:
            imgui.dummy(imgui.ImVec2(0, 8))
            imgui.separator_text("Recent")
            for p in self.recent[:5]:
                icon = I.CAMERA if p.lower().endswith(".png") else I.FILE_CODE
                if imgui.selectable(f"{icon}   {Path(p).name}   ###{p}", False)[0]:
                    self.open(p)
                imgui.same_line(), imgui.text_colored(hex4(theme.FAINT), str(Path(p).parent))
        imgui.end_group()

    def card(self, key, icon, title, blurb, width):
        """A two-line launcher card: accent icon, bold title, dimmed description."""
        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.selectable(f"##{key}", False, 0, imgui.ImVec2(width, 46))[0]
        dl = imgui.get_window_draw_list()
        dl.add_text(imgui.ImVec2(pos.x + 12, pos.y + 13), u32(theme.ACCENT), icon)
        dl.add_text(Fonts.bold, 15, imgui.ImVec2(pos.x + 40, pos.y + 4), u32(theme.TEXT), title)
        dl.add_text(imgui.ImVec2(pos.x + 40, pos.y + 24), u32(theme.DIM), blurb)
        return clicked

    # ------------------------------------------------------------------ panels
    def hierarchy_window(self):
        if imgui.begin(f"{I.LAYERS}  Hierarchy###Hierarchy", None, PANEL)[0] and self.rt:
            imgui.set_next_item_width(-1)
            self.filter = imgui.input_text_with_hint("##filter", f"{I.SEARCH}  Search kinds", getattr(self, "filter", ""))[1]
            imgui.dummy(imgui.ImVec2(0, 2))
            for name, k in self.rt.world.kinds.items():
                if self.filter.lower() not in name.lower():
                    continue
                flags = imgui.TreeNodeFlags_.span_avail_width | imgui.TreeNodeFlags_.open_on_arrow
                if self.selected and self.selected[0] == name and self.selected[1] < 0:
                    flags |= imgui.TreeNodeFlags_.selected
                is_open = imgui.tree_node_ex(f"{I.BOXES}  {name}##kind", flags)
                if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
                    self.selected = (name, -1)
                imgui.same_line(imgui.get_window_width() - imgui.calc_text_size(f"{k.n:,}").x - 16)
                imgui.text_colored(hex4(theme.FAINT), f"{k.n:,}")
                if is_open:
                    clipper = imgui.ListClipper()
                    clipper.begin(k.n)
                    while clipper.step():
                        for i in range(clipper.display_start, clipper.display_end):
                            sel = self.selected == (name, i)
                            if imgui.selectable(f"   {I.BOX}  {name} {i}##{name}{i}", sel)[0]:
                                self.selected = (name, i)
                    imgui.tree_pop()
        imgui.end()

    def inspector_window(self):
        if imgui.begin(f"{I.SLIDERS_HORIZONTAL}  Inspector###Inspector", None, PANEL)[0]:
            sel = self.selection()
            kind = self.rt.world.kinds.get(self.selected[0]) if self.rt and self.selected else None
            if sel:
                self.entity_inspector(*sel)
            elif kind is not None:
                self.kind_inspector(self.selected[0], kind)
            elif self.rt:
                self.render_inspector()
        imgui.end()

    def render_inspector(self):
        """With nothing selected the Inspector edits the game's look: resolution, filtering and the post-effect stack."""
        r = self.renderer
        self.header(I.APERTURE, "Rendering", "nothing selected")
        imgui.text_colored(hex4(theme.DIM), "Pixel size")
        imgui.same_line(96), imgui.set_next_item_width(-1)
        changed, ratio = imgui.slider_float("##ratio", float(r.pixel_ratio), 0.1, 1.0, "%.2f")
        r.pixel_ratio = ratio if changed else r.pixel_ratio
        if imgui.checkbox("Crisp pixels (nearest)", r.pixel_filter == "nearest")[0]:
            r.pixel_filter = "linear" if r.pixel_filter == "nearest" else "nearest"
        imgui.text_colored(hex4(theme.DIM), "Style"), imgui.same_line(96), imgui.set_next_item_width(-1)
        if imgui.begin_combo("##style", getattr(self.rt.world, "style", "custom")):
            for name in fx.STYLES:
                if imgui.selectable(name, False)[0]:
                    fx.style(self.rt.world, name)
                    imgui.set_clipboard_text(f'fx.style(w, "{name}")')
                    self.toast(I.PAINTBRUSH, f'Style "{name}" · code copied, paste it into scene() to keep it')
            imgui.end_combo()
        imgui.dummy(imgui.ImVec2(0, 4))
        imgui.separator_text(f"{I.WAND_SPARKLES}  Post effects")
        aa = [p for p in r.effect_passes if isinstance(p, fx.PPAAPass)]
        passes = [p for p in r.effect_passes if not isinstance(p, fx.PPAAPass)]
        for i, p in enumerate(passes):
            imgui.push_id(i)
            p.enabled = imgui.checkbox(f"##on", p.enabled)[1]
            imgui.same_line()
            name = getattr(p, "preset", type(p).__name__.replace("Pass", ""))
            if imgui.tree_node_ex(f"{name}###fx", imgui.TreeNodeFlags_.span_avail_width):
                for u in p._uniform_data.dtype.names:
                    if u in ("time", "t") or u.startswith("_"):
                        continue
                    v = p._uniform_data[u]
                    imgui.text_colored(hex4(theme.DIM), u.replace("_", " ")[:11]), imgui.same_line(96), imgui.set_next_item_width(-1)
                    if "color" in u or u == "tint":
                        changed, out = imgui.color_edit4(f"##{u}", list(map(float, v)), imgui.ColorEditFlags_.float)
                    elif np.size(v) == 1:
                        changed, out = imgui.drag_float(f"##{u}", float(v), max(abs(float(v)), 0.01) * 0.01)
                    else:
                        changed, out = getattr(imgui, f"drag_float{np.size(v)}")(f"##{u}", list(map(float, v)), 0.01)
                    if changed:
                        p._uniform_data[u] = out
                imgui.tree_pop()
            imgui.same_line(imgui.get_window_width() - 36)
            if imgui.small_button(I.X):
                passes.remove(p)
            imgui.pop_id()
        imgui.set_next_item_width(-1)
        if imgui.begin_combo("##add", f"{I.PLUS}  Add effect"):
            for name, make in fx.PRESETS.items():
                if imgui.selectable(name, False)[0]:
                    (p := make()).preset = name
                    passes.append(p)
            imgui.end_combo()
        if passes + aa != list(r.effect_passes):
            r.effect_passes = passes + aa
        imgui.text_colored(hex4(theme.FAINT), "Live preview. Keep it in scene():\nw.renderer.effect_passes = [fx.bloom(), ...]")

    def header(self, icon, title, sub):
        imgui.push_font(Fonts.bold, 0)
        imgui.text(f"{icon}  {title}")
        imgui.pop_font()
        imgui.same_line(), imgui.text_colored(hex4(theme.FAINT), sub)
        imgui.dummy(imgui.ImVec2(0, 4))

    def field(self, label, value, speed=0.05):
        """A labelled drag field for a column value (scalar or vector). Returns the new value or None."""
        imgui.text_colored(hex4(theme.DIM), label)
        imgui.same_line(96)
        imgui.set_next_item_width(-1)
        v = np.atleast_1d(value).astype(float)
        fn = {1: imgui.drag_float, 2: imgui.drag_float2, 3: imgui.drag_float3, 4: imgui.drag_float4}[len(v)]
        changed, out = fn(f"##{label}", float(v[0]) if len(v) == 1 else v.tolist(), speed, 0, 0, "%.3f")
        if imgui.is_item_activated():
            self.checkpoint()
        return out if changed else None

    def entity_inspector(self, name, k, i):
        self.header(I.BOX, name, f"#{i}")
        if imgui.collapsing_header(f"{I.MOVE}  Transform", imgui.TreeNodeFlags_.default_open):
            if (v := self.field("Position", k.pos[i])) is not None:
                k.pos[i] = v
            euler = np.degrees(la.quat_to_euler(k.rot[i] / np.linalg.norm(k.rot[i])))
            if (v := self.field("Rotation", euler, 0.5)) is not None:
                k.rot[i] = la.quat_from_euler(np.radians(v))
            if (v := self.field("Scale", k.scale[i], 0.02)) is not None:
                k.scale[i] = v
        extra = [c for c in k._cols if c not in ("pos", "rot", "scale")]
        if extra and imgui.collapsing_header(f"{I.CPU}  Data", imgui.TreeNodeFlags_.default_open):
            for c in extra:
                if c == "tint":
                    imgui.text_colored(hex4(theme.DIM), "tint"), imgui.same_line(96), imgui.set_next_item_width(-1)
                    changed, rgb = imgui.color_edit3("##tint", k.tint[i].tolist(), imgui.ColorEditFlags_.float)
                    k.tint[i] = rgb if changed else k.tint[i]
                elif (v := self.field(c, k._cols[c][i])) is not None:
                    k._cols[c][i] = v
        self.material_inspector(name, k)
        imgui.dummy(imgui.ImVec2(0, 8))
        if imgui.button(f"{I.COPY}  Duplicate", imgui.ImVec2(imgui.get_content_region_avail().x / 2 - 4, 0)):
            self.duplicate()
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.text, hex4(theme.ERROR))
        if imgui.button(f"{I.TRASH_2}  Delete", imgui.ImVec2(-1, 0)):
            self.delete()
        imgui.pop_style_color()

    def kind_inspector(self, name, k):
        self.header(I.BOXES, name, f"{k.n:,} entities")
        imgui.text_colored(hex4(theme.DIM), "Columns")
        for c, col in k._cols.items():
            imgui.bullet_text(f"{c}  {col.shape[1:] or 'scalar'}")
        self.material_inspector(name, k)
        imgui.dummy(imgui.ImVec2(0, 8))
        if imgui.button(f"{I.PLUS}  Spawn one", imgui.ImVec2(-1, 0)) and self.mode == "edit":
            self.checkpoint()
            forward = np.array([-np.sin(self.yaw), 0, -np.cos(self.yaw)])
            self.selected = (name, int(k.spawn(pos=self.cam_pos + forward * 6)[0]))

    def material_inspector(self, name, k):
        m = k.material
        if not imgui.collapsing_header(f"{I.PALETTE}  Material", imgui.TreeNodeFlags_.default_open):
            return
        names = {cls: name for name, cls in MATERIALS.items()}
        imgui.text_colored(hex4(theme.DIM), "Shading"), imgui.same_line(96), imgui.set_next_item_width(-1)
        if imgui.begin_combo("##shading", names.get(type(m), type(m).__name__)):
            for name, cls in MATERIALS.items():
                if imgui.selectable(name, type(m) is cls)[0] and type(m) is not cls:
                    k.material = k.obj.material = m = convert(m, cls)
            imgui.end_combo()
        for attr in ("color", "emissive"):
            if hasattr(m, attr):
                imgui.text_colored(hex4(theme.DIM), attr.title())
                imgui.same_line(96), imgui.set_next_item_width(-1)
                changed, rgb = imgui.color_edit3(f"##{attr}", list(getattr(m, attr).rgb), imgui.ColorEditFlags_.display_hex)
                if changed:
                    setattr(m, attr, rgb)
        for attr in ("roughness", "metalness", "emissive_intensity"):
            if hasattr(m, attr):
                imgui.text_colored(hex4(theme.DIM), attr.replace("_", " ").title()[:11])
                imgui.same_line(96), imgui.set_next_item_width(-1)
                changed, v = imgui.slider_float(f"##{attr}", float(getattr(m, attr)), 0, 1 if attr != "emissive_intensity" else 4)
                if changed:
                    setattr(m, attr, v)
        imgui.text_colored(hex4(theme.FAINT), "Live preview. Keep it by editing scene().")
        if imgui.small_button(f"{I.CODE}  Copy as code"):
            imgui.set_clipboard_text(f'w.kind("{name}", ..., color="{m.color.hex}")')
            self.toast(I.CHECK, "Copied to clipboard", theme.OK)

    def project_window(self):
        if imgui.begin(f"{I.FOLDER}  Project###Project", None, PANEL)[0]:
            folder = self.rt.path.parent if self.rt else EXAMPLES
            if icon_button(I.FILE_PLUS, "New game", size=26):
                self.new_game()
            imgui.same_line()
            if icon_button(I.FOLDER_OPEN, "Show in Explorer", size=26) and os.name == "nt":
                os.startfile(folder)
            imgui.same_line(), imgui.text_colored(hex4(theme.FAINT), str(folder))
            imgui.separator()
            for f in sorted(folder.iterdir()) if folder.exists() else ():
                if f.name.startswith((".", "__")):
                    continue
                icon = I.FILE_CODE if f.suffix == ".py" else I.PACKAGE if f.suffix == ".npz" else I.FILE
                current = self.rt and f == self.rt.path
                imgui.push_style_color(imgui.Col_.text, hex4(theme.ACCENT if current else theme.TEXT))
                if imgui.selectable(f"{icon}  {f.name}", bool(current), imgui.SelectableFlags_.allow_double_click)[0]:
                    if imgui.is_mouse_double_clicked(imgui.MouseButton_.left) and f.suffix == ".py":
                        self.edit_script(f)
                imgui.pop_style_color()
                if f.suffix == ".py" and imgui.begin_popup_context_item(f"ctx{f.name}"):
                    if imgui.menu_item(f"{I.PLAY}  Open as game", "", False)[0]:
                        self.open(f)
                    if imgui.menu_item(f"{I.CODE}  Edit script", "", False)[0]:
                        self.edit_script(f)
                    imgui.end_popup()
        imgui.end()

    def console_window(self):
        if imgui.begin(f"{I.TERMINAL}  Console###Console", None, PANEL)[0]:
            c = self.console
            counts = {lvl: sum(1 for l, *_ in c.lines if l == lvl) for lvl in c.filters}
            if imgui.small_button(f"{I.TRASH_2} Clear"):
                c.lines.clear()
            for lvl, icon, color in (("info", I.INFO, theme.INFO), ("warn", I.TRIANGLE_ALERT, theme.WARN), ("error", I.CIRCLE_ALERT, theme.ERROR)):
                imgui.same_line()
                imgui.push_style_color(imgui.Col_.text, hex4(color if c.filters[lvl] else theme.FAINT))
                if imgui.small_button(f"{icon} {counts[lvl]}##{lvl}"):
                    c.filters[lvl] = not c.filters[lvl]
                imgui.pop_style_color()
            imgui.separator()
            imgui.begin_child("log")
            imgui.push_font(Fonts.mono, 0)
            colors = {"info": theme.TEXT, "warn": theme.WARN, "error": theme.ERROR}
            for level, stamp, line in c.lines:
                if c.filters[level]:
                    imgui.text_colored(hex4(theme.FAINT), f"{stamp:>8}"), imgui.same_line()
                    imgui.text_colored(hex4(colors[level]), line.replace("*", ""))
            if imgui.get_scroll_y() >= imgui.get_scroll_max_y() - 4:
                imgui.set_scroll_here_y(1.0)
            imgui.pop_font()
            imgui.end_child()
        imgui.end()

    def edit_script(self, path):
        path = Path(path)
        if path not in self.scripts:
            te = ed.TextEditor()
            te.set_language(ed.TextEditor.Language.python())
            te.set_palette(ed.TextEditor.get_dark_palette())
            te.set_tab_size(4)
            te.set_show_whitespaces_enabled(False)
            te.set_text(path.read_text(encoding="utf-8"))
            self.scripts[path] = [te, te.get_text()]
        self.script_tab = path

    def save_script(self):
        if self.script_tab in self.scripts:
            te, _ = self.scripts[self.script_tab]
            text = te.get_text()
            self.script_tab.write_text(text, encoding="utf-8")
            self.scripts[self.script_tab][1] = text
            self.toast(I.SAVE, f"Saved {self.script_tab.name} · hot reloading", theme.OK)

    def script_window(self):
        self.script_focused = False
        if imgui.begin(f"{I.CODE}  Script###Script", None, PANEL)[0]:
            if not self.scripts:
                imgui.text_colored(hex4(theme.FAINT), "Double-click a .py file in Project to edit it here.")
            elif imgui.begin_tab_bar("scripts", imgui.TabBarFlags_.auto_select_new_tabs | imgui.TabBarFlags_.fitting_policy_scroll):
                for path, (te, saved) in list(self.scripts.items()):
                    dirty = te.get_text() != saved if path == self.script_tab else False
                    flags = imgui.TabItemFlags_.unsaved_document if dirty else 0
                    flags |= imgui.TabItemFlags_.set_selected if getattr(self, "_focus_tab", None) == path else 0
                    opened, keep = imgui.begin_tab_item(f"{I.FILE_CODE} {path.name}###{path}", True, flags)
                    if opened:
                        self.script_tab = path
                        self.error_markers(path, te)
                        imgui.push_font(Fonts.mono, 0)
                        te.render(f"##editor{path}")
                        imgui.pop_font()
                        self.script_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
                        imgui.end_tab_item()
                    if not keep:
                        del self.scripts[path]
                self._focus_tab = None
                imgui.end_tab_bar()
        imgui.end()

    def error_markers(self, path, te):
        key = (path, self.rt.error if self.rt else None)
        if getattr(self, "_markers", None) == key:
            return
        self._markers = key
        te.clear_markers()
        if self.rt and self.rt.error:
            for line in re.findall(rf'File "{re.escape(str(path))}", line (\d+)', self.rt.error, re.I):
                te.add_marker(int(line) - 1, u32(theme.ERROR), u32(theme.ERROR, 0.18), "Error", self.rt.error.splitlines()[-1])

    def draw_toasts(self):
        now, vp = time.perf_counter(), imgui.get_main_viewport()
        self.toasts = [t for t in self.toasts if now - t[3] < 3.2]
        dl, y = imgui.get_foreground_draw_list(), vp.pos.y + vp.size.y - 48
        for icon, text, color, t0 in reversed(self.toasts):
            age = now - t0
            alpha = min(1, age * 6, (3.2 - age) * 3)
            label = f"{icon}   {text}"
            size = imgui.calc_text_size(label)
            x = vp.pos.x + vp.size.x - size.x - 40
            dl.add_rect_filled(imgui.ImVec2(x, y - size.y - 16), imgui.ImVec2(x + size.x + 24, y), u32(theme.RAISED, 0.96 * alpha), 6)
            dl.add_rect_filled(imgui.ImVec2(x, y - size.y - 16), imgui.ImVec2(x + 3, y), u32(color, alpha), 2)
            dl.add_text(imgui.ImVec2(x + 14, y - size.y - 8), u32(theme.TEXT, alpha), label)
            y -= size.y + 24
