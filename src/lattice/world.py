"""Data-oriented world: each entity *kind* is a table of NumPy columns drawn as one GPU-instanced mesh."""

import numpy as np
import pygfx as gfx

BASE = dict(pos=(0.0, 0, 0), vel=(0.0, 0, 0), rot=(0.0, 0, 0, 1), scale=(1.0, 1, 1))


class Kind:
    """A table of entities sharing a mesh and material. Columns are live views: ``k.pos += k.vel * w.dt``."""

    def __init__(self, scene, geometry, material, columns):
        self.scene, self.n, self._cols, self._defaults, self.obj = scene, 0, {}, {}, None
        self.look(geometry, material, columns)

    def __getattr__(self, name):
        if name in (cols := self.__dict__.get("_cols", {})):
            return cols[name][: self.n]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        col = self.__dict__.get("_cols", {}).get(name)
        if col is None:
            object.__setattr__(self, name, value)
        elif not (isinstance(value, np.ndarray) and value.base is col):  # skip `k.pos += ...` self-assignment
            col[: self.n] = value

    def __len__(self):
        return self.n

    def look(self, geometry, material, columns):
        """(Re)declare appearance and columns; existing entity data is kept (this is what makes hot reload work)."""
        self.geometry, self.material = geometry, material
        for name, default in {**BASE, **columns}.items():
            self._defaults[name] = np.asarray(default, np.float32)
            self._cols.setdefault(name, None)
        self._grow(self._cap or 64)

    @property
    def _cap(self):
        return self.obj.instance_buffer.nitems if self.obj else 0

    def _grow(self, cap):
        old = self.obj
        for name, col in self._cols.items():
            d = self._defaults[name]
            self._cols[name] = new = np.broadcast_to(d, (cap, *d.shape)).copy()
            if col is not None:
                new[: self.n] = col[: self.n]
        self.obj = gfx.InstancedMesh(self.geometry, self.material, cap, cast_shadow=True, receive_shadow=True)
        self.scene.remove(old) if old else None
        self.scene.add(self.obj)

    def spawn(self, count=1, **values):
        """Append ``count`` entities. Values broadcast, e.g. ``pos=np.random.randn(count, 3)``. Returns their indices."""
        if self.n + count > self._cap:
            self._grow(max(self._cap * 2, self.n + count))
        idx = slice(self.n, self.n + count)
        self.n += count
        for name, col in self._cols.items():
            col[idx] = values.get(name, self._defaults[name])
        return np.arange(idx.start, idx.stop)

    def kill(self, which):
        """Remove entities by boolean mask or index array (order of survivors is preserved)."""
        keep = np.ones(self.n, bool)
        keep[which] = False
        k = int(keep.sum())
        for col in self._cols.values():
            col[:k] = col[: self.n][keep]
        self.n = k

    def clear(self):
        self.n = 0

    def sync(self):
        """Compose all transforms in one vectorized pass and hand them to the GPU instance buffer."""
        n, buf = self.n, self.obj.instance_buffer
        self.obj.visible = n > 0
        if not n:
            return
        q = self.rot / np.linalg.norm(self.rot, axis=1, keepdims=True)
        x, y, z, w = q.T
        r = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(n, 3, 3)
        m = buf.data["matrix"]  # stored column-major, i.e. transposed
        m[:n, :3, :3] = (r * self.scale[:, None, :]).transpose(0, 2, 1)
        m[:n, 3, :3] = self.pos
        buf.draw_range = (0, n)
        buf.update_range(0, n)


class World:
    """Everything a game file touches: kinds (``w.enemy``), input, time, camera, HUD and the raw pygfx scene."""

    def __init__(self, scene, camera, renderer):
        self.scene, self.camera, self.renderer = scene, camera, renderer
        self.kinds, self.t, self.dt, self.hud = {}, 0.0, 0.0, ""
        self.keys, self.pressed, self.buttons, self.clicked, self.mouse = set(), set(), set(), set(), (0.0, 0.0)
        self.mouse_delta = (0.0, 0.0)
        self._controller = None

    def __getattr__(self, name):
        if name in (kinds := self.__dict__.get("kinds", {})):
            return kinds[name]
        raise AttributeError(name)

    def kind(self, name, geometry, color="#ffffff", material=None, **columns):
        """Declare (or live-update) an entity kind with extra columns given as defaults, e.g. ``hp=3.0``."""
        material = material or gfx.MeshStandardMaterial(color=color)
        if name in self.kinds:
            self.kinds[name].look(geometry, material, columns)
        else:
            self.kinds[name] = Kind(self.scene, geometry, material, columns)
        return self.kinds[name]

    def orbit(self, target=(0, 0, 0)):
        """Mouse orbit/zoom camera control (created once, survives reloads)."""
        self._controller = self._controller or gfx.OrbitController(self.camera, register_events=self.renderer)
        self._controller.target = target

    def capture_mouse(self, on=True):
        """Hide and lock the cursor for mouse-look; ``w.mouse_delta`` keeps reporting motion."""
        if window := getattr(self.renderer.target, "_window", None):
            import glfw

            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED if on else glfw.CURSOR_NORMAL)

    def mouse_ray(self):
        """World-space (origin, direction) of the ray under the mouse cursor."""
        w, h = self.renderer.logical_size
        ndc = np.array([self.mouse[0] / w * 2 - 1, 1 - self.mouse[1] / h * 2])
        inv = np.linalg.inv(self.camera.camera_matrix)
        a, b = (inv @ [*ndc, z, 1] for z in (0, 1))
        a, b = a[:3] / a[3], b[:3] / b[3]
        return a, (b - a) / np.linalg.norm(b - a)

    def mouse_ground(self, height=0.0):
        """Point where the mouse ray hits the horizontal plane ``y = height``."""
        o, d = self.mouse_ray()
        return o + d * ((height - o[1]) / (d[1] if abs(d[1]) > 1e-6 else -1e-6))
