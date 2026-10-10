"""Data-oriented world: each entity *kind* is a table of NumPy columns drawn as one GPU-instanced mesh."""

import copy
from pathlib import Path

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu import load_wgsl
from pygfx.renderers.wgpu.shaders import meshshader

from . import audio, collide


def _copy(value):  # plain data is copied; GPU objects and the like are shared
    return copy.deepcopy(value) if isinstance(value, (np.ndarray, list, dict, set, tuple)) else value

MATERIALS = dict(basic=gfx.MeshBasicMaterial, phong=gfx.MeshPhongMaterial, toon=gfx.MeshToonMaterial,
                 standard=gfx.MeshStandardMaterial, physical=gfx.MeshPhysicalMaterial)
_assets = {}
KEEP = ("color", "map", "emissive", "emissive_map", "emissive_intensity", "opacity", "color_mode", "flat_shading", "side")


def convert(material, cls):
    """The same surface under another shading model (e.g. standard -> toon), keeping colours, maps and opacity."""
    new = cls()
    for attr in KEEP:
        try:
            setattr(new, attr, getattr(material, attr))
        except (AttributeError, TypeError, ValueError):
            pass
    return new


BASE = dict(pos=(0.0, 0, 0), vel=(0.0, 0, 0), rot=(0.0, 0, 0, 1), scale=(1.0, 1, 1), tint=(1.0, 1, 1))
TINT = {  # anchor line in pygfx's mesh.wgsl -> code inserted after it
    "    global_id: u32,": "shade_r: f32, shade_g: f32, shade_b: f32,",
    "        let pick_id = instance_info.global_id;":
        "varyings.tint = vec3<f32>(1.0 - instance_info.shade_r, 1.0 - instance_info.shade_g, 1.0 - instance_info.shade_b);",
    "    var emissive_color = srgb2physical(u_material.emissive_color.rgb) * u_material.emissive_intensity;":
        "$$ if instanced\nemissive_color *= varyings.tint;\n$$ endif",
    "    do_alpha_test(diffuse_color.a);": "$$ if instanced\ndiffuse_color = vec4<f32>(diffuse_color.rgb * varyings.tint, diffuse_color.a);\n$$ endif",
}


def _tinted_wgsl(name, *args):
    """Per-instance colour for every pygfx mesh material. The tint lives in the instance struct's 12 padding bytes, stored as
    (1 - tint) so untouched instances stay white: one kind, any number of colours, still one draw call."""
    code = load_wgsl(name, *args)
    for anchor, extra in TINT.items() if name == "mesh.wgsl" else ():
        code = code.replace(anchor, f"{anchor}\n{extra}")
    return code


meshshader.load_wgsl = _tinted_wgsl


class Kind:
    """A table of entities sharing a mesh and material. Columns are live views: ``k.pos += k.vel * w.dt``."""

    def __init__(self, scene, geometry, material, columns, shadow=True):
        self.scene, self.n, self._cols, self._defaults, self.obj = scene, 0, {}, {}, None
        self.look(geometry, material, columns, shadow)

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

    def look(self, geometry, material, columns, shadow=True):
        """(Re)declare appearance and columns; existing entity data is kept (this is what makes hot reload work)."""
        self.geometry, self.material, self.shadow, self._bounds = geometry, material, shadow, None
        for name, default in {**BASE, **columns}.items():
            self._defaults[name] = np.asarray(default, np.float32)
            self._cols.setdefault(name, None)
        self._grow(self._cap or 64)

    @property
    def bounds(self):
        """The geometry's local (min, max) corners: each instance's collision box before its pos/rot/scale."""
        if self._bounds is None:
            p = self.geometry.positions.data
            self._bounds = (p.min(0).astype(float), p.max(0).astype(float))
        return self._bounds

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
        self.obj = gfx.InstancedMesh(self.geometry, self.material, cap, cast_shadow=self.shadow, receive_shadow=True)
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
        r, m = collide.rotations(self.rot), buf.data["matrix"]  # stored column-major, i.e. transposed
        m[:n, :3, :3] = (r * self.scale[:, None, :]).transpose(0, 2, 1)
        m[:n, 3, :3] = self.pos
        buf.data["_12_bytes_padding"][:n].view(np.float32)[:] = 1 - np.clip(self.tint, 0, None) ** 2.2  # sRGB in, linear out
        buf.draw_range = (0, n)
        buf.update_range(0, n)


class World:
    """Everything a game file touches: kinds (``w.enemy``), input, time, camera, HUD and the raw pygfx scene."""

    def __init__(self, scene, camera, renderer):
        self.scene, self.camera, self.renderer = scene, camera, renderer
        self.kinds, self.t, self.dt, self.hud = {}, 0.0, 0.0, ""
        self.keys, self.pressed, self.buttons, self.clicked, self.mouse = set(), set(), set(), set(), (0.0, 0.0)
        self.mouse_delta = (0.0, 0.0)
        self._controller, self.root, self.files, self.canvas = None, Path.cwd(), set(), None
        self._engine = set(vars(self)) | {"_engine"}  # everything set later is game state (the Runtime extends this)

    def __getattr__(self, name):
        if name in (kinds := self.__dict__.get("kinds", {})):
            return kinds[name]
        raise AttributeError(name)

    def kind(self, name, geometry, color="#ffffff", material=None, shadow=True, **columns):
        """Declare (or live-update) an entity kind with extra columns given as defaults, e.g. ``hp=3.0``.
        ``shadow=False`` stops it casting shadows (clouds, particles, glowing things)."""
        material = self.material(material or "standard", color=color) if isinstance(material, (str, type(None))) else material
        if name in self.kinds:
            self.kinds[name].look(geometry, material, columns, shadow)
        else:
            self.kinds[name] = Kind(self.scene, geometry, material, columns, shadow)
        return self.kinds[name]

    def asset(self, path, load, *variant):
        """Load a file relative to the game once (reloaded when it changes on disk). Used files travel with screenshots."""
        path = (self.root / path).resolve()
        self.files.add(path)
        key = (path, path.stat().st_mtime, *variant)
        if key not in _assets:
            _assets[key] = load(path)
        return _assets[key]

    def texture(self, image, filter="linear", srgb=True):
        """An image file (or an (h, w, 4) uint8 array) as a texture map; ``filter="nearest"`` keeps texels crisp for a retro look."""
        from PIL import Image

        make = lambda a: gfx.Texture(np.ascontiguousarray(a), dim=2, colorspace="srgb" if srgb else "physical", generate_mipmaps=True)
        tex = make(image) if isinstance(image, np.ndarray) else self.asset(image, lambda p: make(Image.open(p).convert("RGBA")), srgb)
        return gfx.TextureMap(tex, filter=filter)

    def material(self, type="standard", filter="linear", **props):
        """A material by name: basic (unlit), phong, toon, standard (PBR) or physical. Any ``map``/``*_map`` given as a
        file path is loaded as a texture: ``w.material("toon", color="#fff", map="brick.png", filter="nearest")``."""
        for key, value in props.items():
            if key.endswith("map") and isinstance(value, (str, Path, np.ndarray)):
                props[key] = self.texture(value, filter, srgb=key in ("map", "emissive_map"))
        return MATERIALS[type](**props)

    def model(self, path):
        """Geometry and material of the first mesh in a glTF/GLB file: ``w.kind("crate", **w.model("crate.glb"))``."""
        mesh = self.asset(path, lambda p: gfx.load_gltf_mesh(str(p), quiet=True)[0])
        return dict(geometry=mesh.geometry, material=mesh.material)

    def play(self, sound, volume=1.0, loop=False, key=None):
        """Play a sound file, or a NumPy sample array (mono or stereo, 44.1 kHz). Returns a voice with live ``.volume``
        and ``.stop()``. With a ``key`` it starts once and later calls return the same voice, so it is safe every frame."""
        if key and (v := audio.mixer.keyed.get(key)) and v.playing:
            return v
        data = self.asset(sound, audio.load) if isinstance(sound, (str, Path)) else audio.as_stereo(sound)
        return audio.mixer.play(data, volume, loop, key)

    def snapshot(self):
        """Copy all entity tables plus game-defined attributes. Powers Play/Stop, undo and scene files."""
        tables = {n: {c: col[: k.n].copy() for c, col in k._cols.items()} for n, k in self.kinds.items()}
        return tables, {key: _copy(v) for key, v in vars(self).items() if key not in self.__dict__.get("_engine", ())}

    def restore(self, snapshot):
        tables, attrs = snapshot
        for name, cols in tables.items():
            if (k := self.kinds.get(name)) is not None:  # an empty kind is falsy (len 0) but must still restore
                k.clear()
                k.spawn(len(cols["pos"]), **cols)
        vars(self).update({key: _copy(v) for key, v in attrs.items()})

    def save(self, path):
        np.savez_compressed(path, **{f"{n}.{c}": a for n, cols in self.snapshot()[0].items() for c, a in cols.items()})

    def load(self, path):
        tables = {}
        with np.load(path) as f:
            for key in f.files:
                kind, col = key.split(".", 1)
                tables.setdefault(kind, {})[col] = f[key]
        self.restore((tables, {}))

    # collision: kinds may be given as objects or names
    def _kind(self, k):
        return self.kinds[k] if isinstance(k, str) else k

    def raycast(self, origin, direction, kinds=None, max_dist=np.inf):
        """The nearest hit along a ray against the oriented boxes of ``kinds`` (default: all), or None. A hit has
        ``.kind``, ``.index``, ``.dist``, ``.point`` and ``.normal``. Mouse picking: ``w.raycast(*w.mouse_ray())``."""
        names = kinds if kinds is not None else self.kinds
        names = [names] if isinstance(names, str) else names
        return collide.raycast({n: self.kinds[n] for n in names}, origin, direction, max_dist)

    def overlaps(self, a, b, radius_a=None, radius_b=None):
        """Index arrays ``(i, j)`` of touching pairs between two kinds, as spheres (radius defaults to each box's largest
        half extent): ``i, j = w.overlaps("bullet", "enemy"); w.enemy.hp[j] -= 1; w.bullet.kill(i)``."""
        return collide.overlaps(self._kind(a), self._kind(b), radius_a, radius_b)

    def push_out(self, mover, walls, radius=0.5):
        """Slide spheres out of solid oriented boxes. ``mover`` is a kind (moved in place) or an (n, 3) / (3,) array (a
        moved copy is returned). Returns ``(positions, normals)``; a normal is zero where nothing was touched and points
        away from the wall otherwise, so ``normals[:, 1] > 0.7`` means standing on something."""
        k = self._kind(mover)
        points = k.pos if isinstance(k, Kind) else np.asarray(k, float)
        pos, normal = collide.push_out(points, radius, self._kind(walls))
        if isinstance(k, Kind):
            k.pos = pos
        return (pos, normal) if points.ndim == 2 else (pos[0], normal[0])

    def orbit(self, target=(0, 0, 0)):
        """Mouse orbit/zoom camera control (created once, survives reloads)."""
        self._controller = self._controller or gfx.OrbitController(self.camera, register_events=self.renderer)
        self._controller.target = target

    def capture_mouse(self, on=True):
        """Hide and lock the cursor for mouse-look; ``w.mouse_delta`` keeps reporting motion."""
        if window := getattr(self.canvas or self.renderer.target, "_window", None):
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
