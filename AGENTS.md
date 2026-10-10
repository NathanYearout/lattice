# Building games with Lattice: a guide for agents

A game is **one Python file** with up to three hooks. The engine hot-reloads it on save.

```python
def scene(w): ...   # declare kinds (mesh + material + columns) and renderer look. Re-run on EVERY save.
def setup(w): ...   # spawn the starting entities, place the camera, set game state. Run once (and on Restart).
def update(w): ...  # per-frame logic. Only runs while playing.
```

## The world `w`
- `w.kind(name, geometry, color="#hex", material=None, shadow=True, **columns)` declares a kind, or re-declares it without losing data. Extra columns are given as defaults, e.g. `hp=3.0` or `vel=(0.0, 0, 0)`. Every kind already has `pos[n,3]`, `vel[n,3]`, `rot[n,4]` (an xyzw quaternion), `scale[n,3]` and `tint[n,3]`.
  - `tint` is a per-instance sRGB colour multiplier on albedo and emissive (values above 1 glow brighter), so colour variety does not need more kinds.
  - `material` may be a pygfx material or a shading name: `"basic"` (unlit), `"phong"`, `"toon"`, `"standard"` or `"physical"`.
- `w.<name>` is the kind's table. Columns are **live NumPy views**: `k.pos += k.vel * w.dt`.
  - `k.spawn(count, **values)` returns the new indices.
  - `k.kill(mask_or_indices)` removes rows.
  - `k.clear()` removes every row. `len(k)` and `k.n` give the row count.
- **Time:** `w.t` (seconds) and `w.dt` (frame delta).
- **Input:**
  - `w.keys` (held) and `w.pressed` (this frame) hold lowercase names: `"w"`, `"space"`, `"shift"`, `"up"`, `"escape"`, …
  - `w.buttons` and `w.clicked` hold mouse buttons: 1 = left, 2 = right.
  - `w.mouse` is the position, `w.mouse_delta` the movement this frame.
  - `w.mouse_ray()` and `w.mouse_ground(y)` give where the cursor points in the world.
- **Scene:**
  - `w.camera` is a pygfx `PerspectiveCamera` (`.local.position`, `.local.rotation`, `.look_at(...)`).
  - `w.sun`, `w.ambient` and `w.background` are the default lights and backdrop.
  - `w.scene` is the raw pygfx scene, and `w.renderer` the pygfx renderer.
- **HUD:** `w.hud = "markdown text"`. Set `w.text.font_size` to change its size.
- **Game state:** any attribute you set on `w` (e.g. `w.score = 0`) is snapshotted on Play and restored on Stop.
- **Snapshots:** `w.snapshot()` / `w.restore(s)` copy everything. `w.save(path)` / `w.load(path)` write and read the entity tables as `.npz`.

## Assets, materials, audio
- `w.material(type="standard", filter="linear", **props)`: any `map`/`*_map` may be a file path (relative to the game) or an `(h, w, 4)` uint8 array. `filter="nearest"` gives crisp texels.
- `w.texture(path_or_array, filter, srgb=True)` returns a `gfx.TextureMap`.
- `w.model("thing.glb")` returns `dict(geometry=..., material=...)` for the first mesh, so use `w.kind("thing", **w.model("thing.glb"))`. Scale is the model's own; glTF files are often in centimetres.
- `w.play(sound, volume=1, loop=False, key=None)` plays a file path (wav/ogg/flac/mp3) or a NumPy array (mono or stereo, 44.1 kHz), and returns a voice with `.volume` and `.stop()`. With a `key`, later calls return the already-playing voice, so `w.play(HUM, loop=True, key="hum").volume = x` is safe every frame.
- Files loaded through these functions are cached until they change, and they are packed into playable screenshots.

## Look: `from lattice import fx`
- `fx.style(w, name)`, at the end of `scene()`, applies a whole art direction: `realistic`, `source`, `cel`, `ps2`, `ps1`, `n64`, `vhs`, `noir`, `gameboy` or `synthwave`. Fog passes you added yourself are kept.
- Presets for `w.renderer.effect_passes`:
  - `fx.bloom(strength)`, `fx.fog(hex, power)`, `fx.noise(amount)`
  - `fx.posterize(levels, dither)`, `fx.vignette(strength)`, `fx.chromatic(px)`, `fx.scanlines(strength, size)`
  - `fx.grade(saturation, contrast, tint)`, `fx.outline(color, threshold)`, `fx.palette(c0, c1, c2, c3)`
- `fx.retro(w.renderer, pixel=0.3)`: low internal resolution with nearest-neighbour upscaling.
- `fx.shader(wgsl, **uniforms)` builds a custom pass from a snippet that edits `color` (vec4f, linear).
  - In scope: `uv`, `px` (vec2i pixel), `res`, `t`, `depth` (0 near .. 1 far, only if you use it), `colorTex`, `depthTex`, `texSampler`, `random(x)` and `u_effect.<uniform>`.
  - Uniforms are floats, vectors (2-4) or hex colours, which become vec4. Set one live with `pass.name = value`.

## Worlds: `from lattice import gen`
- `gen.noise(x, z, seed, octaves, scale)` is vectorised fractal noise in [-1, 1].
- `gen.terrain(size, res, height, seed, scale, ramp=(hex...), flat=fn)` returns a `Terrain`.
  - Use `.geometry` with `material=w.material("standard", color_mode="vertex")`.
  - `.height(x, z)` samples the surface; `.place(points, offset)` stands points on it.
  - `flat(x, z)` returns 0..1 to flatten areas such as plazas, ponds or a city footprint.
- `gen.scatter(n, size, seed, where=fn, min_dist)` returns random xz points, masked and thinned.
- `gen.palette(n, *hexes, seed, jitter)` returns `(n, 3)` tints. `gen.rgb(*hexes)` turns hex colours into arrays.
- `gen.lots(blocks, block, street, per_side, seed)` returns `(centres, footprints)` for a city street grid.
- The fastest way to a beautiful world: **terrain, then scatter, then `land.place`, then `spawn(..., tint=gen.palette(...))`, then `fx.style`.** See `games/meadow.py` and `games/megacity.py`.

## Rules that keep games fast
1. **Never loop over entities in Python.** Use whole-column NumPy: masks, broadcasting, `np.where`.
2. **One kind per mesh/material combination.** Each kind is one draw call. Use the `tint` column for colour variety, not more kinds.
3. **Interactions between kinds** use broadcasting, e.g. `d = np.linalg.norm(a.pos[:, None] - b.pos[None], axis=-1)`.
4. **Mutate columns in place, or assign the whole column.** `k.rot = new_array` writes into the table.

## Verifying your work without a window
```sh
uv run lattice play game.py --snapshot out.png --frames 120   # simulate 120 fixed 1/60 s frames, save the last one (a playable shot)
uv run lattice edit game.py --snapshot editor.png             # screenshot the editor with the game loaded
```
Errors from `scene`, `setup`, `update` and custom shaders are printed with a traceback and drawn on screen. Look at the PNG to check visuals. `lattice play shot.png` resumes any playable screenshot.

## Look and style recipes
- **PS2/retro:** `fx.retro(w.renderer, 0.3)` and `w.renderer.effect_passes = [fx.fog("#1e1b0f", 28), fx.noise(0.04)]`.
- **Glow:** `material="basic"` with a `tint` above 1, plus `fx.bloom()`. Or use an emissive map (lit windows) on `"standard"`.
- **Night cities:** emissive window textures generated as NumPy arrays, neon signs as `basic` kinds with palette tints, then `fx.bloom` and fog. See `games/megacity.py`.
- **Geometry orientation:** pygfx builds cylinders along z. Stand one upright with `rot=(0.7071, 0, 0, 0.7071)`.

Complete examples:
- `games/liminal.py`: first-person horror with procedural audio.
- `games/galaxy.py`: 30k entities.
- `games/meadow.py`: a terrain, scatter and style showcase.
- `games/megacity.py`: a neon city with traffic, rain and bloom.
