# Building games with Lattice: a guide for agents

A game is **one Python file** with up to three hooks. The engine hot-reloads it on save.

```python
def scene(w): ...   # declare kinds (mesh + material + columns) and renderer look. Re-run on EVERY save.
def setup(w): ...   # spawn the starting entities, place the camera, set game state. Run once (and on Restart).
def update(w): ...  # per-frame logic. Only runs while playing.
```

## The world `w`
- `w.kind(name, geometry, color="#hex", material=None, **columns)` declares a kind, or re-declares it without losing data. Extra columns are given as defaults, e.g. `hp=3.0` or `vel=(0.0, 0, 0)`. Every kind already has `pos[n,3]`, `vel[n,3]`, `rot[n,4]` (an xyzw quaternion) and `scale[n,3]`.
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

## Rules that keep games fast
1. **Never loop over entities in Python.** Use whole-column NumPy: masks, broadcasting, `np.where`.
2. **One kind per mesh/material combination.** Each kind is one draw call, so colour variety means more kinds.
3. **Interactions between kinds** use broadcasting, e.g. `d = np.linalg.norm(a.pos[:, None] - b.pos[None], axis=-1)`.
4. **Mutate columns in place, or assign the whole column.** `k.rot = new_array` writes into the table.

## Verifying your work without a window
```sh
uv run lattice play game.py --snapshot out.png --frames 120   # simulate 120 fixed 1/60 s frames, save the last one
uv run lattice edit game.py --snapshot editor.png             # screenshot the editor with the game loaded
```
Errors from `scene`, `setup` or `update` are printed with a traceback and drawn on screen. Look at the PNG to check visuals.

## Look and style recipes
- **PS2/retro:**
  - `w.renderer.pixel_ratio, w.renderer.pixel_filter, w.renderer.ppaa = 0.3, "nearest", "none"`
  - Fog and grain: `w.renderer.effect_passes = [gfx.renderers.wgpu.FogPass(color, power=28), gfx.renderers.wgpu.NoisePass(0.04)]`. Effect-pass colours are linear, so pass `gfx.Color(*gfx.Color(hex).to_physical())`.
- **Glow:** `gfx.MeshStandardMaterial(color=c, emissive=c, emissive_intensity=1)`, or `gfx.MeshBasicMaterial` for unlit.
- **Geometry orientation:** pygfx builds cylinders along z. Stand one upright with `rot=(0.7071, 0, 0, 0.7071)`.

See `games/liminal.py` (a first-person horror game) and `games/galaxy.py` (30k entities) for complete examples.
