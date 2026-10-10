<div align="center">

# LATTICE

**A 3D game engine where games are Python files, worlds are arrays, and everything is live.**

`Python 3.12` · `WebGPU` · `NumPy` · `Dear ImGui` · `uv`

![The Lattice editor editing Fluorescent](docs/editor.png)

</div>

---

Most Python engines give every entity its own Python object, so the interpreter runs once per enemy, bullet and star, every frame. Lattice doesn't. **Each kind of thing is a table of NumPy columns, and each table is drawn as one GPU instanced mesh.** Game logic is array maths over whole tables at once, so Python's cost scales with the number of *kinds*, not the number of *things*.

```python
k.pos += k.vel * w.dt          # moves every star at once
w.enemy.kill(w.enemy.hp <= 0)  # removes every dead enemy at once
```

The demo galaxy simulates **30,000 tumbling, orbiting entities in 3 draw calls**, and stays smooth.

## Highlights

| | |
|---|---|
| **Games are one `.py` file** | Three optional hooks: `scene(w)` for looks, `setup(w)` for the starting state, `update(w)` for every frame. |
| **Hot reload that keeps state** | Save the file and the code is swapped while the world keeps running. Errors show on screen, and the game resumes as soon as you fix them. |
| **Data-oriented by default** | Entities are rows. Components are columns. Systems are NumPy expressions. |
| **A real editor** | Docked panels, transform gizmos, click-to-select, Play/Pause/Step, undo/redo, a built-in code editor, console and project browser. |
| **Play mode that reverts** | Press Play, wreck everything, press Stop, and you're back where you were. `World.snapshot()` makes it free. |
| **Stylised rendering built in** | PBR, shadows, fog, film grain, bloom and low-resolution pixel upscaling come from [pygfx](https://github.com/pygfx/pygfx) on WebGPU (Vulkan, Metal or DX12). |

<table>
<tr>
<td><img src="docs/liminal.png" alt="Fluorescent"></td>
<td><img src="docs/galaxy.png" alt="Galaxy"></td>
</tr>
<tr>
<td align="center"><b>Fluorescent</b><br>PS2-era liminal horror. It only moves when you aren't looking.</td>
<td align="center"><b>Galaxy</b><br>30,000 entities, a few lines of NumPy.</td>
</tr>
</table>

## Quick start

You need [uv](https://docs.astral.sh/uv/) and a GPU with Vulkan, Metal or DX12 support.

```sh
git clone <this repo> lattice && cd lattice
uv sync

uv run lattice                          # open the editor
uv run lattice edit games/liminal.py    # open a game in the editor
uv run lattice play games/liminal.py    # just play it, in a plain window
```

## A whole game

```python
import numpy as np
import pygfx as gfx

def scene(w):                                   # looks: re-run on every save
    w.kind("ground", gfx.box_geometry(), "#2a2a32")
    w.kind("cube", gfx.box_geometry(), "#ffb347", spin=0.0)

def setup(w):                                   # starting state: run once
    w.ground.spawn(pos=(0, -0.5, 0), scale=(40, 1, 40))
    xz = np.mgrid[-4:5, -4:5].reshape(2, -1).T * 2.0
    w.cube.spawn(len(xz), pos=np.c_[xz[:, 0], np.full(len(xz), 0.5), xz[:, 1]],
                 spin=np.random.uniform(-2, 2, len(xz)))
    w.camera.local.position = (12, 10, 14)
    w.camera.look_at((0, 0, 0))

def update(w):                                  # every frame
    c = w.cube
    c.pos[:, 1] = 0.5 + np.abs(np.sin(w.t * 2 + c.pos[:, 0]))
```

**File → New Game** in the editor writes exactly this for you.

## How it works

```
 your_game.py ──save──▶ watchfiles ──▶ Runtime.reload()   (module swapped; entity tables untouched)
                                          │
   w.kind("enemy", mesh, material)        ▼
   ┌───────────── Kind table ─────────────┐   Kind.sync(): one vectorized pass composes all
   │ pos[n,3]  rot[n,4]  scale[n,3]  hp[n]│ ─────▶ transforms into an InstancedMesh buffer
   └──────────────────────────────────────┘            │
                                                       ▼
                                pygfx ▶ wgpu ▶ Vulkan / Metal / DX12   (one draw call per kind)
```

| Module | Lines | Role |
|---|---|---|
| `world.py` | ~170 | `Kind` tables (spawn, kill, live column views), the `World` handle, snapshots |
| `runtime.py` | ~110 | Loads a game, hot reload, error isolation, input, default lighting |
| `app.py` | ~60 | The `lattice` command line |
| `editor/` | ~930 | The editor and its design tokens |

Everything else comes from libraries that already do it well: [pygfx](https://github.com/pygfx/pygfx) and [wgpu-py](https://github.com/pygfx/wgpu-py) for rendering, [imgui-bundle](https://github.com/pthom/imgui_bundle) for the UI ([Dear ImGui](https://github.com/ocornut/imgui), [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo), [ImGuiColorTextEdit](https://github.com/goossens/ImGuiColorTextEdit)), [watchfiles](https://github.com/samuelcolvin/watchfiles), [NumPy](https://numpy.org) and [pylinalg](https://github.com/pygfx/pylinalg).

## Editor

| Action | Input |
|---|---|
| Fly the scene camera | Hold **RMB**, then **W A S D** / **Q E** (**Shift** goes faster, the wheel sets speed) |
| Pan · zoom | **MMB** drag · mouse wheel |
| Select | **LMB** in the viewport, or a row in the Hierarchy |
| Move · rotate · scale | **W** · **E** · **R** |
| Frame the selection | **F** |
| Play / stop | **Ctrl+P** |
| Undo · redo | **Ctrl+Z** · **Ctrl+Y** |
| Duplicate · delete | **Ctrl+D** · **Del** |
| Save the script, or the scene (`.scene.npz`) | **Ctrl+S** |

The **Scene** view is a clean, lit editing view. The sparkle and sun toggles preview the game's post effects and lighting. The **Game** view shows exactly what players see.

<div align="center"><img src="docs/editor_galaxy.png" alt="Editing the galaxy" width="85%"></div>

### Design

The UI follows Material's dark-theme guidance:
- near-black surfaces, where a lighter grey means higher elevation;
- one desaturated accent, a CRT-phosphor amber;
- body text at 15.9:1 contrast and secondary text at 6.2:1 (WCAG AA needs 4.5:1);
- spacing on a 4/8 px grid.

The type is [IBM Plex](https://github.com/IBM/plex) and the icons are [Lucide](https://lucide.dev). All of the tokens live in [`editor/theme.py`](src/lattice/editor/theme.py).

## Command line

```
lattice [edit] [game.py] [--snapshot out.png]       open the editor (or screenshot it headlessly)
lattice play game.py [--snapshot out.png --frames N] play in a window (or render frame N offscreen)
```

Headless snapshots mean you, CI or an AI agent can check a game visually without a window. See [AGENTS.md](AGENTS.md).

## Roadmap

These came out of a research pass on what no other engine does yet:

- [ ] **Fluorescent: Infinite.** Seeded, seamless chunk streaming, where **rooms regenerate behind you when you aren't looking**. It's the stalker's rule applied to the building itself.
- [ ] **GPU systems.** Hot-reloaded WGSL compute kernels write straight into a kind's instance buffer, so its transforms never touch the CPU.
- [ ] **An MCP server and an in-editor agent.** A world that is deterministic, tabular and snapshot-able is the easiest thing in the world for an agent to query, test and branch.
- [ ] **`lattice test`.** A fixed timestep, an engine-owned RNG, recorded input tapes and state hashes.
- [ ] Audio (a procedural fluorescent hum), glTF import, and standalone builds via PyInstaller.

## Third-party licenses

pygfx, wgpu-py and pylinalg are BSD-2-Clause. imgui-bundle, Dear ImGui, ImGuizmo, ImGuiColorTextEdit and watchfiles are MIT. NumPy is BSD-3-Clause. Lucide icons are ISC ([license](src/lattice/assets/LUCIDE_LICENSE.txt)). IBM Plex is SIL OFL 1.1 ([license](src/lattice/assets/OFL.txt)).
