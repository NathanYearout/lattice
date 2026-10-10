"""``lattice build game.py``: a standalone executable (PyInstaller), so players need no Python. The game runs headlessly
once first, so every asset it loads (textures, models, sounds, its scene file) is found and bundled with it."""

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LAUNCHER = """import sys
from pathlib import Path

from lattice.app import play

args = sys.argv[1:]  # `game.exe --snapshot out.png` renders headlessly, for checking a build
snapshot = args[args.index("--snapshot") + 1] if "--snapshot" in args else None
play(str(Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "game" / {game!r}), snapshot, title={title!r})
"""
COLLECT = ("pygfx", "wgpu", "rendercanvas", "glfw", "_sounddevice_data", "_soundfile_data")  # shaders, fonts, native libraries


def imports(path):
    """Top-level modules a game imports. Games are loaded from a file at runtime, so PyInstaller can't see them itself."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and not n.level}
    names |= {f"{n.module}.{a.name}" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "lattice" for a in n.names}
    return sorted(names)


def assets(game):
    """Every file the game touches while loading, found by running scene() and setup() offscreen."""
    import pygfx as gfx
    from rendercanvas.offscreen import RenderCanvas

    from . import audio
    from .runtime import Runtime

    audio.mixer.enabled = False
    rt = Runtime(game, gfx.WgpuRenderer(RenderCanvas(size=(64, 64))), log=lambda level, msg: print(msg) if level == "error" else None)
    rt.step(1 / 60)
    files = {f for f in rt.world.files if f.is_file() and f.is_relative_to(game.parent)}
    rt.close()
    return files


def build(game, out="dist", console=False):
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        sys.exit("lattice build needs PyInstaller: uv add --dev pyinstaller")
    game = Path(game).resolve()
    work = Path(tempfile.mkdtemp(prefix="lattice-build-"))
    launcher = work / f"{game.stem}.py"
    launcher.write_text(LAUNCHER.format(game=game.name, title=game.stem.replace("_", " ").title()), encoding="utf-8")
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--name", game.stem, "--distpath", str(Path(out).resolve()),
            "--workpath", str(work / "build"), "--specpath", str(work), "--exclude-module", "imgui_bundle", "--log-level", "WARN"]
    args += [] if console else ["--windowed"]
    for f in {game, *assets(game)}:
        args += ["--add-data", f"{f}{os.pathsep}{Path('game', f.relative_to(game.parent).parent).as_posix()}"]
    for module in imports(game):
        args += ["--hidden-import", module]
    for package in COLLECT:
        args += ["--collect-all", package]
    print(f"[lattice] building {game.name} (this takes a minute)...")
    subprocess.run([*args, str(launcher)], check=True)
    exe = Path(out).resolve() / (game.stem + (".exe" if os.name == "nt" else ""))
    print(f"[lattice] built {exe} ({exe.stat().st_size / 1e6:.0f} MB): send it to anyone, no Python required")
    return exe
