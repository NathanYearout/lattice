@AGENTS.md

## Working on the engine itself
- Run commands through uv: `uv run lattice …`. Dependencies live in `pyproject.toml` and `uv.lock`.
- Keep code minimal. Reach for an existing library (pygfx, imgui-bundle, watchfiles, pylinalg) before writing anything.
- Check changes visually with the `--snapshot` flags. The editor can also be driven from a script: build `Editor(path, snapshot=...)`, call `canvas.request_draw(e.gui.render)`, then call `canvas.draw()`.
- Write files as UTF-8 (`encoding="utf-8"`): on Windows Python defaults to cp1252.
- imgui-bundle is pinned below 1.92.900 because the wgpu-py 0.32 ImGui backend needs `ImDrawData.cmd_lists_count`.
