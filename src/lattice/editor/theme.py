"""Design tokens and the Dear ImGui style built from them.

Follows Material's dark-theme guidance: near-black (not pure black) surfaces where elevation is a lighter grey,
one desaturated accent, body text >= 4.5:1 contrast (WCAG AA), and spacing on a 4/8 px grid.
The accent is a warm CRT-phosphor amber: modern layout, retro soul.
"""

from pathlib import Path

from imgui_bundle import imgui

from .. import icons

ASSETS = Path(__file__).parent.parent / "assets"


def hex4(h, a=1.0):
    h = h.lstrip("#")
    return imgui.ImVec4(*(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)), a)


# Surfaces by elevation, text, accent, semantic states
BG, PANEL, RAISED, HOVER, ACTIVE, BORDER = "#0b0b0d", "#121215", "#1a1a1f", "#24242b", "#2e2e37", "#26262e"
TEXT, DIM, FAINT = "#ebe9e4", "#9a979f", "#5d5b63"  # 15.9:1, 6.2:1 and 2.9:1 (decorative only) on PANEL
ACCENT, ACCENT_HOVER, ACCENT_DIM = "#ffb347", "#ffc670", "#7a5420"
OK, WARN, ERROR, INFO = "#7ee0a1", "#ffd36b", "#ff6e6e", "#7cc4ff"


class Fonts:
    ui = bold = mono = title = None


def apply():
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.docking_enable | imgui.ConfigFlags_.nav_enable_keyboard
    io.config_windows_move_from_title_bar_only = True

    def font(name, size, with_icons=True):
        f = io.fonts.add_font_from_file_ttf(str(ASSETS / name), size)
        if with_icons:
            cfg = imgui.ImFontConfig()
            cfg.merge_mode, cfg.glyph_offset, cfg.glyph_min_advance_x = True, imgui.ImVec2(0, 2), size
            io.fonts.add_font_from_file_ttf(str(ASSETS / "lucide.ttf"), size, cfg)
        return f

    Fonts.ui = font("IBMPlexSans-Medium.ttf", 16)
    Fonts.bold = font("IBMPlexSans-SemiBold.ttf", 16)
    Fonts.mono = font("IBMPlexMono-Regular.ttf", 14)
    Fonts.title = font("IBMPlexSans-SemiBold.ttf", 44, with_icons=False)
    io.font_default = Fonts.ui

    s = imgui.get_style()
    s.window_padding, s.frame_padding, s.cell_padding = imgui.ImVec2(8, 8), imgui.ImVec2(8, 5), imgui.ImVec2(8, 4)
    s.item_spacing, s.item_inner_spacing = imgui.ImVec2(8, 6), imgui.ImVec2(6, 4)
    s.indent_spacing, s.scrollbar_size, s.grab_min_size = 16, 10, 8
    s.window_rounding = s.child_rounding = s.popup_rounding = 6
    s.frame_rounding = s.grab_rounding = s.tab_rounding = s.scrollbar_rounding = 4
    s.window_border_size, s.frame_border_size, s.popup_border_size, s.tab_border_size = 1, 0, 1, 0
    s.window_title_align, s.window_menu_button_position = imgui.ImVec2(0, 0.5), imgui.Dir.none
    s.separator_text_border_size, s.docking_separator_size = 1, 2

    C = imgui.Col_
    for col, value in {
        C.text: TEXT, C.text_disabled: FAINT, C.window_bg: PANEL, C.child_bg: PANEL, C.popup_bg: RAISED,
        C.border: BORDER, C.frame_bg: RAISED, C.frame_bg_hovered: HOVER, C.frame_bg_active: ACTIVE,
        C.title_bg: BG, C.title_bg_active: BG, C.title_bg_collapsed: BG, C.menu_bar_bg: BG,
        C.scrollbar_bg: PANEL, C.scrollbar_grab: HOVER, C.scrollbar_grab_hovered: ACTIVE, C.scrollbar_grab_active: ACCENT_DIM,
        C.check_mark: ACCENT, C.slider_grab: ACCENT, C.slider_grab_active: ACCENT_HOVER,
        C.button: RAISED, C.button_hovered: HOVER, C.button_active: ACTIVE,
        C.header: HOVER, C.header_hovered: HOVER, C.header_active: ACTIVE,
        C.separator: BORDER, C.separator_hovered: ACCENT_DIM, C.separator_active: ACCENT,
        C.resize_grip: BORDER, C.resize_grip_hovered: ACCENT_DIM, C.resize_grip_active: ACCENT,
        C.tab: BG, C.tab_hovered: HOVER, C.tab_selected: PANEL, C.tab_selected_overline: ACCENT,
        C.tab_dimmed: BG, C.tab_dimmed_selected: PANEL, C.tab_dimmed_selected_overline: BORDER,
        C.docking_preview: ACCENT, C.docking_empty_bg: BG, C.plot_lines: ACCENT, C.plot_histogram: ACCENT,
        C.table_header_bg: RAISED, C.table_border_strong: BORDER, C.table_border_light: BORDER,
        C.text_selected_bg: ACCENT_DIM, C.drag_drop_target: ACCENT, C.nav_cursor: ACCENT,
        C.modal_window_dim_bg: BG,
    }.items():
        s.set_color_(col, hex4(value, 0.85 if col == C.modal_window_dim_bg else 0.35 if col == C.docking_preview else 1))
    return icons
