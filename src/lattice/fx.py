"""Post effects. ``shader()`` turns a few lines of WGSL into a hot-reloadable pass; the rest are stylised presets built on it
(or on pygfx's own passes). Stack them: ``w.renderer.effect_passes = [fx.outline(), fx.bloom(), fx.posterize()]``."""

import time

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu import EffectPass, FogPass, NoisePass, PhysicalBasedBloomPass, PPAAPass

TEMPLATE = """{$ include 'pygfx.noise.wgsl' $}
@fragment
fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
    let uv = varyings.texCoord;
    let px = vec2i(varyings.position.xy);
    let res = vec2f(textureDimensions(colorTex));
    let t = u_effect.t;
    var color = textureLoad(colorTex, px, 0);
    DEPTH
    BODY
    return color;
}"""


class Shader(EffectPass):
    """A custom pass; its uniforms read and write as attributes, e.g. ``crt.strength = 2``."""

    def __setattr__(self, name, value):
        if name in type(self).uniform_type:
            self._uniform_data[name] = linear(value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in type(self).uniform_type:
            return self._uniform_data[name]
        raise AttributeError(name)

    def render(self, *args):
        self._uniform_data["t"] = time.perf_counter() % 3600
        super().render(*args)


def linear(value):
    """Hex colours become linear RGBA (what effect passes compute in); numbers pass through."""
    return (*gfx.Color(value).to_physical(), 1.0) if isinstance(value, str) else value


def shader(wgsl, name="Shader", **uniforms):
    """A pass from a WGSL snippet that edits ``color`` (vec4f, linear). In scope: ``uv``, ``px`` (pixel), ``res``, ``t`` (s),
    ``depth`` (0 near .. 1 far, if you use it), ``colorTex``/``depthTex``/``texSampler``, ``random(x)``, ``u_effect.<uniform>``.
    Uniforms are floats, vectors (2-4) or hex colours (become vec4)."""
    types = {k: "f4" if np.size(linear(v)) == 1 else f"{np.size(linear(v))}xf4" for k, v in uniforms.items()}
    code = TEMPLATE.replace("BODY", wgsl).replace("DEPTH", "let depth = textureLoad(depthTex, px, 0);" if "depth" in wgsl else "")
    cls = type(name, (Shader,), dict(uniform_type=dict(EffectPass.uniform_type, t="f4", **types), USES_DEPTH="depth" in wgsl, wgsl=code))
    fx = cls()
    for k, v in uniforms.items():
        setattr(fx, k, v)
    return fx


def retro(renderer, pixel=0.3):
    """Low internal resolution, blown up with crisp nearest-neighbour pixels and no anti-aliasing: the PS1/PS2 base look."""
    renderer.pixel_ratio, renderer.pixel_filter, renderer.ppaa = pixel, "nearest", "none"


def fog(color="#808080", power=28.0):
    return FogPass(gfx.Color(*gfx.Color(color).to_physical()), power=power)


def noise(amount=0.04):
    return NoisePass(amount)


def bloom(strength=0.06, radius=0.005):
    return PhysicalBasedBloomPass(bloom_strength=strength, filter_radius=radius)


def posterize(levels=24.0, dither=1.0):
    """PS1-style colour banding with a 4x4 ordered dither."""
    return shader(name="Posterize", wgsl="""
    var bayer = array<f32, 16>(0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);
    let d = (bayer[(px.x % 4) + (px.y % 4) * 4] / 16.0 - 0.5) * u_effect.dither / u_effect.levels;
    let c = pow(max(color.rgb, vec3f(0.0)), vec3f(1.0 / 2.2)) + d;
    color = vec4f(pow(floor(c * u_effect.levels + 0.5) / u_effect.levels, vec3f(2.2)), color.a);""", levels=levels, dither=dither)


def vignette(strength=0.6):
    return shader(name="Vignette", wgsl="let e = uv - 0.5; color = vec4f(color.rgb * (1.0 - u_effect.strength * dot(e, e) * 2.0), color.a);", strength=strength)


def chromatic(amount=2.0):
    """Lens fringing: red and blue split apart towards the screen edges (``amount`` in pixels at the corners)."""
    return shader(name="Chromatic", wgsl="""
    let o = (uv - 0.5) * 2.0 * u_effect.amount / res;
    color = vec4f(textureSample(colorTex, texSampler, uv + o).r, color.g, textureSample(colorTex, texSampler, uv - o).b, color.a);""",
                  amount=amount)


def scanlines(strength=0.25, size=2.0):
    return shader(name="Scanlines", wgsl="color = vec4f(color.rgb * (1.0 - u_effect.strength * step(0.5, fract(varyings.position.y / u_effect.size / 2.0))), "
                  "color.a);", strength=strength, size=size)


def grade(saturation=1.15, contrast=1.1, tint="#ffffff"):
    """Colour grading: saturation, contrast around mid grey and a multiplied tint."""
    return shader(name="Grade", wgsl="""
    let l = dot(color.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let s = mix(vec3f(l), color.rgb, u_effect.saturation);
    color = vec4f(max((s - 0.18) * u_effect.contrast + 0.18, vec3f(0.0)) * u_effect.tint.rgb, color.a);""",
                  saturation=saturation, contrast=contrast, tint=tint)


def outline(color="#000000", threshold=0.04, strength=1.0):
    """Ink lines wherever depth jumps: an anime/cel look when paired with ``w.material("toon")``."""
    return shader(name="Outline", wgsl="""
    let lin = 1.0 / (1.0001 - depth);
    let n = 1.0 / (1.0001 - textureLoad(depthTex, px + vec2i(1, 0), 0)) + 1.0 / (1.0001 - textureLoad(depthTex, px - vec2i(1, 0), 0))
          + 1.0 / (1.0001 - textureLoad(depthTex, px + vec2i(0, 1), 0)) + 1.0 / (1.0001 - textureLoad(depthTex, px - vec2i(0, 1), 0));
    let edge = clamp((abs(n - 4.0 * lin) / lin - u_effect.threshold) * 8.0, 0.0, 1.0) * u_effect.strength;
    color = vec4f(mix(color.rgb, u_effect.color.rgb, edge), color.a);""", color=color, threshold=threshold, strength=strength)


def palette(dark="#0f380f", mid_dark="#306230", mid_light="#8bac0f", light="#9bbc0f", dither=0.5):
    """Map brightness onto exactly four colours, Game Boy style (any four you like), with an ordered dither."""
    return shader(name="Palette", wgsl="""
    var bayer = array<f32, 16>(0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);
    let l = pow(dot(max(color.rgb, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722)), 1.0 / 2.2) * 1.15
          + (bayer[(px.x % 4) + (px.y % 4) * 4] / 16.0 - 0.5) * u_effect.dither / 4.0;
    let i = clamp(i32(l * 4.0), 0, 3);
    color = select(select(u_effect.mid_light, u_effect.light, i == 3), select(u_effect.dark, u_effect.mid_dark, i == 1), i < 2);""",
                  dark=dark, mid_dark=mid_dark, mid_light=mid_light, light=light, dither=dither)


PRESETS = dict(palette=palette, bloom=bloom, fog=fog, noise=noise, posterize=posterize, vignette=vignette, chromatic=chromatic, scanlines=scanlines,
               grade=grade, outline=outline)


STYLES = {  # name: (pixel ratio, filter, anti-aliasing, effects, material shading or None to keep)
    "realistic": (1.0, "linear", "ddaa", lambda: [bloom(0.04), grade(1.05, 1.05), vignette(0.25)], "physical"),
    "source": (1.0, "linear", "fxaa", lambda: [bloom(0.02), grade(0.85, 1.08, "#fff1dc"), vignette(0.35)], "phong"),
    "cel": (1.0, "linear", "fxaa", lambda: [outline(), grade(1.35, 1.05), bloom(0.03)], "toon"),
    "ps2": (0.33, "nearest", "none", lambda: [posterize(48, 0.6), vignette(0.4), noise(0.035)], None),
    "ps1": (0.25, "nearest", "none", lambda: [posterize(20, 1.0), noise(0.05)], "phong"),
    "n64": (0.4, "linear", "fxaa", lambda: [grade(1.2, 0.95), posterize(32, 0.3)], "phong"),
    "vhs": (0.5, "linear", "none", lambda: [chromatic(2.5), scanlines(0.2), grade(0.8, 1.1, "#e6efff"), vignette(0.6), noise(0.08)], None),
    "noir": (1.0, "linear", "fxaa", lambda: [grade(0.0, 1.35), bloom(0.03), vignette(0.8), noise(0.06)], None),
    "gameboy": (0.2, "nearest", "none", lambda: [palette()], None),
    "synthwave": (0.6, "nearest", "none", lambda: [bloom(0.12), chromatic(1.0), grade(1.4, 1.1, "#ffd0ff"), scanlines(0.12)], None),
}


def style(w, name):
    """One line, a whole art direction: resolution, filtering, post stack and (for some) shading model. Fog passes you set
    up yourself are kept. Call it at the end of ``scene()``, e.g. ``fx.style(w, "cel")``."""
    from .world import MATERIALS, convert

    pixel, filt, aa, effects, shading = STYLES[name]
    r = w.renderer
    r.effect_passes = [p for p in r.effect_passes if isinstance(p, FogPass)] + effects()
    r.pixel_ratio, r.pixel_filter, r.ppaa = pixel, filt, aa  # ppaa appends its own pass, so it goes last
    for k in w.kinds.values() if shading else ():
        if type(k.material) is not MATERIALS[shading] and isinstance(k.material, (gfx.MeshStandardMaterial, gfx.MeshPhongMaterial)):
            k.material = k.obj.material = convert(k.material, MATERIALS[shading])
    w.style = name
