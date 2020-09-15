from bokeh.palettes import viridis
from matplotlib.colors import rgb2hex

from shap.plots.colors import red_blue


def _shap_palette(new_n):
    new_palette = red_blue._resample(new_n)
    hex_palette = []

    for i in range(new_palette.N):
        rgb = new_palette(i)[:3]
        hex_palette.append(rgb2hex(rgb))

    return hex_palette


_PALETTE_FUNCS = {"Viridis": viridis, "Shap": _shap_palette}

_PALETTE = "Shap"

PALETTE_FUNC = _PALETTE_FUNCS.get(_PALETTE)
PALETTE_LARGE = PALETTE_FUNC(256)
TOP_BOX_COLOR = PALETTE_FUNC(8)[-1]
BOTTOM_BOX_COLOR = PALETTE_FUNC(8)[1]
LINE_COLOR = "#33FF33"
