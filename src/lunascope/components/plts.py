#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  Luna is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with Luna. If not, see <http:#www.gnu.org/licenses/>.
# 
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

import lunapi as lp
import numpy as np
from lunascope.helpers import winsorize_array

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import pyplot as plt


def _reset_figure_axes(ax):
    """Keep *ax* as the only plotting axes before redrawing."""
    fig = ax.figure
    for extra_ax in list(fig.axes):
        if extra_ax is not ax:
            extra_ax.remove()
    return fig

def _plot_background(show_legend=False):
    return "white" if show_legend else "black"

def _set_plot_background(ax, show_legend=False):
    bg = _plot_background(show_legend)
    ax.figure.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return bg

def _cmap_with_bad(cmap, bad_color):
    cm = colormaps[cmap] if isinstance(cmap, str) else cmap
    try:
        cm = cm.copy()
    except AttributeError:
        pass
    try:
        cm.set_bad(bad_color)
    except AttributeError:
        pass
    return cm

@staticmethod
def hypno(ss, e=None, ax=None, *, title=None, xsize=20, ysize=2, clear=True):
    """Plot a hypnogram into an existing Axes if provided."""
    from matplotlib import colors as mcolors
    ssn = np.array(lp.stgn(ss), dtype=float)
    if e is None:
        e = np.arange(len(ssn), dtype=float)
    e = e / 120.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(xsize, ysize))
    elif clear:
        ax.clear()

    # Detect background brightness so guide colours adapt to dark/light themes
    try:
        bg_rgb = mcolors.to_rgb(ax.get_facecolor())
        lum = 0.299*bg_rgb[0] + 0.587*bg_rgb[1] + 0.114*bg_rgb[2]
        dark_bg = lum < 0.3
    except Exception:
        dark_bg = False

    guide_col  = '#4a4a4a' if dark_bg else '#d0d0d0'
    back_col   = '#888888' if dark_bg else '#b8b8b8'
    vtrans_col = '#5c5c5c' if dark_bg else '#aaaaaa'

    # Five guide lines — one per stage, aligned with the actual hypnogram track
    for y in [-3, -2, -1, 0, 1]:
        ax.axhline(y, color=guide_col, linewidth=0.5, zorder=1)

    n = len(e)
    ep_dur = float(e[1] - e[0]) if n > 1 else 1.0 / 120.0
    colors = lp.stgcol(ss)

    # Backline: wide neutral step so dark stages (e.g. N3) stay visible
    for i in range(n):
        y = ssn[i]
        if np.isfinite(y):
            ax.plot([e[i], e[i] + ep_dur], [y, y],
                    color=back_col, linewidth=9.0, solid_capstyle='butt', zorder=2)

    # Coloured stage segments on top
    for i in range(n):
        y = ssn[i]
        if np.isfinite(y):
            ax.plot([e[i], e[i] + ep_dur], [y, y],
                    color=colors[i], linewidth=5.5, solid_capstyle='butt', zorder=3)

    # Vertical transitions between stages
    for i in range(n - 1):
        y0, y1 = ssn[i], ssn[i + 1]
        if np.isfinite(y0) and np.isfinite(y1) and y0 != y1:
            ax.plot([e[i] + ep_dur, e[i] + ep_dur], [y0, y1],
                    color=vtrans_col, linewidth=1.0, zorder=2)

    ax.set_ylabel('Sleep stage')
    ax.set_xlabel('Time (hrs)')
    ax.set_ylim(-3.5, 2.5)
    ax.set_xlim(0, float(np.nanmax(e)) + ep_dur)
    ax.set_yticks([-3, -2, -1, 0, 1, 2], labels=['N3','N2','N1','R','W','?'])
    if title:
        ax.set_title(title)
    return ax

@staticmethod
def spec(ss, e=None, ax=None, *, title=None, xsize=20, ysize=2, clear=True):
    ssn = lp.stgn(ss)
    if e is None:
        e = np.arange(len(ssn), dtype=float)
    e = e / 120.0

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(xsize, ysize))
        created = True
    elif clear:
        ax.clear()

    ax.plot(e, ssn, color='gray', linewidth=0.5, zorder=2)
    ax.scatter(e, ssn, c=lp.stgcol(ss), s=10, zorder=3)
    ax.set_ylabel('Sleep stage')
    ax.set_xlabel('Time (hrs)')
    ax.set_ylim(-3.5, 2.5)
    ax.set_xlim(0, float(np.nanmax(e)))
    ax.set_yticks([-3, -2, -1, 0, 1, 2], labels=['N3','N2','N1','R','W','?'])
    if title:
        ax.set_title(title)
    return ax  # caller decides whether to draw


# --------------------------------------------------------------------------------
# plot a Hjorthgram

def derive_hjorth_data(ch, p, winsor=0.0, epoch_dur=30):
    res = p.silent_proc_lunascope(f'EPOCH dur={epoch_dur} verbose & SIGSTATS epoch sig={ch}')
    df = res.get('SIGSTATS: CH_E')
    dt = res.get('EPOCH: E')
    if df is None or dt is None or len(df) == 0 or len(dt) == 0:
        return None

    # Align Hjorth rows to epoch START using E, so gaps map consistently
    # with the spectrogram x-axis.
    if "E" in df.columns and "E" in dt.columns and "START" in dt.columns:
        dx = df[["E"]].merge(dt[["E", "START"]], on="E", how="left")
        if not dx["START"].notna().any():
            return None
        x = dx["START"].to_numpy(float)
    elif "START" in dt.columns:
        x = dt["START"].to_numpy(float)
        if len(x) != len(df):
            x = x[:len(df)]
    else:
        return None
    
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        r = mx - mn
        if not np.isfinite(r) or r <= 1e-8:
            r = 1.0
        y = (arr - mn) / r
        y[~np.isfinite(y)] = 0.0
        return y


    # standardize Hjorth values
    y1 = _norm(winsorize_array(df["H1"].to_numpy(float), winsor))
    y2 = _norm(winsorize_array(df["H2"].to_numpy(float), winsor))
    y3 = _norm(winsorize_array(df["H3"].to_numpy(float), winsor))

    return {
        "x": x,
        "y1": y1,
        "y2": y2,
        "y3": y3,
        "epoch_dur": float(epoch_dur),
    }


def draw_hjorth_data(data, ax, show_legend=False):
    fig = _reset_figure_axes(ax)
    ax.clear()

    if not data:
        return ax

    x = np.asarray(data["x"], dtype=float)
    y1 = np.asarray(data["y1"], dtype=float)
    y2 = np.asarray(data["y2"], dtype=float)
    y3 = np.asarray(data["y3"], dtype=float)
    elen = float(data.get("epoch_dur", 30.0))
    if x.size == 0:
        return ax

    # color axes
    idx2 = np.clip(np.rint(y2 * 99).astype(int), 0, 99)
    idx3 = np.clip(np.rint(y3 * 99).astype(int), 0, 99)
    colors2 = colormaps["turbo"](y2)  # y2 in [0,1]
    colors3 = colormaps["turbo"](y3)

    midy = 0

    rects_top = [Rectangle((xi, midy), elen, hi) for xi, hi in zip(x, y1)]
    rects_bot = [Rectangle((xi, midy - hi), elen, hi) for xi, hi in zip(x, y1)]
    pc_top = PatchCollection(rects_top, facecolors=colors2, edgecolor="none", linewidth=0)
    pc_bot = PatchCollection(rects_bot, facecolors=colors3, edgecolor="none", linewidth=0)
    ax.add_collection(pc_top)
    ax.add_collection(pc_bot)


    fig.set_constrained_layout(False)      # or: fig.set_layout_engine(None)
    ax.margins(x=0, y=0)

    ax.set_xlim(0, max(x) + elen)
    ax.set_ylim(-1, 1)
    ax.margins(0)
    _set_plot_background(ax, show_legend)
    ax.set_aspect("auto")

    if show_legend:
        fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.95, wspace=0, hspace=0)
        ax.set_position([0.08, 0.16, 0.74, 0.79])
        ax.set_axis_on()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hjorth activity")
        ax.set_yticks([-1, 0, 1])
        cbar = fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=colormaps["turbo"]),
            ax=ax,
            fraction=0.035,
            pad=0.025,
        )
        cbar.set_label("Normalized mobility / complexity")
    else:
        # no auto layout padding; make the axes fill the figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        ax.set_position([0, 0, 1, 1])
        # no axes decorations
        ax.set_axis_off()
        ax.axis("off")
    
    return ax


@staticmethod
def plot_hjorth( ch , ax , p , gui , epoch_dur=30, show_legend=False ):
    data = derive_hjorth_data(ch, p, winsor=gui.spin_win.value(), epoch_dur=epoch_dur)
    return draw_hjorth_data(data, ax, show_legend=show_legend)


# --------------------------------------------------------------------------------
# plot a spectrogram
        
@staticmethod
def plot_spec( xi,yi,zi, ch, minf, maxf, ax , gui, clear = True, show_legend=False):

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(xsize, ysize))
        created = True
    elif clear:
        _reset_figure_axes(ax)
        ax.clear()
        
    if len(xi) == 0: return ax

    fig = ax.figure
    fig.set_constrained_layout(False)
    bg = _set_plot_background(ax, show_legend)
    if show_legend:
        fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.95, wspace=0, hspace=0)
        ax.set_position([0.08, 0.16, 0.74, 0.79])
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        ax.set_position([0, 0, 1, 1])

    ax.set_xlabel('Time (s)' if show_legend else 'Epoch')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_axis_on()
    ax.set_ylim(float(yi[0]), float(yi[-1]))
    p1 = ax.pcolormesh(xi, yi, zi, cmap=_cmap_with_bad("turbo", bg))
    if len(xi) > 1:
        ax.set_xlim(0, float(np.nanmax(xi)))
    ax.set_aspect("auto")
    ax.margins(x=0, y=0)
    if show_legend:
        cbar = fig.colorbar(p1, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("PSD (dB)")
    return ax  


@staticmethod
def plot_tf_heatmap(xi, yi, zi, title, ax, *, y_label="Frequency (Hz)",
                    cbar_label="Value", y_ticklabels=None, show_legend=False,
                    clear=True, cmap="turbo", center_zero=False):
    if ax is None:
        _, ax = plt.subplots()
    elif clear:
        _reset_figure_axes(ax)
        ax.clear()

    fig = ax.figure
    fig.set_constrained_layout(False)
    bg = _set_plot_background(ax, show_legend)
    if show_legend:
        fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.95, wspace=0, hspace=0)
        ax.set_position([0.08, 0.16, 0.74, 0.79])
    else:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        ax.set_position([0, 0, 1, 1])

    if len(xi) == 0 or len(yi) == 0:
        if show_legend:
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(y_label)
        else:
            ax.set_axis_off()
        return ax

    norm = None
    if center_zero:
        data = np.ma.asarray(zi, dtype=float).compressed()
        data = data[np.isfinite(data)]
        vmax = float(np.nanmax(np.abs(data))) if data.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    p1 = ax.pcolormesh(xi, yi, zi, cmap=_cmap_with_bad(cmap, bg), norm=norm)
    ax.set_aspect("auto")
    ax.margins(x=0, y=0)
    if len(xi) > 1:
        ax.set_xlim(float(np.nanmin(xi)), float(np.nanmax(xi)))
    if len(yi) > 1:
        ax.set_ylim(float(np.nanmin(yi)), float(np.nanmax(yi)))
    ax.set_axis_on()
    if show_legend:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        if y_ticklabels is not None:
            ticks = 0.5 * (np.asarray(yi[:-1], dtype=float) + np.asarray(yi[1:], dtype=float))
            ax.set_yticks(ticks, labels=list(y_ticklabels))
        cbar = fig.colorbar(p1, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label(cbar_label)
    else:
        ax.set_axis_off()
    return ax


@staticmethod
def hypno_density( probs , ax ):
   ax.clear()
   if len(probs) == 0: return
   pp_cols = ["PP_N1", "PP_N2", "PP_N3", "PP_R", "PP_W"]
   res = probs.reindex(columns=pp_cols, fill_value=0.0).copy()
   ne = len(res)
   x = np.arange(1, ne+1, 1)
   y = res.to_numpy(dtype=float)
   xsize = 20
   ysize=2.5
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Prob(stage)')
   ax.stackplot(x, y.T , colors = lp.stgcol([ 'N1','N2','N3','R','W']) )
   ax.set(xlim=(1, ne), xticks=[ 1 , ne ] , 
          ylim=(0, 1), yticks=np.arange(0, 1))                                                                                             
   return ax
