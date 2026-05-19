
#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  LUNA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Luna. If not, see <http://www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

"""EEG topoplot rendering engine.

Two usage patterns:

1. One-shot (Results mode):
       draw_topo(ax, values, positions, ...)

2. Animated (Live mode) — avoids re-running interpolation each frame:
       renderer = TopoRenderer(positions, grid_res=150)
       renderer.setup(ax, fig, cmap='RdBu_r')
       ...
       renderer.update(values, vmin, vmax)  # called every frame
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HEAD_RADIUS = 0.5
_NOSE_W     = 0.045   # half-width of nose triangle at head edge
_NOSE_H     = 0.075   # height above head circle
_EAR_H      = 0.12    # vertical span of ear bump
_EAR_DX     = 0.035   # how far ear protrudes beyond head circle
_MIN_INTERP = 8       # minimum channels required to attempt interpolation
_GRID_PAD   = 1.35    # grid extends to HEAD_RADIUS * _GRID_PAD


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_TOPO_AX_RECT = [0.20, 0.08, 0.56, 0.84]
_TOPO_CBAR_RECT = [0.82, 0.18, 0.025, 0.64]


def create_topo_axes(fig):
    """Create fixed-position axes for the topo plot and its colorbar."""
    fig.clear()
    ax = fig.add_axes(_TOPO_AX_RECT)
    cax = fig.add_axes(_TOPO_CBAR_RECT)
    fig.patch.set_facecolor("#0d1117")
    return ax, cax


# ---------------------------------------------------------------------------
# Head outline helper
# ---------------------------------------------------------------------------

def _draw_head_outline(ax, color: str = "white", lw: float = 1.5, zorder: int = 4):
    """Draw head circle, nose, and ear stubs onto *ax*."""
    r = HEAD_RADIUS

    # head circle
    circ = plt.Circle((0, 0), r, fill=False, color=color, lw=lw, zorder=zorder)
    ax.add_patch(circ)

    # nose (triangle pointing up)
    nose_xs = [-_NOSE_W, 0.0, _NOSE_W]
    nose_ys = [r, r + _NOSE_H, r]
    ax.plot(nose_xs, nose_ys, color=color, lw=lw, solid_capstyle="round", zorder=zorder)

    # ears (small semicircular bumps, left and right)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 40)
    for side in (-1, 1):
        ear_x = side * (r + _EAR_DX * np.sin(np.linspace(0, np.pi, 40)))
        ear_y = _EAR_H * np.cos(theta)
        ax.plot(ear_x, ear_y, color=color, lw=lw, zorder=zorder)


# ---------------------------------------------------------------------------
# One-shot renderer
# ---------------------------------------------------------------------------

def draw_topo(
    ax,
    values:    dict[str, float],
    positions: dict[str, tuple[float, float]],
    *,
    cax=None,
    mode:        str   = "both",    # "dots" | "interp" | "both"
    cmap:        str   = "RdBu_r",
    vmin:        float | None = None,
    vmax:        float | None = None,
    dot_size:    float | None = None,
    show_labels: bool  = True,
    min_interp:  int   = _MIN_INTERP,
    grid_res:    int   = 180,
    bg:          str   = "#0d1117",
    fg:          str   = "#c9d1d9",
    label_fontsize: int = 6,
) -> None:
    """Render a topomap onto *ax* (one-shot, redraws from scratch)."""
    fig = ax.get_figure()
    fig.patch.set_facecolor(bg)
    ax.clear()
    ax.set_aspect("equal")
    ax.set_facecolor(bg)
    ax.set_axis_off()

    # gather matched channels
    shared = [ch for ch in values if ch in positions and np.isfinite(values[ch])]
    if not shared:
        ax.text(0.5, 0.5, "No matching channels", color=fg,
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        return

    xs = np.array([positions[ch][0] for ch in shared])
    ys = np.array([positions[ch][1] for ch in shared])
    zs = np.array([values[ch]       for ch in shared], dtype=float)

    if vmin is None: vmin = float(np.nanmin(zs))
    if vmax is None: vmax = float(np.nanmax(zs))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # --- interpolated surface ---
    do_interp = mode in ("interp", "both") and len(shared) >= min_interp
    if do_interp:
        from scipy.interpolate import griddata
        r    = HEAD_RADIUS * _GRID_PAD
        xi   = np.linspace(-r, r, grid_res)
        yi   = np.linspace(-r, r, grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi   = griddata((xs, ys), zs, (Xi, Yi), method="cubic")
        mask = (Xi ** 2 + Yi ** 2) > HEAD_RADIUS ** 2
        Zi   = np.ma.array(Zi, mask=(mask | ~np.isfinite(Zi)))
        ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, norm=norm,
                      shading="auto", zorder=1, rasterized=True)
    elif mode == "interp":
        ax.text(0.5, 0.03,
                f"N={len(shared)} < {min_interp} — dots only",
                color=fg, ha="center", va="bottom", fontsize=7,
                transform=ax.transAxes)

    # --- head outline ---
    _draw_head_outline(ax, color=fg, zorder=5)

    # --- electrode dots ---
    show_scatter = mode in ("dots", "both") or (mode == "interp" and not do_interp)
    if show_scatter:
        dot_size  = (70 if not do_interp else 35) if dot_size is None else float(dot_size)
        dot_alpha = 1.0 if not do_interp else 0.75
        ax.scatter(xs, ys, c=zs, cmap=cmap, norm=norm,
                   s=dot_size, zorder=6, edgecolors=bg, linewidths=0.6,
                   alpha=dot_alpha)

    # --- channel labels ---
    if show_labels:
        for ch in shared:
            px, py = positions[ch]
            ax.text(px, py + 0.03, ch,
                    color=fg, ha="center", va="bottom",
                    fontsize=label_fontsize, zorder=7)

    # --- colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax) if cax is not None else fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(colors=fg, labelsize=7)
    cbar.outline.set_edgecolor(fg)

    # axes limits
    pad = 0.18
    ax.set_xlim(-HEAD_RADIUS - pad, HEAD_RADIUS + pad)
    ax.set_ylim(-HEAD_RADIUS - pad, HEAD_RADIUS + _NOSE_H + pad)


# ---------------------------------------------------------------------------
# Animated renderer — caches grid and interpolation weights
# ---------------------------------------------------------------------------

class TopoRenderer:
    """Stateful topoplot renderer that caches interpolation weights.

    Call :meth:`setup` once (or whenever positions/cmap change), then
    :meth:`update` on every animation frame.  Only array data is modified
    in-place, avoiding expensive figure reconstruction.
    """

    def __init__(
        self,
        positions: dict[str, tuple[float, float]],
        *,
        grid_res:   int  = 150,
        min_interp: int  = _MIN_INTERP,
        bg:         str  = "#0d1117",
        fg:         str  = "#c9d1d9",
    ):
        self.positions   = positions
        self.grid_res    = grid_res
        self.min_interp  = min_interp
        self.bg          = bg
        self.fg          = fg

        self._ax         = None
        self._fig        = None
        self._cmap       = "RdBu_r"
        self._mesh       = None   # pcolormesh artist
        self._scatter    = None   # scatter artist
        self._cbar       = None   # colorbar
        self._head_drawn = False
        self._mode       = "both"
        self._dot_size   = None
        self._labels     = []

        # pre-compute interpolation grid and weights
        self._Xi = self._Yi = self._mask = None
        self._W  = None     # (n_grid_points × n_ch) weight matrix or None
        self._channels: list[str] = []
        self._do_interp = False
        self._precompute()

    # ------------------------------------------------------------------

    def _precompute(self):
        r    = HEAD_RADIUS * _GRID_PAD
        xi   = np.linspace(-r, r, self.grid_res)
        yi   = np.linspace(-r, r, self.grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        self._Xi   = Xi
        self._Yi   = Yi
        self._mask = (Xi ** 2 + Yi ** 2) > HEAD_RADIUS ** 2

        chs = list(self.positions.keys())
        self._channels  = chs
        self._do_interp = len(chs) >= self.min_interp

        if self._do_interp:
            # pre-compute RBF (thin-plate) interpolation weights so that
            # each frame only needs a matrix–vector multiply
            from scipy.interpolate import RBFInterpolator
            xs = np.array([self.positions[ch][0] for ch in chs])
            ys = np.array([self.positions[ch][1] for ch in chs])
            src = np.column_stack([xs, ys])
            dst = np.column_stack([Xi.ravel(), Yi.ravel()])
            # unit-impulse columns: weight matrix W so that Z_grid = W @ z_ch
            n_src = len(chs)
            W = np.zeros((dst.shape[0], n_src), dtype=np.float32)
            for i in range(n_src):
                e_i        = np.zeros(n_src)
                e_i[i]     = 1.0
                rbf        = RBFInterpolator(src, e_i, kernel="thin_plate_spline")
                W[:, i]    = rbf(dst).astype(np.float32)
            self._W = W

    # ------------------------------------------------------------------

    def setup(
        self,
        ax,
        fig,
        cax=None,
        cmap: str = "RdBu_r",
        dot_size: float | None = None,
        show_labels: bool = True,
        mode: str = "both",
    ):
        """Create all static artists on *ax*.  Must be called before :meth:`update`."""
        self._ax   = ax
        self._fig  = fig
        self._cax  = cax
        self._cmap = cmap
        self._mode = mode
        self._dot_size = dot_size

        ax.clear()
        ax.set_aspect("equal")
        ax.set_facecolor(self.bg)
        ax.set_axis_off()
        fig.patch.set_facecolor(self.bg)

        # placeholder mesh (data filled by update())
        dummy = np.zeros((self.grid_res, self.grid_res))
        dummy_m = np.ma.array(dummy, mask=self._mask)
        self._mesh = ax.pcolormesh(
            self._Xi, self._Yi, dummy_m,
            cmap=cmap, vmin=-1, vmax=1,
            shading="auto", zorder=1, rasterized=True,
        )
        self._mesh.set_visible(self._do_interp)

        # scatter
        chs = self._channels
        xs  = np.array([self.positions[ch][0] for ch in chs])
        ys  = np.array([self.positions[ch][1] for ch in chs])
        dot_size = (70 if not self._do_interp else 35) if dot_size is None else float(dot_size)
        self._scatter = ax.scatter(
            xs, ys, c=np.zeros(len(chs)), cmap=cmap,
            vmin=-1, vmax=1,
            s=dot_size, zorder=6, edgecolors=self.bg, linewidths=0.6,
        )
        self._apply_mode()

        # head outline
        _draw_head_outline(ax, color=self.fg, zorder=5)

        # labels
        if show_labels:
            for ch in chs:
                px, py = self.positions[ch]
                self._labels.append(
                    ax.text(px, py + 0.03, ch,
                            color=self.fg, ha="center", va="bottom",
                            fontsize=6, zorder=7)
                )

        # colorbar
        self._cbar = (
            fig.colorbar(self._mesh, cax=cax)
            if cax is not None else
            fig.colorbar(self._mesh, ax=ax)
        )
        self._cbar.ax.tick_params(colors=self.fg, labelsize=7)
        self._cbar.outline.set_edgecolor(self.fg)

        pad = 0.18
        ax.set_xlim(-HEAD_RADIUS - pad, HEAD_RADIUS + pad)
        ax.set_ylim(-HEAD_RADIUS - pad, HEAD_RADIUS + _NOSE_H + pad)

    # ------------------------------------------------------------------

    def update(
        self,
        values: dict[str, float],
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """Push new data to the existing artists (fast, no redraw of statics)."""
        if self._ax is None:
            return

        chs = self._channels
        zs  = np.array(
            [values.get(ch, np.nan) for ch in chs], dtype=np.float32
        )
        finite = np.isfinite(zs)
        if not finite.any():
            return

        _vmin = float(np.nanmin(zs)) if vmin is None else vmin
        _vmax = float(np.nanmax(zs)) if vmax is None else vmax
        if _vmin == _vmax:
            _vmin -= 1.0
            _vmax += 1.0
        norm = plt.Normalize(vmin=_vmin, vmax=_vmax)

        # update mesh
        if self._do_interp and self._W is not None:
            z_safe = np.where(finite, zs, 0.0)
            Zi_flat = self._W @ z_safe
            Zi = Zi_flat.reshape(self.grid_res, self.grid_res)
            Zi_m = np.ma.array(Zi, mask=self._mask)
            self._mesh.set_array(Zi_m.ravel())
            self._mesh.set_norm(norm)
            self._mesh.set_visible(True)

        # update scatter
        self._scatter.set_array(zs)
        self._scatter.set_norm(norm)
        if self._dot_size is not None:
            self._scatter.set_sizes(np.full(len(chs), float(self._dot_size)))

        # update colorbar limits
        if self._cbar is not None:
            self._cbar.mappable.set_norm(norm)
            self._cbar.update_normal(self._cbar.mappable)
        self._apply_mode()

    def set_mode(self, mode: str):
        self._mode = mode
        self._apply_mode()

    def _apply_mode(self):
        show_mesh = self._mode in ("interp", "both") and self._do_interp
        show_scatter = self._mode in ("dots", "both") or (
            self._mode == "interp" and not self._do_interp
        )
        if self._mesh is not None:
            self._mesh.set_visible(show_mesh)
        if self._scatter is not None:
            self._scatter.set_visible(show_scatter)

    def set_dot_size(self, dot_size: float | None):
        self._dot_size = dot_size
        if self._scatter is not None and dot_size is not None:
            n = len(self._channels)
            self._scatter.set_sizes(np.full(n, float(dot_size)))
