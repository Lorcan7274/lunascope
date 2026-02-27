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

from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import colormaps
from matplotlib import pyplot as plt

@staticmethod
def hypno(ss, e=None, ax=None, *, title=None, xsize=20, ysize=2, clear=True):
    """Plot a hypnogram into an existing Axes if provided."""
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

@staticmethod
def plot_hjorth( ch , ax , p , gui ):

    ax.clear()

    # get stats
    res = p.silent_proc_lunascope('EPOCH dur=30 verbose & SIGSTATS epoch sig=' + ch)
    df = res.get('SIGSTATS: CH_E')
    dt = res.get('EPOCH: E')
    if df is None or dt is None or len(df) == 0 or len(dt) == 0:
        return ax

    # Align Hjorth rows to epoch START using E, so gaps map consistently
    # with the spectrogram x-axis.
    if "E" in df.columns and "E" in dt.columns and "START" in dt.columns:
        dx = df[["E"]].merge(dt[["E", "START"]], on="E", how="left")
        if not dx["START"].notna().any():
            return ax
        x = dx["START"].to_numpy(float)
    elif "START" in dt.columns:
        x = dt["START"].to_numpy(float)
        if len(x) != len(df):
            x = x[:len(df)]
    else:
        return ax
    
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
    w = gui.spin_win.value()
    y1 = _norm(lp.winsorize( df["H1"].to_numpy(float) , limits = [w,w] ))
    y2 = _norm(lp.winsorize( df["H2"].to_numpy(float) , limits = [w,w] ))
    y3 = _norm(lp.winsorize( df["H3"].to_numpy(float) , limits = [w,w] ))

    # color axes
    idx2 = np.clip(np.rint(y2 * 99).astype(int), 0, 99)
    idx3 = np.clip(np.rint(y3 * 99).astype(int), 0, 99)
    colors2 = colormaps["turbo"](y2)  # y2 in [0,1]
    colors3 = colormaps["turbo"](y3)

    midy = 0
    elen = 30

    rects_top = [Rectangle((xi, midy), elen, hi) for xi, hi in zip(x, y1)]
    rects_bot = [Rectangle((xi, midy - hi), elen, hi) for xi, hi in zip(x, y1)]
    pc_top = PatchCollection(rects_top, facecolors=colors2, edgecolor="none", linewidth=0)
    pc_bot = PatchCollection(rects_bot, facecolors=colors3, edgecolor="none", linewidth=0)
    ax.add_collection(pc_top)
    ax.add_collection(pc_bot)


    fig = ax.figure
    # no auto layout padding
    fig.set_constrained_layout(False)      # or: fig.set_layout_engine(None)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # make the axes fill the figure
    ax.set_position([0, 0, 1, 1])
    # no data margins or axes decorations
    ax.margins(x=0, y=0)
    ax.set_axis_off()

    ax.set_xlim(0, max(x))
    ax.set_ylim(-1, 1)
    ax.margins(0)
    ax.axis("off")
    ax.figure.patch.set_facecolor("white")
    ax.set_aspect("auto")
    
    return ax


# --------------------------------------------------------------------------------
# plot a spectrogram
        
@staticmethod
def plot_spec( xi,yi,zi, ch, minf, maxf, ax , gui, clear = True):

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(xsize, ysize))
        created = True
    elif clear:
        ax.clear()
        
    if len(xi) == 0: return ax

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frequency (Hz)')
    ax.set( ylim=(minf,maxf) )
    p1 = ax.pcolormesh(xi, yi, zi, cmap = 'turbo' )
    if len(xi) > 1:
        ax.set_xlim(0, float(np.nanmax(xi)))
    ax.margins(x=0, y=0.2)
    return ax  


@staticmethod
def hypno_density( probs , ax ):
   ax.clear()
   if len(probs) == 0: return
   res = probs[ ["PP_N1","PP_N2","PP_N3","PP_R","PP_W" ]  ]
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
