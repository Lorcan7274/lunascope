
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

import re
from pathlib import Path
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QSignalBlocker

from .tbl_funcs import set_filter_for_channel
from ..helpers import override_colors

# helper to parse and store cmaps


#  [par]

#  [mod]
#    label ch= type= [ f= ]
#  [pal]
#    label 18-cols
#  [sig]
#    label [c=]col ylim=lwr,upr mod=label,pal [f=lwr,upr] 
#  [ann]
#    label [c=]col

# [sig] & [ann] determine ordering

# mod = ch , type , ( f=lwr,upr } 

#
#  ch 

import os, sys
import pyqtgraph as pg
from  ..helpers import random_darkbg_colors

_CMAP_DEFAULT_TEXT = """\
%  -- lunascope config guide --
%  % = comment char
%  ... = continue previous entry
%  Sections: [par][mod][pal][sig]
%            [ann]
%  [par] assumed at top if absent.
%
%  [par] global settings
%  line-weight = 1      (float)
%  show-lines  = Y      (Y/N)
%  na-token    = NA
%  table-allow-empty = N
%  day-anchor  = 12     (0-23)
%  pops-path   = ~/pops/
%  pops-model  = s2
%  Docks (Y/N):
%   project-dock  settings-dock
%   console-dock  hypnogram-dock
%   spectrogram-dock  mask-dock
%   signal-dock   annots-dock
%   instance-dock outputs-dock
%
%  [mod] signal modulator
%  label ch=CHAN type=TYPE
%        [f=L,U] [bins=abs|pct]
%  type: raw | amp | phase
%  bins: abs=range  pct=percentile
%  f: bandpass Hz
%     (required for amp, phase)
%  mod SR != display SR is OK
%
%  [pal] custom 18-colour palette
%  label col1..col18 (#rrggbb)
%  Exactly 18; may span lines.
%  bin0=lowest  bin17=highest
%
%  Builtins - amplitude:
%   gray   black->white
%   hot    blk->red->yellow->white
%   cool   navy->teal->cyan
%   ember  dark->red->orange
%   plasma violet->magenta->yellow
%   thermal infrared camera
%   aurora dark->teal->green
%   saturation gray->gold
%
%  Builtins - phase (bin0=-pi):
%   rwb    blue->gray->red
%   rainbow full HSV 20-deg steps
%
%  Threshold builtins (bins=pct):
%   w10  top 11% white / else blk
%   w20  top 22% white / else blk
%   r10  top 11% red   / else blk
%   r20  top 22% red   / else blk
%
%  Builtins - decorative:
%   flux violet->blue->cyan->yell
%
%  [sig] signal display
%  label [col=C] [ylim=L,U]
%        [y=V,...] [f=L,U]
%        [mod=MOD,PAL]
%  col:  colour (hex or named)
%  ylim: fixed y-axis scale
%  y:    guide lines
%  f:    bandpass Hz
%  mod:  sigmod + palette
%
%  [ann] annotation colours
%  label [col=C]
%  Stage cols (W N1 N2 N3 R ..)
%  built-in; col= overrides.
%  Order here = display order.
"""

class CMapsMixin:

    def _init_cmaps(self):

        self.cfg = { } 
        self.cmap_explicit_ann_cols = set()

        # extracted : ch -> mod_label
        self.sigmods = { } 

        self.sigmod_colors = {}

        self.cmap_fixed_min = {}
        self.cmap_fixed_max = {}

        self.cmap_ylines = { } # ch -> [ y-vals ]
        self.cmap_ylines_idx = { }
        self.cmap_n_ylines = 0

        cached_pops_path = None
        if hasattr(self, "_load_cached_pops_path"):
            cached_pops_path = self._load_cached_pops_path()

        self.cfg_pops_path = cached_pops_path or '~/dropbox/pops/'
        self.cfg_pops_model = 's2'

        self.ui.txt_pops_path.setText( self.cfg_pops_path )
        self.ui.txt_pops_model.setText( self.cfg_pops_model )

        self.cmap_use_na_for_empty = True
        self.cmap_na_token = "NA"
        self.cfg_day_anchor = 12

        # show reference text on first load (when widget is still empty)
        if not self.ui.txt_cmap.toPlainText().strip():
            self.ui.txt_cmap.setPlainText(_CMAP_DEFAULT_TEXT)


    #
    # clear the current cmap
    #
    
    def _clear_cmaps(self):
        self._init_cmaps()
        
    
    #
    # apply the current cmap (on attach/refresh) only called on 'render'
    #  --> colors, y-scales, ordering, etc
    #  (sigmod is only done at render stage)
    
    def _apply_cmaps(self):

        # reset priors
        self._clear_cmaps()
        
        # helpers
        def ucfirst(s: str) -> str:
            return s[:1].upper() + s[1:]

        def istrue(s: str) -> bool:
            return ucfirst(s) in [ 'Y', 'T', '1' ]

        # take current text in        
        text = self.ui.txt_cmap.toPlainText()
        try:
            self.cfg = parse_cmap(text)
        except ValueError as e:
            QMessageBox.critical(self.ui, "CMap parse error", str(e))
            return False

        # load sigmod palettes: builtins + any [pal] sections from cfg
        self.sigmod_colors[ 'rwb'        ] = self.rwb_sigmod_colors
        self.sigmod_colors[ 'gray'       ] = self.gray_sigmod_colors
        self.sigmod_colors[ 'hot'        ] = self.hot_sigmod_colors
        self.sigmod_colors[ 'cool'       ] = self.cool_sigmod_colors
        self.sigmod_colors[ 'ember'      ] = self.ember_sigmod_colors
        self.sigmod_colors[ 'plasma'     ] = self.plasma_sigmod_colors
        self.sigmod_colors[ 'thermal'    ] = self.thermal_sigmod_colors
        self.sigmod_colors[ 'aurora'     ] = self.aurora_sigmod_colors
        self.sigmod_colors[ 'saturation' ] = self.saturation_sigmod_colors
        self.sigmod_colors[ 'rainbow'    ] = self.rainbow_sigmod_colors
        self.sigmod_colors[ 'w10'        ] = self.w10_sigmod_colors
        self.sigmod_colors[ 'w20'        ] = self.w20_sigmod_colors
        self.sigmod_colors[ 'r10'        ] = self.r10_sigmod_colors
        self.sigmod_colors[ 'r20'        ] = self.r20_sigmod_colors
        self.sigmod_colors[ 'flux'       ] = self.flux_sigmod_colors
        for pal_label, colors in self.cfg['pal'].items():
            self.sigmod_colors[ pal_label ] = colors
        
        #
        # set various cfg vars
        #

        self.cfg_line_weight = 1;

        self.cfg_show_zero_line = True;

        if 'line-weight' in self.cfg['par']:
            self.cfg_line_weight = float( self.cfg['par']['line-weight'] )
            if getattr(self, "_line_weight_spin", None) is not None:
                b = QSignalBlocker(self._line_weight_spin)
                self._line_weight_spin.setValue(self.cfg_line_weight)
                del b
            
        if 'show-lines' in self.cfg['par']:
            t = self.cfg['par']['show-lines']            
            if t == "1" or t == "Y" or t == "T":
                self.cfg_show_zero_line = True
            else:
                self.cfg_show_zero_line = False

        # copy/save tbls: treatment of empty values
        if 'table-allow-empty' in self.cfg['par']:
            self.cmap_use_na_for_empty = not istrue( self.cfg['par']['table-allow-empty'] )
                       
        if 'na-token' in self.cfg['par']:
            self.cmap_na_token = self.cfg['par']['na-token']                       
            
        # POPS

        if 'pops-path' in self.cfg['par']:
            self.cfg_pops_path = self.cfg['par']['pops-path']
            if hasattr(self, "_set_pops_path"):
                self._set_pops_path(self.cfg_pops_path)
            else:
                self.ui.txt_pops_path.setText(self.cfg_pops_path)
	    
        if 'pops-model' in self.cfg['par']:
            self.cfg_pops_model = self.cfg['par']['pops-model']
            self.ui.txt_pops_model.setText( self.cfg_pops_model )

        if 'day-anchor' in self.cfg['par']:
            try:
                v = int(self.cfg['par']['day-anchor'])
                if 0 <= v <= 23:
                    self.cfg_day_anchor = v
            except (ValueError, TypeError):
                pass

                        
        # dock viz
        if 'project-dock' in self.cfg['par']:
            t = self.cfg['par']['project-dock']
            self.ui.dock_slist.setVisible( istrue(t) )
                
        if 'settings-dock' in self.cfg['par']:
            t = self.cfg['par']['settings-dock']
            self.ui.dock_settings.setVisible( istrue(t) )

        if 'console-dock' in self.cfg['par']:
            t = self.cfg['par']['console-dock']
            self.ui.dock_console.setVisible( istrue(t) )

        if 'hypnogram-dock' in self.cfg['par']:
            t = self.cfg['par']['hypnogram-dock']
            self.ui.dock_hypno.setVisible( istrue(t) )
        
        if 'spectrogram-dock' in self.cfg['par']:
            t = self.cfg['par']['spectrogram-dock']
            self.ui.dock_spectrogram.setVisible( istrue(t) )
                
        if 'mask-dock' in self.cfg['par']:
            t = self.cfg['par']['mask-dock']
            self.ui.dock_mask.setVisible( istrue(t) )

        if 'signal-dock' in self.cfg['par']:
            t = self.cfg['par']['signal-dock']
            self.ui.dock_sig.setVisible( istrue(t) )

        if 'annots-dock' in self.cfg['par']:
            t = self.cfg['par']['annots-dock']
            self.ui.dock_annot.setVisible( istrue(t) )

        if 'instance-dock' in self.cfg['par']:
            t = self.cfg['par']['instance-dock']
            self.ui.dock_annots.setVisible( istrue(t) )

        if 'outputs-dock' in self.cfg['par']:
            t = self.cfg['par']['outputs-dock']
            self.ui.dock_outputs.setVisible( istrue(t) )

        #
        # channel filters
        #

        for ch, ch_spec in self.cfg['sig'].items():
            if 'f' in ch_spec:
                fcode = ch_spec['f']
                if fcode is not None:
                    lwr = fcode[0]
                    upr = fcode[1]
                    if lwr < upr and lwr >= 0 and upr >= 0:
                        # user-specific filter map: { ch : [ lwr , upr ] }
                        self.user_fmap_frqs[ ch ] = [ lwr , upr ]                        
                    else:
                        print( 'unknown filter values:' , fcode )

        #
        # channel ylims
        #

        for ch, ch_spec in self.cfg['sig'].items():
            if 'ylim' in ch_spec:
                ycode = ch_spec['ylim']
                if ycode is not None:
                    lwr = ycode[0]
                    upr = ycode[1]
                    if lwr < upr:                        
                        self.cmap_fixed_min[ ch ] = lwr
                        self.cmap_fixed_max[ ch ] = upr
                    else:
                        print( 'unknown ylim value:' , ycode )


        # c-map y-lines
        self.cmap_n_ylines = 0
        for ch, ch_spec in self.cfg['sig'].items():
            if 'y' in ch_spec:
                ycode = ch_spec['y']
                if ycode is not None:
                    ycode = list( ycode ) 
                    self.cmap_ylines[ch] = ycode
                    n = len( ycode )
                    self.cmap_ylines_idx[ch] = list( range( self.cmap_n_ylines , self.cmap_n_ylines + n ) ) 
                    self.cmap_n_ylines += len( self.cmap_ylines[ch] )

        
        #
        # channel orders/colors
        #

        self.cmap = {}
        self.cmap_list = [ ]
        self.cmap_rlist = [ ]

        for ch in self.cfg['sig_order']:
            # ch order
            self.cmap_list.append( ch )
            # optionaly, a color
            col = self.cfg['sig'][ch].get('col')
            if col is not None and str(col).strip() != "":
                self.cmap[ch] = col
                
        for ann in self.cfg['ann_order']:
            # ch order
            self.cmap_list.append( ann )
            # optionaly, a color
            col = self.cfg['ann'][ann].get('col')
            if col is not None and str(col).strip() != "":
                self.cmap[ann] = col
                self.cmap_explicit_ann_cols.add(ann)

        # reverse order (for plotting goes y 0 - 1 is bottom - top currently
        # and can't be bothered to fix
        self.cmap_rlist = list(reversed(self.cmap_list))
                
        # and flag that we'll have bespoke as the default
        if len(self.cmap) != 0:            
            self.palset = 'bespoke'

        return True

                
    
                    
    #
    # render-stage cmaps
    #

    def _render_cmaps(self):

        # segsrv.make_sigmod( label , ch , type , SR, filter-order, lwr, upr

        # Reset sigmods before rebuilding so stale entries don't survive
        # when the user removes all [mod] entries from the config.
        self.sigmods = {}

        # generate all mods
        for mod_label, mod_spec in self.cfg['mod'].items():

            okay = True;

            ch1 = mod_spec[ 'ch' ]
            if ch1 not in self.srs:
                okay = False

            type1 = mod_spec[ 'type' ]
            if type1 not in [ 'raw' , 'amp' , 'phase' ]:
                okay = False

            # append _pct suffix so C++ uses percentile-based binning
            if mod_spec.get('bins', 'abs') == 'pct':
                type1 = type1 + '_pct'

            frqs = mod_spec.get('f')
            if okay is False:
                print( f"[sigmod] skipping '{mod_label}': channel '{ch1}' not found or type invalid" )
            elif frqs is not None:
                self.ss.make_sigmod( mod_label , ch1, type1 , self.srs[ch1], 4, frqs[0] , frqs[1] )
            else:
                self.ss.make_sigmod_raw( mod_label , ch1, type1 )

        # map ch -> mod (done once after all mods are registered, not per-iteration)
        for ch, ch_spec in self.cfg['sig'].items():
            if 'mod' in ch_spec and ch_spec['mod'] is not None:
                self.sigmods[ ch ] = ch_spec['mod']
                
                    
    #
    # handle palettes
    #

    def _init_colors(self):

        self.cmap = {}

        self.cmap_list = [ ]

        self.cmap_rlist = [ ] 

        self.stgcols_hex = {
            'N1': '#20B2DA',  # rgba(32,178,218,1)
            'N2': '#0000FF',  # blue
            'N3': '#000080',  # navy
            'R':  '#FF0000',  # red
            'SP': '#800080',  # purple (blend of NREM blue and REM red)
            'WP': '#008000',  # green (CSS "green")
            'W':  '#008000',  # green (CSS "green")
            '?':  '#808080',  # gray
            'L':  '#FFFF00',  # yellow
        }

        # signal modulation colors: red-white-blue
        # peaks: pure red (pos) / pure blue (neg); ZC fades to light gray
        self.rwb_sigmod_colors = [
        (  0,   0, 255),  #  0  -pi      neg peak (blue)
        ( 53,  53, 244),  #  1
        (105, 105, 233),  #  2
        (158, 158, 221),  #  3
        (210, 210, 210),  #  4  -pi/2    ZC (light gray)
        (219, 168, 168),  #  5
        (228, 126, 126),  #  6
        (237,  84,  84),  #  7
        (246,  42,  42),  #  8
        (255,   0,   0),  #  9   0       pos peak (red)
        (244,  53,  53),  # 10
        (233, 105, 105),  # 11
        (221, 158, 158),  # 12
        (210, 210, 210),  # 13  +pi/2    ZC (light gray)
        (158, 158, 221),  # 14
        (105, 105, 233),  # 15
        ( 53,  53, 244),  # 16
        (  0,   0, 255),  # 17  +pi      neg peak (blue, wrap)
        ]

        # ── Amplitude palettes (bin 0 = lowest, bin 17 = highest) ──────────

        # gray: near-black → white  (dark-background friendly)
        self.gray_sigmod_colors = [
        ( 20, 20, 20), ( 34, 34, 34), ( 48, 48, 48), ( 61, 61, 61),
        ( 75, 75, 75), ( 89, 89, 89), (103,103,103), (117,117,117),
        (131,131,131), (144,144,144), (158,158,158), (172,172,172),
        (186,186,186), (200,200,200), (214,214,214), (227,227,227),
        (241,241,241), (255,255,255),
        ]

        # hot: black → red → yellow → white  (classic thermal heatmap)
        self.hot_sigmod_colors = [
        (  0,  0,  0), ( 45,  0,  0), ( 90,  0,  0), (135,  0,  0),
        (180,  0,  0), (225,  0,  0), (255, 15,  0), (255, 60,  0),
        (255,105,  0), (255,150,  0), (255,195,  0), (255,240,  0),
        (255,255, 30), (255,255, 75), (255,255,120), (255,255,165),
        (255,255,210), (255,255,255),
        ]

        # cool: dark navy → ocean blue → teal → light cyan
        self.cool_sigmod_colors = [
        (  0,  0, 50), (  0, 10, 80), (  0, 25,110), (  0, 45,140),
        (  0, 70,165), (  0, 95,185), (  0,120,200), (  0,145,210),
        (  0,165,215), (  0,185,218), ( 20,200,220), ( 50,210,225),
        ( 80,218,230), (110,225,235), (140,230,240), (165,235,245),
        (190,240,250), (215,245,255),
        ]

        # ember: near-black → deep red → orange → pale warm white
        #        (like glowing coals; no blue channel)
        self.ember_sigmod_colors = [
        ( 10,  0,  0), ( 30,  0,  0), ( 60,  0,  0), (100,  0,  0),
        (140,  0,  0), (180, 20,  0), (210, 50,  0), (230, 80,  0),
        (245,110,  0), (255,140,  0), (255,165,  0), (255,185, 20),
        (255,200, 50), (255,215, 80), (255,225,120), (255,235,160),
        (255,244,200), (255,252,240),
        ]

        # plasma: dark blue-violet → magenta → orange → bright yellow
        #         (matplotlib plasma-inspired)
        self.plasma_sigmod_colors = [
        ( 13,  8,135), ( 41,  7,143), ( 70,  6,152), ( 98,  4,160),
        (127,  3,168), (143, 17,158), (158, 31,149), (174, 45,139),
        (189, 58,130), (204, 71,120), (215, 90,106), (226,110, 92),
        (237,129, 78), (248,148, 64), (246,173, 56), (244,198, 48),
        (242,224, 40), (240,249, 33),
        ]

        # thermal: infrared-camera style
        #          black → purple → blue → teal → green → yellow → orange → warm white
        self.thermal_sigmod_colors = [
        (  0,  0,  0), ( 13,  0, 33), ( 27,  0, 67), ( 40,  0,100),
        ( 27,  0,140), ( 13,  0,180), (  0,  0,220), (  0, 33,180),
        (  0, 67,140), (  0,100,100), ( 67,133, 67), (133,167, 33),
        (200,200,  0), (218,150,  0), (237,100,  0), (255, 50,  0),
        (255,135, 90), (255,220,180),
        ]

        # aurora: dark → teal → vivid green → yellow-green
        #         (aurora borealis feel on a dark background)
        self.aurora_sigmod_colors = [
        (  0,  5, 15), (  0, 19, 29), (  0, 33, 43), (  0, 47, 57),
        (  0, 60, 70), (  0, 80, 73), (  0,100, 75), (  0,120, 78),
        (  0,140, 80), ( 20,155, 70), ( 40,170, 60), ( 60,185, 50),
        ( 80,200, 40), (113,212, 33), (147,224, 27), (180,235, 20),
        (210,245, 90), (240,255,160),
        ]

        # saturation: fully desaturated gray (low amp) → saturated gold (high amp)
        self.saturation_sigmod_colors = [
        (128,128,128), (136,132,121), (143,136,113), (150,140,106),
        (158,145, 98), (165,149, 90), (173,153, 82), (180,158, 75),
        (188,162, 67), (195,166, 60), (203,170, 52), (210,175, 45),
        (218,179, 37), (225,183, 30), (233,187, 22), (240,192, 14),
        (248,196,  7), (255,200,  0),
        ]

        # ── Phase/cyclic palettes ────────────────────────────────────────────

        # rainbow: full HSV spectrum in 20° steps (good for cyclic phase mods)
        #          bin 0 = -pi (red), wraps back at bin 17
        self.rainbow_sigmod_colors = [
        (255,  0,  0), (255, 85,  0), (255,170,  0), (255,255,  0),
        (170,255,  0), ( 85,255,  0), (  0,255,  0), (  0,255, 85),
        (  0,255,170), (  0,255,255), (  0,170,255), (  0, 85,255),
        (  0,  0,255), ( 85,  0,255), (170,  0,255), (255,  0,255),
        (255,  0,170), (255,  0, 85),
        ]

        # ── Discrete threshold palettes ──────────────────────────────────────
        # w10 / w20: top N% of amplitude shown as white, rest near-black
        # r10 / r20: same but red highlight (useful with bins=pct)

        _BLK = (12, 12, 12)
        _WHT = (255,255,255)
        _RED = (220, 30, 30)

        # w10 ≈ top 11% (bins 16-17 of 18)
        self.w10_sigmod_colors = [_BLK]*16 + [_WHT]*2

        # w20 ≈ top 22% (bins 14-17 of 18)
        self.w20_sigmod_colors = [_BLK]*14 + [_WHT]*4

        # r10 / r20: red variant of above
        self.r10_sigmod_colors = [_BLK]*16 + [_RED]*2
        self.r20_sigmod_colors = [_BLK]*14 + [_RED]*4

        # ── Arbitrary cool palettes ──────────────────────────────────────────

        # flux: dark violet → vivid blue → cyan → bright green → yellow-green
        #       (flowing "energy" look; vivid on dark backgrounds)
        self.flux_sigmod_colors = [
        ( 20,  0, 40), ( 35,  0, 80), ( 50,  0,120), ( 65,  0,160),
        ( 80,  0,200), ( 60, 20,214), ( 40, 40,228), ( 20, 60,241),
        (  0, 80,255), (  0,123,240), (  0,167,225), (  0,210,210),
        (  7,220,167), ( 14,230,123), ( 20,240, 80), ( 93,245, 53),
        (167,250, 27), (240,255,  0),
        ]

    def _set_default_palette(self):        
        if not hasattr(self, 'palset'):
            self._set_spectrum_palette()
            self.palset = 'spectrum'


    def set_palette(self):
        if not hasattr(self, 'palset'):
            self._set_default_palette()
        if self.palset == 'spectrum': self._set_spectrum_palette()
        if self.palset == 'white': self._set_white_palette()
        if self.palset == 'black': self._set_black_palette()
        if self.palset == 'muted': self._set_muted_palette()
        if self.palset == 'random': self._set_random_palette()
        if self.palset == 'bespoke': self._set_bespoke_palette()
        if self.palset == 'user': self._set_user_palette()
            
    def _set_spectrum_palette(self):
        self.palset = 'spectrum'
        self.ui.pg1.setBackground('black')        
        nchan = len( self.ui.tbl_desc_signals.checked() )
        self.colors = [pg.intColor(i, hues=nchan) for i in range(nchan)]
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = [pg.intColor(i, hues=nanns) for i in range(nanns)]
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()
        
    def _set_white_palette(self):        
        self.palset = 'white'
        self.ui.pg1.setBackground('#E0E0E0')
        nchan = len( self.ui.tbl_desc_signals.checked() )      
        self.colors = ['#101010'] * nchan
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = ['#101010'] * nanns
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()

    def _set_muted_palette(self):
        self.palset = 'muted'
        self.ui.pg1.setBackground('#D0C0D0')
        nchan = len( self.ui.tbl_desc_signals.checked() )
        self.colors = ['#403020'] * nchan
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = ['#403020'] * nanns
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()

    def _set_black_palette(self):
        self.palset = 'black'
        self.ui.pg1.setBackground('#101010')
        nchan = len( self.ui.tbl_desc_signals.checked() )
        self.colors = ['#E0E0E0'] * nchan
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = ['#E0E0E0'] * nanns
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()
        
    def _set_random_palette(self):
        self.palset = 'random'
        self.ui.pg1.setBackground('#101010')
        nchan = len( self.ui.tbl_desc_signals.checked() )
        self.colors = random_darkbg_colors( nchan )
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = random_darkbg_colors( nanns )
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()

    def _select_user_palette(self):
        self.palset = 'user'
        self.c1, self.c2 = pick_two_colors()
        self.ui.pg1.setBackground(self.c1)
        nchan = len( self.ui.tbl_desc_signals.checked() )
        self.colors = [self.c2] * nchan
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = [self.c2] * nanns
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()

    def _set_user_palette(self):
        self.palset = 'user'
        # assume self.c1 and self.c2 already set
        #self.c1, self.c2 = pick_two_colors()
        self.ui.pg1.setBackground(self.c1)
        nchan = len( self.ui.tbl_desc_signals.checked() )
        self.colors = [self.c2] * nchan
        anns = self.ui.tbl_desc_annots.checked()
        nanns = len( anns )
        self.acolors = [self.c2] * nanns
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()
        
    def _set_bespoke_palette(self):        
        # back default black (i.e. for things not seen)
        self._set_black_palette()
        self.palset = 'bespoke'
        chs = self.ui.tbl_desc_signals.checked()
        # re-order list
        if self.cmap_list:
            chs = sorted( chs, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + chs.index(x)))
            chs.reverse()
        nchan = len( chs )
        # set signal colors
        self.colors = override_colors(self.colors, chs, self.cmap)
        # and annots
        anns = self.ui.tbl_desc_annots.checked()
        if self.cmap_rlist:
            anns = sorted( anns, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + anns.index(x)))
        self.acolors = override_colors(self.acolors, anns, self.cmap)
        self.acolors = self._update_stage_cols( self.acolors , anns )
        self._update_cols()


    def _update_stage_cols(self,pal,anns):
        updated = []
        explicit = getattr(self, "cmap_explicit_ann_cols", set())
        for a_i, p_i in zip(anns, pal):
            if a_i in explicit and a_i in self.cmap:
                updated.append(self.cmap[a_i])
            else:
                updated.append(self.stgcols_hex.get(a_i, p_i))
        return updated

    
    def _load_palette(self):
        txt_file, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Open color map",
            "",
            "Text (*.txt *.map *.pal);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if txt_file:
            try:
                text = open(txt_file, "r", encoding="utf-8").read()
                
                self.cmap = {}
                self.cmap_list = [ ]
                self.cmap_rlist = [ ] 
                self.cmap_explicit_ann_cols = set()
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace("=", " ").replace("\t", " ").split()
                    if len(parts) >= 2:
                        self.cmap[parts[0]] = parts[1]
                        self.cmap_list.append( parts[0] )

                # reverse order (for plotting goes y 0 - 1 is bottom - top currently
                # and can't be bothered to fix
                self.cmap_rlist = list(reversed(self.cmap_list))
                
                # and set them
                self._set_bespoke_palette()
                
            except (UnicodeDecodeError, OSError) as e:
                QMessageBox.critical(
                    self.ui,
                    "Error opening color map",
                    f"Could not load {txt_file}\nException: {type(e).__name__}: {e}"
                )
            
    def _update_cols(self):
        lw = float(getattr(self, "cfg_line_weight", 1.0))
        for c, col in zip(self.curves, self.colors):
            c.setPen(pg.mkPen(col, width=lw, cosmetic=True))
        for c, col in zip(self.sigmod_curves, self.sigmod_curve_colors):
            c.setPen(pg.mkPen(col, width=lw, cosmetic=True))
        for c, col in zip(self.annot_curves, self.acolors):
            c.setPen(pg.mkPen(col, width=lw, cosmetic=True))



            
        

# --------------------------------------------------------------------------------
# cmap file format parser




def parse_cmap(text: str):
    """
    Parse cmap text with sections [par], [mod], [pal], [sig], [ann].
    
    Returns a dict:
        {
          "par": { "key":value, ... },
          "mod": { label: {"ch": str, "type": str, "f": (lwr, upr) or None}, ... },
          "pal": { label: [color1, ..., color18], ... },
          "sig": { label: {"col": str|None,
                           "ylim": (lwr,upr)|None,
                           "y": [y0,y1,y2,...]|None
                           "f": (lwr,upr)|None,
                           "mod": {"mod": mod_label, "pal": pal_label} | None},
                   ... },
          "ann": { label: {"col": str|None}, ... },
          "sig_order": [label1, ...],
          "ann_order": [label1, ...],
        }
    """
    cmap = {
        "par": {},
        "mod": {},
        "pal": {},
        "sig": {},
        "ann": {},
        "sig_order": [],
        "ann_order": [],
    }

    BUILTIN_PALETTES = {
        "rwb",
        # amplitude (linear)
        "gray", "hot", "cool", "ember", "plasma", "thermal", "aurora", "saturation",
        # phase / cyclic
        "rainbow",
        # discrete threshold
        "w10", "w20", "r10", "r20",
        # arbitrary
        "flux",
    }

    # assume [par] at start 
    current_section = 'par'
    
    # For [pal] we may span multiple lines to collect 18 colors
    pending_pal_label = None
    pending_pal_colors = []

    def finish_pending_palette():
        nonlocal pending_pal_label, pending_pal_colors
        if pending_pal_label is not None:
            if len(pending_pal_colors) != 18:
                raise ValueError(
                    f"Palette '{pending_pal_label}' has {len(pending_pal_colors)} colors; expected 18"
                )
            cmap["pal"][pending_pal_label] = pending_pal_colors[:]
            pending_pal_label = None
            pending_pal_colors = []

    # pending entry buffer: supports '+' continuation lines in [mod]/[sig]/[ann]
    pending_tokens = []
    pending_lineno = None

    def process_entry(tokens, lineno, section):
        """Commit a complete buffered entry (label + fields) for mod/sig/ann."""
        if not tokens:
            return
        label = tokens[0]
        fields = tokens[1:]

        if section == "mod":
            rec = {"ch": None, "type": None, "f": None, "bins": "abs"}
            for field in fields:
                if field.startswith("ch="):
                    rec["ch"] = field[3:]
                elif field.startswith("type="):
                    rec["type"] = field[5:]
                elif field.startswith("f="):
                    vals = field[2:].split(",")
                    if len(vals) != 2:
                        raise ValueError(
                            f"Bad f= in [mod] '{label}' on line {lineno}: {field}"
                        )
                    rec["f"] = (float(vals[0]), float(vals[1]))
                elif field.startswith("bins="):
                    v = field[5:]
                    if v not in ("abs", "pct"):
                        raise ValueError(
                            f"bins= in [mod] '{label}' must be 'abs' or 'pct' (line {lineno})"
                        )
                    rec["bins"] = v
                else:
                    raise ValueError(
                        f"Unknown token in [mod] '{label}' on line {lineno}: {field}"
                    )
            if rec["ch"] is None or rec["type"] is None:
                raise ValueError(
                    f"[mod] '{label}' missing ch= or type= (line {lineno})"
                )
            cmap["mod"][label] = rec

        elif section == "sig":
            rec = {
                "col": None,
                "ylim": None,
                "y": None,
                "f": None,
                "mod": None,
            }
            for field in fields:
                if field.startswith("col="):
                    if rec["col"] is not None:
                        raise ValueError(
                            f"Duplicate col in [sig] '{label}' on line {lineno}"
                        )
                    rec["col"] = field[4:]
                elif field.startswith("ylim="):
                    vals = field[5:].split(",")
                    if len(vals) != 2:
                        raise ValueError(
                            f"Bad ylim= in [sig] '{label}' on line {lineno}: {field}"
                        )
                    rec["ylim"] = (float(vals[0]), float(vals[1]))
                elif field.startswith("y="):
                    if rec["y"] is not None:
                        raise ValueError(
                            f"Duplicate y in [sig] '{label}' on line {lineno}"
                        )
                    rec["y"] = [float(x) for x in field[2:].split(",") if x]
                elif field.startswith("f="):
                    vals = field[2:].split(",")
                    if len(vals) != 2:
                        raise ValueError(
                            f"Bad f= in [sig] '{label}' on line {lineno}: {field}"
                        )
                    rec["f"] = (float(vals[0]), float(vals[1]))
                elif field.startswith("mod="):
                    vals = field[4:].split(",")
                    if len(vals) != 2:
                        raise ValueError(
                            f"Bad mod= in [sig] '{label}' on line {lineno}: {field}"
                        )
                    rec["mod"] = {"mod": vals[0], "pal": vals[1]}
                else:
                    raise ValueError(
                        f"Unknown token in [sig] '{label}' on line {lineno}: {field}"
                    )
            labels = label.split(",")
            if any(lbl == "" for lbl in labels):
                raise ValueError(
                    f"Bad signal label list in [sig] on line {lineno}: '{label}'"
                )
            for sig_label in labels:
                rec1 = rec.copy()
                if rec["y"] is not None:
                    rec1["y"] = list(rec["y"])
                if rec["mod"] is not None:
                    rec1["mod"] = dict(rec["mod"])
                cmap["sig"][sig_label] = rec1
                cmap["sig_order"].append(sig_label)

        elif section == "ann":
            rec = {"col": None}
            for field in fields:
                if field.startswith("col="):
                    if rec["col"] is not None:
                        raise ValueError(
                            f"Duplicate col in [ann] '{label}' on line {lineno}"
                        )
                    rec["col"] = field[4:]
                else:
                    raise ValueError(
                        f"Unknown token in [ann] '{label}' on line {lineno}: {field}"
                    )
            cmap["ann"][label] = rec
            cmap["ann_order"].append(label)

    def flush_pending(section):
        nonlocal pending_tokens, pending_lineno
        if pending_tokens:
            process_entry(pending_tokens, pending_lineno, section)
        pending_tokens = []
        pending_lineno = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # strip comments
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue

        # Section headers
        if line in ("[par]", "[mod]", "[pal]", "[sig]", "[ann]"):
            if current_section == "pal":
                finish_pending_palette()
            elif current_section in ("mod", "sig", "ann"):
                flush_pending(current_section)
            current_section = line[1:-1]
            continue

        if current_section == "pal":
            # Palette: label then exactly 18 color codes, possibly across lines
            tokens = line.split()
            if not tokens:
                continue
            if pending_pal_label is None:
                pending_pal_label = tokens[0]
                pending_pal_colors.extend(tokens[1:])
            else:
                pending_pal_colors.extend(tokens)
            if len(pending_pal_colors) == 18:
                finish_pending_palette()
            elif len(pending_pal_colors) > 18:
                raise ValueError(
                    f"Too many colors for palette '{pending_pal_label}' "
                    f"(line {lineno})"
                )
            continue

        if current_section == "par":
            line_norm = re.sub(r"\s*=\s*", "=", line)
            if "=" in line_norm:
                left, right = (p.strip() for p in line_norm.split("=", 1))
                cmap["par"][left] = right
            continue

        # [mod] / [sig] / [ann]: buffer entries; '...' continues the previous entry
        is_continuation = line.startswith("...")
        if is_continuation:
            line = line[3:].strip()
            if not pending_tokens:
                raise ValueError(
                    f"Continuation '...' on line {lineno} with no preceding entry"
                )

        line_norm = re.sub(r"\s*=\s*", "=", line)
        new_tokens = line_norm.split()
        if not new_tokens:
            continue

        if is_continuation:
            pending_tokens.extend(new_tokens)
        else:
            flush_pending(current_section)
            pending_tokens = new_tokens
            pending_lineno = lineno

    if current_section == "pal":
        finish_pending_palette()
    elif current_section in ("mod", "sig", "ann"):
        flush_pending(current_section)

    # Validate mod/pal references in [sig]
    for label, rec in cmap["sig"].items():
        m = rec.get("mod")
        if m is None:
            continue
        mod_label = m["mod"]
        pal_label = m["pal"]

        if mod_label not in cmap["mod"]:
            raise ValueError(
                f"[sig] '{label}' refers to unknown mod '{mod_label}'"
            )

        if (pal_label not in cmap["pal"]) and (pal_label not in BUILTIN_PALETTES):
            raise ValueError(
                f"[sig] '{label}' refers to unknown palette '{pal_label}' "
                f"(not in [pal] and not a builtin: {sorted(BUILTIN_PALETTES)})"
            )

    return cmap
