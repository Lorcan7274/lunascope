
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

class CMapsMixin:

    def _init_cmaps(self):

        self.cfg = { } 

        # extracted : ch -> mod_label
        self.sigmods = { } 

        self.sigmod_colors = {}

        self.cmap_fixed_min = {}
        self.cmap_fixed_max = {}

        self.cmap_ylines = { } # ch -> [ y-vals ] 
        self.cmap_ylines_idx = { }

        self.cfg_pops_path = '~/dropbox/pops/'
        self.cfg_pops_model = 's2'

        self.ui.txt_pops_path.setText( self.cfg_pops_path )
        self.ui.txt_pops_model.setText( self.cfg_pops_model )

        
    #
    # clear the current cmap
    #
    
    def _clear_cmaps(self):
        pass
    
    #
    # apply the current cmap (on attach/refresh) only called on 'render'
    #  --> colors, y-scales, ordering, etc
    #  (sigmod is only done at render stage)
    
    def _apply_cmaps(self):

        # reset priors
        self._clear_cmaps()
        
        # take current text in        
        text = self.ui.txt_cmap.toPlainText()
        try:
            self.cfg = parse_cmap(text)
        except ValueError as e:
            QMessageBox.critical(self.ui, "CMap parse error", str(e))
            return

#        print( self.cfg ) 
        # get sigmod pals
        # for now, just the default (rwb)

        self.sigmod_colors[ 'rwb' ] = self.rwb_sigmod_colors
        
        #
        # set various cfg vars
        #

        self.cfg_line_weight = 1;

        self.cfg_show_zero_line = True;

        if 'line-weight' in self.cfg['par']:
            self.cfg_line_weight = float( self.cfg['par']['line-weight'] )
            
        if 'show-lines' in self.cfg['par']:
            t = self.cfg['par']['show-lines']            
            if t == "1" or t == "Y" or t == "T":
                self.cfg_show_zero_line = True
            else:
                self.cfg_show_zero_line = False


        # POPS

        if 'pops-path' in self.cfg['par']:
            self.cfg_pops_path = self.cfg['par']['pops-path']
            self.ui.txt_pops_path.setText( self.cfg_pops_path )
	    
        if 'pops-model' in self.cfg['par']:
            self.cfg_pops_model = self.cfg['par']['pops-model']
            self.ui.txt_pops_model.setText( self.cfg_pops_model )

                
        def ucfirst(s: str) -> str:
            return s[:1].upper() + s[1:]

        def istrue(s: str) -> bool:
            return ucfirst(s) in [ 'Y', 'T', '1' ]
        
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
            if 'col' in self.cfg['sig'][ch]:
                self.cmap[ch] = self.cfg['sig'][ch]['col']
                

        # reverse order (for plotting goes y 0 - 1 is bottom - top currently
        # and can't be bothered to fix
        self.cmap_rlist = list(reversed(self.cmap_list))
                
        # and flag that we'll have bespoke as the default
        if len(self.cmap) != 0:            
            self.palset = 'bespoke'

                
    
                    
    #
    # render-stage cmaps
    #

    def _render_cmaps(self):

        # segsrv.make_sigmod( label , ch , type , SR, filter-order, lwr, upr
                
        # generate all mods
        for mod_label, mod_spec in self.cfg['mod'].items():

            okay = True;
            
            ch1 = mod_spec[ 'ch' ]
            if ch1 not in self.srs:
                okay = False
            
            type1 = mod_spec[ 'type' ]
            if type1 not in [ 'raw' , 'amp' , 'phase' ]:
                okay = False

            frqs = [ ]
            if 'f' in mod_spec:
                frqs = mod_spec['f']
                if okay is False:
                    QMessageBox.critical( self.ui, "Sigmod Error", f"Could not attach sigmod load {mod_label}" )                                          
                else:
                    self.ss.make_sigmod( mod_label , ch1, type1 , self.srs[ch1], 4, frqs[0] , frqs[1] )

            else:
                if okay is False:
                   QMessageBox.critical( self.ui, "Sigmod Error", f"Could not attach sigmod load {mod_label}" )
                else:
                    self.ss.make_sigmod_raw( mod_label , ch1, type1 )
                    
            # map ch -> mod 
            self.sigmods = { } 
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
            'W':  '#008000',  # green (CSS "green")
            '?':  '#808080',  # gray
            'L':  '#FFFF00',  # yellow
        }

        # signal modulation colors: red-white-blue
        self.rwb_sigmod_colors = [
            (255,255,255),
            (204,217,255),
            (140,179,255),
            (77,128,255),
            (26,64,255),
            (102,140,255),
            (179,204,255),
            (255,255,255),
            (255,179,179),
            (255,115,115),
            (255,64,64),
            (255,26,26),
            (255,102,102),
            (255,179,179),
            (255,217,217),
            (255,242,242),
            (255,255,255),
            (255,255,255),
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
        return [self.stgcols_hex.get(a_i, p_i) for a_i, p_i in zip(anns, pal)]

    
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
        for c, col in zip(self.curves, self.colors):
            c.setPen(pg.mkPen(col, width=1, cosmetic=True))
        for c, col in zip(self.annot_curves, self.acolors):
            c.setPen(pg.mkPen(col, width=1, cosmetic=True))



            
        

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

    BUILTIN_PALETTES = {"rwb", "rainbow", "saturation"}
            
    current_section = None

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

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # strip comments
        line = raw_line.split("%", 1)[0].strip()
        if not line:
            continue

        # Section headers
        if line in ("[par]","[mod]", "[pal]", "[sig]", "[ann]"):
            if current_section == "pal":
                finish_pending_palette()
            current_section = line[1:-1]  # 'par', 'mod', 'pal', 'sig', 'ann'
            continue

        if current_section is None:
            # ignore anything before first section header
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

        # For other sections, normal “one row per object”
        # Normalize "key = value" → "key=value"
        line_norm = re.sub(r"\s*=\s*", "=", line)
        tokens = line_norm.split()
        if not tokens:
            continue

        label = tokens[0]
        fields = tokens[1:]

        if current_section == "mod":
            # [mod] label ch=X type=Y [f=lwr,upr]
            rec = {"ch": None, "type": None, "f": None}
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
                else:
                    raise ValueError(
                        f"Unknown token in [mod] '{label}' on line {lineno}: {field}"
                    )
            if rec["ch"] is None or rec["type"] is None:
                raise ValueError(
                    f"[mod] '{label}' missing ch= or type= (line {lineno})"
                )
            cmap["mod"][label] = rec

        elif current_section == "sig":
            # [sig]
            #   label col=color y=-1,1 ylim=lwr,upr mod=mod_label,pal_label [f=lwr,upr]
            rec = {
                "col": None,
                "ylim": None,
                "y": None,
                "f": None,
                "mod": None,  # {"mod": mod_label, "pal": pal_label}
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
                    s = field[2:] 
                    rec["y"] = [float(x) for x in s.split(",") if x]
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

            cmap["sig"][label] = rec
            cmap["sig_order"].append(label)

        elif current_section == "ann":
            # [ann]
            #   label col=color
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

        elif current_section == "par":
            line_norm = re.sub(r"\s*=\s*", "=", line)
            if "=" in line_norm:
                left, right = (p.strip() for p in line_norm.split("=", 1))
                cmap["par"][left] = right


    if current_section == "pal":
        finish_pending_palette()

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
