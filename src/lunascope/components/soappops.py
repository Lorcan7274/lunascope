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

from PySide6.QtWidgets import QVBoxLayout, QMessageBox
from PySide6.QtCore import Qt
import os
from pathlib import Path
import pandas as pd

from .mplcanvas import MplCanvas
from .plts import hypno_density, hypno
        
class SoapPopsMixin:


    # valid staging:
    #   - EDF/annotations attached
    #   - found at least some stage-aliased annotations
    #   - no overlapping staging annotations
    #   - no conflicts in epoch-assignment

    def _has_staging(self, require_multiple = True ):
        
        if not hasattr(self, "p"):
            return False

        # CONTAINS stages allows for possible conflicting stages
        try:
            res = self.p.silent_proc('CONTAINS stages')
            df = self.p.table( 'CONTAINS' )
        except Exception:
            return False

            
        if 'df' in locals() and isinstance(df, pd.DataFrame) and not df.empty:

            # no staging info
            if df.at[df.index[0], "STAGES"] != 1:
                return False
            
            # overlapping stage annotations
            if 'OVERLAP' in df.columns and len(df) == 1 and df.at[df.index[0], 'OVERLAP'] == 1:
                return False

            # fewer than 2 unique stages
            if require_multiple:
                if 'UNIQ_STAGES' in df.columns and len(df) == 1 and df.at[df.index[0], 'UNIQ_STAGES'] < 2:
                    return False

            # any conflicts (will generate an 'E' table) 
            df2 = self.p.table( 'CONTAINS' , 'E' )
            if 'df2' in locals() and isinstance(df2, pd.DataFrame) and not df2.empty:
                return False
                
        else:
            # some other problem if not getting the df
            return False

        # if here, we must have good staging
        return True

    
    def _init_soap_pops(self):

        # SOAP hypnodensity plot
        self.ui.host_soap.setLayout(QVBoxLayout())
        self.soapcanvas = MplCanvas(self.ui.host_soap)
        self.ui.host_soap.layout().setContentsMargins(0,0,0,0)
        self.ui.host_soap.layout().addWidget( self.soapcanvas )
        
        # POPS hypnodensity plot
        self.ui.host_pops.setLayout(QVBoxLayout())
        self.popscanvas = MplCanvas(self.ui.host_pops)
        self.ui.host_pops.layout().setContentsMargins(0,0,0,0)
        self.ui.host_pops.layout().addWidget( self.popscanvas )

        # POPS resources
        pops_path = self.ui.txt_pops_path.text()
        
        # wiring
        self.ui.butt_soap.clicked.connect( self._calc_soap )
        self.ui.butt_pops.clicked.connect( self._calc_pops )

        self.ui.radio_pops_hypnodens.toggled.connect( self._render_pops_hypno )
        
    def _update_soap_list(self):

        if not hasattr(self, "p"): return

        # first clear
        self.ui.combo_soap.clear()
        self.ui.combo_pops.clear()

        # list all channels with sample frequencies > 32 Hz 
        df = self.p.headers()

        if df is not None:
            chs = df.loc[df['SR'] >= 32, 'CH'].tolist()
        else:
            chs = [ ]

        self.ui.combo_soap.addItems( chs )
        self.ui.combo_pops.addItems( chs )

        
    # ------------------------------------------------------------
    # Run SOAP

    def _calc_soap(self):

        # requires attached individal
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return
        
        # requires staging
        if not self._has_staging():
            QMessageBox.critical( self.ui , "Error", "No valid stating information:\n overlaps, epoch conflicts, or fewer than 2 valid stages" )
            return

        # requires 1+ channel
        count = self.ui.combo_soap.model().rowCount()
        if count == 0:
            QMessageBox.critical( self.ui , "Error", "No suitable signal for SOAP" )
            return

        # parameters
        soap_ch = self.ui.combo_soap.currentText()
        soap_pc = self.ui.spin_soap_pc.value()

        # run SOAP
        try:
            cmd_str = 'EPOCH align & SOAP sig=' + soap_ch + ' epoch pc=' + str(soap_pc)
            self.p.eval( cmd_str )
        except Exception:
            QMessageBox.critical( self.ui , "Error", "Problem running SOAP" )
            return
            
        # channel details
        df = self.p.table( 'SOAP' , 'CH' )        
        df = df[ [ 'K' , 'K3' , 'ACC', 'ACC3' ] ]

        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
            
        for c in df.select_dtypes(include=['float', 'float64', 'float32']).columns:
            df[c] = df[c].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "")

        # display...
        k, k3 = df.loc[0, ['K', 'K3']].astype(float)
        self.ui.txt_soap_k.setText( f"K = {k:.2f}" )
        self.ui.txt_soap_k3.setText( f"K3 = {k3:.2f}" )
        
        
        # hypnodensities
        df = self.p.table( 'SOAP' , 'CH_E' )
        df = df[ [ 'PRIOR', 'PRED' , 'PP_N1' , 'PP_N2', 'PP_N3', 'PP_R', 'PP_W' , 'DISC' ] ]                                                     
        hypno_density( df , ax=self.soapcanvas.ax)                                                                                               
        self.soapcanvas.draw_idle()                                                                                                              
               
    # ------------------------------------------------------------
    # Run POPS

    def _calc_pops(self):
      
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return
        
        # requires 1+ channel
        count = self.ui.combo_pops.model().rowCount()
        if count == 0:
            QMessageBox.critical( self.ui , "Error", "No suitable signal for POPS" )
            return

        # parameters
        pops_chs = self.ui.combo_pops.currentText()
        if type( pops_chs ) is str: pops_chs = [ pops_chs ] 
        pops_chs = ",".join( pops_chs )

        pops_path = self.ui.txt_pops_path.text()
        pops_model = self.ui.txt_pops_model.text()
        ignore_obs = self.ui.check_pops_ignore_obs.checkState() == Qt.Checked
        
        has_staging = self._has_staging()
        # requires staging
        if not has_staging:
            ignore_obs = True

        # ignore existing staging
        opts = ""
        if ignore_obs:
            opts += " ignore-obs=T"
            has_staging = False
            

        # test if resource file exists
        base = Path(pops_path).expanduser()
        base = Path(os.path.expandvars(str(base))).resolve()   # absolute
        pops_mod = base / f"{str(pops_model).strip()}.mod"
        if not pops_mod.is_file():
            QMessageBox.critical(
                self.ui,
                "Error",
                "Could not open POPS files; double check file path"
            )
            return None


        # save currents channels/annots selections
        # (needed by _render_tables() used below)
        self.curr_chs = self.ui.tbl_desc_signals.checked()                   
        self.curr_anns = self.ui.tbl_desc_annots.checked()

        
        # run POPS
        try:
            cmd_str = 'EPOCH align & RUN-POPS sig=' + pops_chs
            cmd_str += ' path=' + pops_path
            cmd_str += ' model=' + pops_model
            cmd_str += opts
                        
            self.p.eval( cmd_str )
            
        except (RuntimeError) as e:
            QMessageBox.critical(
                self.ui,
                "Error running POPS",
                f"Exception: {type(e).__name__}: {e}"
            )
            return

        
        # hypnodensity plot
        df = self.p.table( 'RUN_POPS' , 'E' )
        if has_staging:
            df = df[ [ 'E', 'START', 'PRIOR', 'PRED' , 'PP_N1' , 'PP_N2', 'PP_N3', 'PP_R', 'PP_W'  ] ]
        else:
            df = df[ [ 'E', 'START', 'PRED' , 'PP_N1' , 'PP_N2', 'PP_N3', 'PP_R', 'PP_W'  ] ]

        self.pops_df = df

        self._render_pops_hypno()

        # populate main output and update annotations (e.g. N1, N2, ... or pN1, pN2, ...)
        tbls = self.p.strata()
        self._render_tables( tbls )

        # if did not have original staging, we will create a new one
        if not has_staging:
            self._render_hypnogram()
            self._update_hypnogram()



    def _render_pops_hypno(self):

        if hasattr(self, 'pops_df') and isinstance(self.pops_df, pd.DataFrame) and not self.pops_df.empty:

            # either draw hypnodensity or hypnogram
            if self.ui.radio_pops_hypnodens.isChecked():
                hypno_density( self.pops_df , ax=self.popscanvas.ax)
            else:
                hypno( self.pops_df.PRED , ax=self.popscanvas.ax)

            self.popscanvas.draw_idle()        
