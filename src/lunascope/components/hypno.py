
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

from PySide6.QtWidgets import QVBoxLayout, QHeaderView, QMessageBox
from PySide6.QtCore import Qt

class HypnoMixin:

    def _ensure_hypno_canvas(self, *_args):
        if getattr(self, "hypnocanvas", None) is not None:
            return self.hypnocanvas

        layout = self.ui.host_hypnogram.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.ui.host_hypnogram.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)

        from .mplcanvas import MplCanvas
        self.hypnocanvas = MplCanvas(self.ui.host_hypnogram)
        layout.addWidget(self.hypnocanvas)
        return self.hypnocanvas

    def _init_hypno(self):

        self.hypnocanvas = None
        if self.ui.host_hypnogram.layout() is None:
            self.ui.host_hypnogram.setLayout(QVBoxLayout())
        self.ui.host_hypnogram.layout().setContentsMargins(0,0,0,0)

        # wiring
        self.ui.butt_calc_hypnostats.clicked.connect( self._calc_hypnostats )

    # ------------------------------------------------------------
    # Run hypnostats

    def _calc_hypnostats(self):
        self._ensure_hypno_canvas()

        # clear items first
        self.hypnocanvas.ax.cla()
        self.hypnocanvas.figure.canvas.draw_idle()
        
        # test if we have somebody attached        
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance selected")
            return

        # who has at least some staging available
        diag = self._stage_validation_diagnostics()
        if not diag["ok"]:
            if self.ui.radio_assume_staging.isChecked():
                self._show_staging_message(
                    diag["message"] + "\n\nUncheck Staging to suppress this warning."
                )
            return
        
        # make hypnogram
        ss = self.p.stages()
        from .plts import hypno
        hypno(ss.STAGE, ax=self.hypnocanvas.ax)
        self.hypnocanvas.draw_idle()
        
        # build HYPNO command
        cmd_str = 'EPOCH align & HYPNO'

        cmd_str += ' req-pre-post=' + str( self.ui.spin_req_pre_post.value() )
        cmd_str += ' end-wake=' + str( self.ui.spin_end_wake.value() )
        cmd_str += ' end-sleep=' + str( self.ui.spin_end_sleep.value() )
        
        # annotations?
        if self.ui.check_hypno_annots.isChecked():
            cmd_str += " annot"

        # lights
        if self.ui.check_lights_out.isChecked():
            dt = self.ui.dt_lights_out.dateTime()
            s = dt.toString("dd/MM/yy-HH:mm:ss")
            cmd_str += " lights-off="+s
            
        if self.ui.check_lights_on.isChecked():
            dt = self.ui.dt_lights_on.dateTime()
            s = dt.toString("dd/MM/yy-HH:mm:ss")
            cmd_str += " lights-on="+s

        # save currents channels/annots selections
        # (needed by _render_tables() used below)
        self.curr_chs = self.ui.tbl_desc_signals.checked()                   
        self.curr_anns = self.ui.tbl_desc_annots.checked()

        # Luna call to get full HYPNO outputs
        try:
            res = self.p.silent_proc(cmd_str)
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Error",
                f"Problem running HYPNO:\n{cmd_str}\nCommand failed:\n{e}",
            )
            return

        
        # pull bespoke output for hypno dock
        # (as _render_tables() wipes main output)
        df1 = self.p.table( 'HYPNO' )
        df2 = self.p.table( 'HYPNO' , 'SS' )
        #df3 = self.p.table( 'HYPNO' , 'C' )
        
        # possible that df2 and df3 will be empty - i.e. if only W
        
        # populate tables
        if not df1.empty: 

            df1 = df1.T.reset_index()
            df1.columns = ["Variable", "Value"]        
            df1 = df1[df1.iloc[:, 0] != "ID"]

            v_TST = df1.loc[df1["Variable"] == "TST", "Value"].squeeze()
            self.ui.hyp_TST.setText( f"TST : {v_TST} mins" )
        
            v_WASO = df1.loc[df1["Variable"] == "WASO", "Value"].squeeze()
            self.ui.hyp_WASO.setText( f"WASO : {v_WASO} mins" )

            
        # populate stage table
        if not df2.empty: 
            df = df2.drop(columns=["ID"])

            mins_n1 = df.loc[df["SS"] == "N1", "MINS"].squeeze() if "N1" in df["SS"].values else 0
            mins_n2 = df.loc[df["SS"] == "N2", "MINS"].squeeze() if "N2" in df["SS"].values else 0
            mins_n3 = df.loc[df["SS"] == "N3", "MINS"].squeeze() if "N3" in df["SS"].values else 0
            mins_r  = df.loc[df["SS"] == "R", "MINS"].squeeze() if "R" in df["SS"].values else 0
            
            p_n1 = 100 * df.loc[df["SS"] == "N1", "PCT"].squeeze() if "N1" in df["SS"].values else 0
            p_n2 = 100 * df.loc[df["SS"] == "N2", "PCT"].squeeze() if "N2" in df["SS"].values else 0
            p_n3 = 100 * df.loc[df["SS"] == "N3", "PCT"].squeeze() if "N3" in df["SS"].values else 0
            p_r  = 100 * df.loc[df["SS"] == "R",  "PCT"].squeeze() if "R" in df["SS"].values else 0

            self.ui.hyp_N1.setText( f"N1 : {mins_n1:.1f} mins ({p_n1:.1f}%)" )
            self.ui.hyp_N2.setText( f"N2 : {mins_n2:.1f} mins ({p_n2:.1f}%)" )
            self.ui.hyp_N3.setText( f"N3 : {mins_n3:.1f} mins ({p_n3:.1f}%)" )
            self.ui.hyp_R.setText( f"R : {mins_r:.1f} mins ({p_r:.1f}%)" )

            
        # finally, use standard output mechanism to show full output
        # this will also update annotations if any added by the hypno
        # run

        tbls = self.p.strata()
        self._render_tables( tbls )
