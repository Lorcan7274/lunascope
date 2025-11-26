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


from PySide6.QtWidgets import QMessageBox


class MasksMixin:
    
    def _init_masks(self):
        
        # wiring
        self.ui.butt_generic_mask.clicked.connect( self._apply_mask )

    # ------------------------------------------------------------
    # Update list of potential mask annots
        
    def _update_mask_list(self):

        # clear first
        self.ui.combo_ifnot_mask.clear()
        self.ui.combo_if_mask.clear()

        anns = [ '<none>' ]

        lst = self.p.edf.annots() 
        lst = [s for s in lst if s != "SleepStage"]
        anns.extend( lst )
        
        self.ui.combo_ifnot_mask.addItems( anns )

        self.ui.combo_if_mask.addItems( anns )

        
    # ------------------------------------------------------------
    # Apply MASK

    def _apply_mask(self):

        # requires attached individal
        if not hasattr(self, "p"): return

        # what has been set?
        gen_msk = self.ui.txt_generic_mask.text()
        if_msk = self.ui.combo_if_mask.currentText()
        ifnot_msk = self.ui.combo_ifnot_mask.currentText()

        n = 0
        msk = ''
        if gen_msk != "": n = n + 1 ; msk = gen_msk
        if if_msk != "<none>": n = n + 1; msk = 'if='+if_msk
        if ifnot_msk != "<none>": n = n + 1; msk = 'ifnot='+ifnot_msk

        # nothing to do

        if n == 0:
            QMessageBox.warning( None, "Invalid mask", "No mask values specified")
            return

        # more than one mask set

        if n != 1:
            QMessageBox.warning( None, "Invalid mask", "More than one mask set" )
            return
        
        
        # save selections

        self.curr_chs = self.ui.tbl_desc_signals.checked()
        self.curr_anns = self.ui.tbl_desc_annots.checked()
        
        # run MASK

        self.p.eval( 'MASK ' + msk + ' & RE ' )

        # update the things that need updating

        self._set_render_status( self.rendered , False )
        self._update_metrics()
        self._update_pg1()
        
        self.ui.tbl_desc_signals.set_checked_by_labels( self.curr_chs )
        self.ui.tbl_desc_annots.set_checked_by_labels( self.curr_anns )
        self._update_instances( self.curr_anns )
