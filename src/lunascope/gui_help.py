"""Centralized GUI help registry and wiring for Lunascope widgets/actions."""

from __future__ import annotations

from typing import Any


# Single source of truth for GUI help text.
WIDGET_HELP: dict[str, str] = {
    # Project dock
    "butt_load_slist": "Load a sample-list file (.lst/.txt). Requires a prepared list of IDs and file paths.",
    "butt_build_slist": "Build a sample list from a folder of EDF files.",
    "butt_load_edf": "Load one EDF directly as an internal single-row sample list.",
    "butt_load_annot": "Load an annotation file for the currently selected EDF/sample.",
    "butt_refresh": "Reload the currently selected sample and re-attach it.",
    "radio_assume_staging": "If checked, warn when staging is missing/invalid for hypnogram workflows.",
    "flt_slist": "Filter sample-list rows by comma-separated terms.",
    "tbl_slist": "Select one row to attach that individual record.",

    # Main signal controls
    "butt_render": "Render selected signals/annotations for the current window and settings.",
    "spin_spacing": "Vertical spacing between traces.",
    "spin_scale": "Global amplitude scale multiplier for rendered traces.",
    "spin_fixed_max": "Upper Y limit used when Fixed Y is enabled.",
    "spin_fixed_min": "Lower Y limit used when Fixed Y is enabled.",
    "radio_fixedscale": "Use fixed Y limits from Fixed min/max controls.",
    "radio_clip": "Clip traces to the current Y range.",
    "radio_empiric": "Use empiric/auto scaling from current rendered data.",
    "check_labels": "Show or hide signal labels in the main trace panel.",

    # Signals/annotations/instances docks
    "butt_sig": "Toggle select-all/select-none for visible signal rows.",
    "txt_signals": "Filter signals by comma-separated terms.",
    "tbl_desc_signals": "Signal table: choose channels and optional filters.",
    "butt_annot": "Toggle select-all/select-none for visible annotation classes.",
    "txt_annots": "Filter annotation classes by comma-separated terms.",
    "tbl_desc_annots": "Annotation class table used for rendering and analysis context.",
    "txt_events": "Filter annotation instances/events by comma-separated terms.",
    "tbl_desc_events": "Annotation instance table for per-event inspection.",

    # Console / analysis
    "txt_inp": "Enter Luna commands/scripts here, then click Execute.",
    "txt_out": "Console output from the last command or project evaluation.",
    "butt_anal_load": "Load command text into the console input editor.",
    "butt_anal_save": "Save current command text from the console input editor.",
    "butt_anal_clear": "Clear console input and output text.",
    "butt_anal_exec": "Run commands in txt_inp for the currently attached sample.",
    "anal_tables": "Returned table list (command/strata). Select a row to view table contents.",
    "radio_transpose": "Transpose the currently displayed output table.",
    "flt_table": "Filter rows/columns of the selected output table.",
    "anal_table": "Data table for the selected result set.",

    # Parameters/config
    "tab_settings": "Parameter and config tabs. Param/Config are editable; Current/Aliases are derived.",
    "txt_param": "Runtime key=value parameters applied on attach/refresh.",
    "txt_cmap": "Display/config rules (colors, filters, dock visibility, POPS defaults).",
    "tbl_param": "Current resolved runtime variables from Luna.",
    "tbl_aliases": "Command/parameter alias table from Luna.",
    "butt_load_param": "Load Param/Config text from file for the active tab.",
    "butt_save_param": "Save Param/Config text from the active tab.",
    "butt_reset_param": "Reset Param/Config text and reinitialize corresponding runtime state.",

    # Spectrogram dock
    "combo_spectrogram": "Signal to analyze (requires SR >= 32 Hz).",
    "butt_spectrogram": "Compute and draw spectrogram for selected signal.",
    "butt_hjorth": "Compute and draw Hjorth summary using selected signal.",
    "spin_lwrfrq": "Lower spectrogram frequency bound in Hz.",
    "spin_uprfrq": "Upper spectrogram frequency bound in Hz.",
    "spin_win": "Winsorization fraction applied to PSD values.",

    # Hypnogram dock
    "butt_calc_hypnostats": "Run HYPNO and update hypnogram/statistics outputs.",
    "check_lights_out": "Include lights-off time in HYPNO command.",
    "check_lights_on": "Include lights-on time in HYPNO command.",
    "dt_lights_out": "Lights-off timestamp used when Lights out is checked.",
    "dt_lights_on": "Lights-on timestamp used when Lights on is checked.",
    "spin_end_wake": "HYPNO end-wake parameter (minutes).",
    "spin_end_sleep": "HYPNO end-sleep parameter (minutes).",
    "spin_req_pre_post": "HYPNO req-pre-post parameter.",
    "check_hypno_annots": "If checked, HYPNO will add/update staging annotations.",

    # SOAP / POPS
    "combo_soap": "Channel used for SOAP (requires valid staging).",
    "spin_soap_pc": "SOAP P_C threshold.",
    "butt_soap": "Run SOAP and plot hypnodensity output.",
    "combo_pops": "Channel(s) used for POPS. Open dropdown and check one or more channels.",
    "txt_pops_path": "Folder containing POPS model files (*.mod).",
    "txt_pops_model": "POPS model basename (for example: s2).",
    "check_pops_ignore_obs": "Ignore observed staging and run POPS in prediction-only mode.",
    "radio_pops_hypnodens": "Toggle POPS hypnodensity rendering mode.",
    "butt_pops": "Run POPS and populate output tables/plots.",

    # Masks
    "txt_generic_mask": "Manual MASK expression (use exactly one mask input mode).",
    "combo_if_mask": "Mask epochs if annotation is present.",
    "combo_ifnot_mask": "Mask epochs if annotation is absent.",
    "butt_generic_mask": "Apply mask and refresh dependent tables/plots.",

    # Existing Luna command helper dock (kept separate from GUI help)
    "flt_ctree": "Filter Luna command tree entries.",
    "tree_helper": "Luna command reference browser (not GUI usage help).",
}


ACTION_HELP: dict[str, str] = {
    "project_load_slist": "Load a sample-list file into the Project dock.",
    "project_build_slist": "Build a sample list from a selected folder.",
    "project_load_edf": "Load one EDF file as an internal sample list.",
    "project_load_annot": "Load annotations for current sample/EDF.",
    "project_refresh": "Refresh and reattach the currently selected sample.",
    "project_eval": "Run the command in project mode across all visible samples.",
    "project_save_session": "Save current window layout and core control values to a session file.",
    "project_load_session": "Load a saved session file and restore layout/control values.",
    "about_help": "Show Lunascope/Luna version info and documentation link.",
    "palette_spectrum": "Apply spectrum color palette.",
    "palette_white": "Apply white/light palette.",
    "palette_muted": "Apply muted palette.",
    "palette_black": "Apply black/dark palette.",
    "palette_random": "Generate and apply a random dark-background palette.",
    "palette_pick": "Pick custom foreground/background colors.",
    "palette_bespoke": "Apply palette and styling from current Config text.",
}


def _apply_text_help(obj: Any, text: str) -> None:
    if not text:
        return
    for meth in ("setToolTip", "setStatusTip", "setWhatsThis"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            fn(text)


def apply_gui_help(ui: Any, actions: dict[str, Any] | None = None) -> None:
    """Apply registry help text to widgets and optional actions."""
    for obj_name, text in WIDGET_HELP.items():
        widget = getattr(ui, obj_name, None)
        if widget is not None:
            _apply_text_help(widget, text)

    if actions:
        for action_name, text in ACTION_HELP.items():
            action = actions.get(action_name)
            if action is not None:
                _apply_text_help(action, text)


def set_render_button_help(ui: Any, rendered: bool, current: bool) -> None:
    """Context-sensitive tooltip for the Render button state."""
    if not hasattr(ui, "butt_render"):
        return

    if not rendered:
        state = "Status: not rendered yet (red style)."
    elif current:
        state = "Status: rendered and current (green style)."
    else:
        state = "Status: rendered but stale after changes (amber style). Click Render to refresh."

    text = (
        "Render selected signals/annotations using current controls. "
        "Requires an attached sample and selected channels. "
        + state
    )
    _apply_text_help(ui.butt_render, text)
