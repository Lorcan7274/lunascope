---
title: 'The Luna ecosystem: integrated tools for sleep signal visualization and analysis'
tags:
  - Python
  - sleep
  - polysomnography
  - EEG
  - signal visualization
  - NSRR
authors:
  - name: Lorcan Purcell
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: Nataliia Kozhemiako
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Shaun M. Purcell
    orcid: 0000-0000-0000-0000
    affiliation: "1, 3"
affiliations:
  - name: Department of Psychiatry, Columbia University Irving Medical Center and New York State Psychiatric Institute, New York, NY, United States
    index: 1
    ror: 03m8km719
  - name: Division of Sleep and Circadian Disorders, Brigham and Women's Hospital, Harvard Medical School, Boston, MA, United States
    index: 2
    ror: 04b6nzv94
  - name: "[Confirm additional affiliation]"
    index: 3
date: "[SET DATE BEFORE SUBMISSION]"
bibliography: paper.bib
---

# Summary

Sleep physiology generates rich, high-dimensional signals that are
central to understanding brain function, aging, and disease, yet the
field has long been constrained by proprietary data formats,
closed-source algorithms, and a relative lack of open, scalable tools
that support both large cohort analyses and detailed inspection of
individual recordings. The Luna ecosystem is an open-source sleep
analysis stack built around the Luna C/C++ library [@Purcell:2024], its
Python interface (`lunapi`), and Lunascope, an interactive desktop
application for visualization, annotation, and review. Scripted and
graphical workflows therefore operate on the same underlying data model
and command set. Lunascope provides a synchronized multi-channel viewer
with hypnogram display, spectral summaries, annotation handling,
automated sleep staging, cohort-level exploration, multiday actigraphy
views, and direct access to the National Sleep Research Resource
[NSRR, the NHLBI National Sleep Research Resource, https://sleepdata.org;
@Zhang:2018]. An embedded scripting console exposes Luna's
command language inside the application, allowing users to move between
visual inspection and programmatic analysis within a single session.

# Statement of need

Computational analysis of polysomnographic (PSG) recordings has become
central to sleep research. Large-scale datasets hosted by the NSRR
have made thousands of whole-night recordings freely available,
enabling population-level studies of sleep architecture, spindle
activity, slow oscillations, and EEG-derived biomarkers
[@Purcell:2017; @Kozhemiako:2022a; @Kozhemiako:2023]. These analyses
increasingly rely on open-source tools that can process recordings at
scale, but many sleep researchers, clinicians, and trainees lack
fluency with terminal-based or notebook-based workflows, and existing
graphical tools for sleep data often emphasize either signal viewing or
analysis rather than both within a reproducible open-source workflow.

Luna is also part of the broader NSRR analytical ecosystem: in
addition to sharing data, the NSRR has supported the use of common
open-source tools for sleep research, and Luna has been used within
NSRR analysis workflows. Lunascope builds on that connection by
providing desktop access to Luna-based methods, while its Moonbeam
module imports NSRR data directly into the same environment used for
visualization, review, and downstream analysis.

The Luna/`lunapi`/Lunascope stack was developed to bridge this gap.
Luna provides a reusable analytical core, `lunapi` makes that core
available in Python for notebooks and batch workflows, and Lunascope
adds an interactive environment for review and exploratory analysis.
This architecture supports both large-scale computation and detailed
inspection of individual recordings using the same library, data model,
and command language. Luna has been used as the primary analytic

platform in studies analyzing over 10,000 PSG recordings across
multiple cohorts [@Purcell:2017; @Kozhemiako:2022a;
@Kozhemiako:2022b; @Kozhemiako:2023], and Lunascope brings those
methods into an interactive graphical workflow.

# Relationship to existing tools

Existing software relevant to PSG analysis reflects different design
goals and usage contexts. EDFbrowser [@vanBeelen:2010] is a mature and
widely used EDF viewer, well suited for manual signal inspection, but
it is primarily a viewing application rather than an integrated
sleep-analysis environment. Wonambi provides sleep-oriented
functionality, including sleep scoring and event detection, but it is
not designed around a shared analytical engine spanning command-line,
Python, and GUI workflows, nor around cohort-scale exploration or
direct repository integration. The Snooz Toolbox
(https://github.com/SnoozToolbox/snooz-toolbox) similarly emphasizes
open PSG analysis workflows and cohort-oriented processing, but is not
centered on a single analytical engine shared across command-line,
Python, and interactive review contexts. EEGLAB [@Delorme:2004] is a
general EEG
analysis environment, but it is MATLAB-based, oriented primarily toward
EEG/ERP workflows, and does not provide a dedicated PSG review
environment or native sleep-analysis pipeline. MNE-Python is a powerful
electrophysiology toolkit, but it is a programming library rather than
a standalone application, so users must assemble their own PSG
visualization, annotation, and cohort workflows. YASA [@Vallat:2021]
provides accessible Python implementations of several sleep-analysis
algorithms, including staging and spindle detection, but it is designed
for scripted and notebook-based analysis rather than persistent
interactive review or integrated cohort exploration. Commercial PSG
review systems support clinical scoring and reporting, but are
typically closed-source, proprietary, and difficult to adapt for
reproducible research pipelines or large-scale open-data workflows.

Taken together, existing tools tend to separate signal viewing,
algorithmic sleep analysis, and cohort-scale data exploration into
different environments. This separation creates friction for
reproducible research: visual review, scripted analysis, annotation
management, and population-level summaries often occur in disconnected
tools with different data assumptions. To our knowledge, relatively few
open tools aim to combine interactive PSG visualization, access to a
mature sleep-analysis engine, cohort-level workflows, annotation
support, and scripted analysis within one reproducible environment.

The Luna ecosystem was developed to reduce this separation. The Luna
C/C++ library provides the shared analytical engine, `lunapi` provides
Python access to the same recordings, commands, and result structures,
and Lunascope provides the interactive application layer for review,
annotation, staging, and visualization. The main contribution of this
work is the integration of a scalable sleep-analysis engine, Python
access through `lunapi`, an interactive desktop application for PSG
review and annotation, cohort-level exploration tools, and direct
workflows for NSRR-hosted data.

# Software design

The software stack is organized in three layers. Luna, implemented in
C/C++, provides the analysis engine and command set for sleep signal
processing. It supports EDF/EDF+ inputs together with XML and Luna
annotation formats, and spans EDF manipulation and restructuring,
including workflows that accept ASCII TSV input to generate EDF/EDF+,
signal processing, spectral analysis, masking and epoching, event
detection, automated staging, multichannel decomposition,
physiological summary measures, actigraphy, and prediction workflows
including neurobiological age models. The native API exposes
recordings and derived data as `Eigen` arrays and matrices, evaluates
the same domain-specific Luna scripting language used at the command
line, and includes embedded command metadata that can be queried
programmatically. Some modeling components, including automated staging
and related prediction workflows, build on external numerical libraries
such as LightGBM. Luna also supports association modeling, including
linear-model and permutation-based inference using approaches such as
Freedman--Lane, implemented with efficient `Eigen`-based matrix
algebra.

`lunapi` exposes that engine to Python, allowing the same recording
objects and commands to be used in scripts, notebooks, and other
applications. It supports single-recording and project-level workflows,
including sample-list construction, command execution across cohorts,
and structured retrieval of Luna result tables into Python objects.

Luna's output system is designed to separate computation from
presentation. The same command outputs can be emitted to standard
output, written to SQLite-backed result stores that can be queried as
virtual TSV-like tables, returned to Python as structured objects,
consumed in R-facing workflows, displayed in Qt tables within
Lunascope, or written as compressed text tables for larger results.
This common output model helps preserve comparable results across
command-line, scripted, and interactive settings.

Lunascope is implemented in Python (3.9--3.14) using PySide6 (Qt 6)
and pyqtgraph as the graphical layer on top of `lunapi`, meaning every
Luna command available at the command line is also available within the
desktop application. A native desktop application was chosen over a
web-based or Jupyter-embedded approach because PSG recordings are
typically large (hundreds of megabytes), multi-channel, and require
responsive pan/zoom interaction across hours of data. Luna already
provides a lighter-weight embedded viewer (`scope`) for JupyterLab;
Lunascope is the more full-featured desktop environment within the same
ecosystem.

The interface is organized as synchronized docks around a central
signal viewer. Docks can be shown, hidden, detached, repositioned, or
tabbed, and the layout persists across launches. Several modules ---
including Moonbeam, the Explorer, and the Annotator --- operate as
floating windows. Full application state can be saved and restored via
`.lss` session files, preserving window geometry, dock placement, text
buffers, control states, and the selected sample-list row.

Within Lunascope, the viewer is tightly coupled to the Luna data model
rather than operating as a passive display layer. Signals and
annotations are attached through `lunapi`, can be inspected at multiple
timescales, and can be queried or modified in the same session that
executes Luna commands. For efficient navigation of long recordings,
Lunascope maintains rendered, decimated signal caches and summary
envelopes for broad time windows while preserving access to the
underlying sample-level data. This supports interactive measurement,
annotation-aware review, and direct visual inspection of outputs from
Luna analyses.

Several interface elements are particularly important within the wider
ecosystem. The embedded console executes native Luna command strings and
returns structured result tables, so graphical and scripted analyses
share one command language. The command browser and output tables use
Luna's embedded documentation metadata to expose parameter and variable
descriptions inside the application. Moonbeam provides direct access to
NSRR studies and imports recordings into the same analytic context used
for interactive review. The Explorer extends the single-recording
viewer to cohort-level displays, including hypnogram alignment
(`Hypnoscope`), annotation summaries, waveform displays, and table-based
plots. Lunascope also exposes stack-specific methods such as POPS and
SOAP, multiday actigraphy views for EDF+D-style recordings, and
rendered trace modulation by derived signal properties, allowing Luna
outputs to be inspected as part of the visual workflow rather than only
as exported tables.

The broader Luna ecosystem is also designed for embarrassingly
parallelizable workflows, including high-performance computing
environments in which many recordings are processed independently. In
that setting, per-job outputs can be written separately and later
combined with tools such as `destrat`, which compiles result tables
across many files produced by distributed runs.

Lunascope is distributed via PyPI and as standalone macOS and Windows
binaries built via GitHub Actions. Within the broader stack, `lunapi`
also supports direct Python use independent of the GUI, while Luna
remains available for command-line workflows. Data loading in Lunascope
supports single EDFs (with automatic annotation file discovery),
annotation-only files, Luna sample lists, recording folders with
automatic sample-list construction, and saved sessions. A configuration
file system controls signal and annotation ordering, coloring, Y-axis
limits, dock visibility defaults, and signal modulation rules,
separate from Luna parameter files.

![Overview of the Lunascope interface, showing the spectrogram (top), 
multi-channel signal viewer (center), POPS automated staging with 
hypnodensity (bottom left), structured output tables (bottom center), 
and signal/annotation management panels.\label{fig:overview}](pb1.png)


# Research impact statement

The Luna/`lunapi`/Lunascope stack supports a shared workflow spanning
batch analysis, scripted exploration, and interactive review. Within
the Purcell laboratory, Lunascope is used for PSG review, quality
assessment, and exploratory analysis in projects that also rely on
Luna's analytical pipeline, including work on sleep EEG spectral
variation [@Kozhemiako:2022a], neurodevelopmental sleep architecture
[@Kozhemiako:2023], and the GRINS schizophrenia neurophysiology
consortium [@Wang:2024; @Kozhemiako:2022b]. This integration is
particularly relevant for training and reproducibility because the same
operations can be accessed through the command line, Python, and the
desktop application. The software is available at
https://github.com/Lorcan7274/lunascope with documentation at
https://zzz-luna.org/lunascope/. Luna documentation is available at
https://zzz.nyspi.org/luna/ and the Luna source repository at
https://github.com/remnrem/luna.

# AI usage disclosure

Generative AI tools (Claude, Anthropic) were used to assist in drafting
and editing the text of this manuscript, and during software
development to help refactor code, generate help-related functions and
documentation, and assist with use of the Qt library. All content and
code were reviewed, verified, and revised by the authors.

# Acknowledgements

Luna development is supported by NIH grants NHLBI R01HL146339, NHLBI
R21HL145492, NIMH R03 MH108908 (PI: S.M. Purcell), NHLBI R35HL135818,
and NHLBI R24HL114473 (PI: S. Redline).

# References








