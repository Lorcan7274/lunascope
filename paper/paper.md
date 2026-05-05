---
title: 'The Luna ecosystem: integrated tools for sleep signal visualization and analysis'

tags:
  - sleep
  - polysomnography
  - electroencephalography
  - sleep staging
  - biomedical signal processing
  - neurophysiology
  - visualization
  - Python
  - C++
  - open source software


authors:
  - name: Lorcan C. Purcell
    email: lorcan7274@gmail.com
    corresponding: true
    affiliation: 1

  - name: Tejas Karkera
    email: tkarkera@mgh.harvard.edu
    affiliation: 2

  - name: Senthil Palanivelu
    email: senthil.palanivelu@tarento.com
    affiliation: 3

  - name: Shyamal Agarwal
    email: agarwal.shy@northeastern.edu
    affiliation: 3
  
  - name: Christopher Mow
    email: cmow1@mgb.org
    affiliation: 4

  - name: Nataliia Kozhemiako
    orcid: 0000-0002-6450-4959
    email: nkozhemiako@bwh.harvard.edu
    affiliation: 3
    
  - name: Shaun M. Purcell
    orcid: 0000-0002-7402-5812
    corresponding: true
    email: shaun.purcell@nyspi.columbia.edu
    affiliation: "5, 6"
    
affiliations:

  - name: Northfield Mount Hermon School, Mount Hermon, MA, United States
    index: 1

  - name: Division of Sleep and Circadian Disorders, Brigham and Women's Hospital, Harvard Medical School, Boston, MA, United States
    index: 2
    ror: 04b6nzv94

  - name: Department of Psychiatry, Brigham and Women's Hospital, Harvard Medical School, Boston, MA, United States
    index: 3
    ror: 04b6nzv94

  - name: Mass General Brigham Enterprise Research IS, Boston, MA, United States
    index: 4
    ror: 04py2rh25

  - name: Department of Psychiatry, Columbia University Irving Medical Center, New York, NY, United States
    index: 5
    ror: 03m8km719

  - name: New York State Psychiatric Institute, New York, NY, United States 
    index: 6
    ror: 04aqjf708

date: 30 April 2026

bibliography: paper.bib

---

# Summary

Sleep physiology generates rich, high-dimensional signals central to
understanding brain function, aging, and disease, yet the field has
long been constrained by proprietary formats, closed-source
algorithms, and limited open, scalable tools spanning both
large-cohort analysis and detailed inspection of individual recordings. The Luna ecosystem is an open-source sleep
analysis stack built around the Luna C/C++ library, its Python
interface (`lunapi`), and Lunascope, an interactive desktop
application for visualization, annotation, and review. Scripted and
graphical workflows operate on a common data model and command set.
Lunascope provides a synchronized multichannel
viewer with hypnogram display, spectral summaries, annotation
handling, automated sleep staging, cohort-level exploration, multiday
actigraphy views, and direct access to the National Sleep Research
Resource (NSRR). An embedded scripting console exposes Luna's command
language inside the application, allowing users
to move between visual inspection and programmatic analysis within a
single session.

# Statement of need

Computational analysis of polysomnographic (PSG) recordings has become
central to sleep research. Large-scale datasets hosted by the NSRR
have made thousands of whole-night recordings freely available,
enabling population-level studies of sleep architecture, spindle
activity, slow oscillations, and EEG-derived biomarkers
[@purcell2017; @kozhemiako2022slope; @kozhemiako2024]. These analyses
increasingly rely on open-source tools that can process recordings at
scale, but many sleep researchers, clinicians, and trainees lack
fluency with terminal-based or notebook-based workflows, and existing
graphical tools for sleep data often emphasize either signal viewing or
analysis rather than both within a reproducible open-source workflow.

The Luna/`lunapi`/Lunascope stack was developed to bridge this gap.
Luna provides a reusable analytical core, `lunapi` makes that core
available in Python for notebooks and batch workflows, and Lunascope
adds an interactive environment for review and exploratory analysis.
Luna has been used as the primary
analytic platform in studies analyzing over 10,000 PSG recordings
across multiple cohorts in work by our group [@purcell2017;
@djonlagic2021; @kozhemiako2022slope; @kozhemiako2022;
@kozhemiako2024] and others [@adra2022;@hanif2025;@lokhandwala2025].
Lunascope brings those methods into an interactive graphical workflow.

Luna is also part of the broader US National Heart, Lung, and Blood
Institute NSRR [@nsrr; @zhang2018] analytical ecosystem: in
addition to sharing data, the NSRR has supported the use of common
open-source tools for sleep research, and Luna has been used within
NSRR analysis workflows. Lunascope includes a Moonbeam module that
imports NSRR data directly into the environment used for
visualization, review, and downstream analysis.


# Relationship to existing tools

Existing software relevant to PSG analysis reflects different design
goals and usage contexts. EDFbrowser [@edfbrowser] is a mature and
widely used viewer, well suited for manual signal inspection, but
it is primarily a viewing application rather than an integrated
sleep-analysis environment. Broader EEG ecosystems such as EEGLAB
[@Delorme:2004] and MNE-Python [@gramfort2013] are powerful and widely
used, but they are general electrophysiology frameworks rather than
dedicated PSG review environments. Sleep-focused research tools
including YASA [@vallat2021], Wonambi [@wonambi], SleepTrip
[@cox2024sleeptrip], the Snooz Toolbox [@snooztoolbox], and SleepEEGpy
[@sleepeegpy2025] support useful scripted workflows for staging,
detection, preprocessing, or artifact cleaning, but they tend to
emphasize individual pipelines or high-level wrapper layers rather than
a shared analytical engine spanning command-line, Python, and
interactive desktop review. Commercial PSG
review systems support clinical scoring and reporting, but are
typically closed-source, proprietary, and difficult to adapt for
reproducible research pipelines or large-scale open-data workflows.

Taken together, existing tools tend to separate signal viewing,
algorithmic sleep analysis, and cohort-scale data exploration into
different environments. This separation creates friction for
reproducible research: visual review, scripted analysis, annotation
management, and population-level summaries often occur in disconnected
tools with different data assumptions. The Luna ecosystem was
developed to reduce this separation by combining a scalable
sleep-analysis engine, Python access through `lunapi`, an interactive
desktop application for PSG review and annotation, cohort-level
exploration tools, and direct workflows for NSRR-hosted data.

Many existing tools also concentrate on narrower task domains such as
staging, visualization, or selected spindle, spectral, or
preprocessing pipelines, whereas Luna spans a broader analytical
range, including dataset manipulation with full support for gapped 
recordings; interval-level annotation handling; multi-day and
actigraphy-based analyses; linear modeling; multivariate statistical
approaches; neurobiological age prediction; high-density EEG support,
including interpolation, connectivity, and EEG microstate analysis;
and multiple spectral methods, including Welch, multitaper, wavelet,
Hilbert, and IRASA-based analyses.

# Software design

The software stack is organized in three layers. Luna,
implemented in C/C++, provides the analysis engine and command set for
sleep signal processing. It supports European Data Format (EDF/EDF+) inputs [@kemp1992edf;
@kemp2003edfplus; @edfplus_site] together with XML and Luna annotation
formats; the latter uses a simple tab-delimited format supporting
clock-time or elapsed-time encodings, interval durations, and metadata. It also includes EDF manipulation and restructuring
workflows, including support for generating EDF/EDF+ from ASCII TSV
input. The native API exposes recordings and derived data as `Eigen`
[@eigenweb] arrays and matrices, evaluates the same domain-specific
Luna scripting language used at the command line, and includes
embedded command metadata that can be queried programmatically. Some
time/frequency analyses and modeling components, including automated
staging and related prediction workflows, build on external numerical
libraries such as FFTW3 [@frigo2005fftw3] and LightGBM
[@ke2017lightgbm]. Luna also supports association modeling, including
linear-model and permutation-based inference implemented with
efficient `Eigen`-based matrix algebra.  \autoref{fig:overview} gives an overview
of the core Luna tools, the scope of primary functions, and examples
of the visual Lunascope interface.

![Figure 1. Overview of the Luna & Lunascope. A. The family of Luna tools. B. Schematic
of core domains of functionality in Luna. C. Visualization in Lunascope.
\label{fig:overview}](figure1c.png)

`lunapi` exposes that engine to Python, allowing the same recording
objects and commands to be used in scripts, notebooks, and other
applications. It supports single-recording and project-level workflows,
including sample-list construction, command execution across cohorts,
and structured retrieval of Luna result tables as Python objects.

Luna's output system is designed to separate computation from
presentation. The same command outputs can be emitted to standard
output, written to SQLite-backed [@sqlite] result stores queryable as
virtual TSV-like tables, returned to Python as structured objects,
displayed in Qt tables within Lunascope, or written as compressed
text tables. Luna also retains a legacy R interface for direct
integration with R workflows. This output model helps preserve comparable results across
command-line, scripted, and interactive settings.

Lunascope is implemented in Python (3.9--3.14) using PySide6 (Qt 6)
and pyqtgraph as the graphical layer on top of `lunapi`, meaning every
Luna command available at the command line is also available within the
desktop application. A native desktop application was chosen over a
web-based approach because PSG recordings are typically large,
multi-channel, and require responsive pan/zoom interaction across
hours of data. `lunapi` provides a lighter-weight embedded viewer
(`scope`) for JupyterLab; Lunascope is the more full-featured desktop
environment within the same ecosystem.

The Lunascope interface is organized as synchronized docks around a central
signal viewer. Docks can be shown, hidden, detached, repositioned, or
tabbed, and the layout persists across launches. Several modules ---
including Moonbeam, the Explorer, and the Annotator --- operate as
floating windows. Full application state can be saved and restored via
session files.

Within Lunascope, the viewer is tightly coupled to the Luna data model
rather than operating as a passive display layer. Signals and
annotations are attached through `lunapi`, can be inspected at multiple
timescales, and can be queried or modified in the same session that
executes Luna commands. Changes made in the visual annotation editor, for
example, are immediately propagated back to the underlying Luna data
model. Conversely, signals or annotations created or modified through
Luna commands executed in the Lunascope console are immediately reflected
in the viewer.

More generally, interval-level annotations are
treated as first-class objects within Luna, and arbitrary expressions
can be evaluated against those intervals, their metadata, and the
underlying signals. For efficient navigation of long recordings,
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
NSRR studies and imports recordings into the analytic context used
for interactive review. The Explorer extends the single-recording
viewer to cohort-level displays, including hypnogram alignment
(`Hypnoscope`), annotation summaries, waveform displays, and table-based
plots. Lunascope also exposes stack-specific methods such as POPS and
SOAP, multiday actigraphy views for EDF+D-style recordings, and
rendered trace modulation by derived signal properties, allowing Luna
outputs to be inspected as part of the visual workflow rather than only
as exported tables.

The command-line Luna tool supports embarrassingly parallel
high-performance computing workflows, with per-job outputs combined
using `destrat`, which compiles result tables across distributed runs.



# Research impact statement

The Luna/`lunapi`/Lunascope stack supports a shared workflow spanning
batch analysis, scripted exploration, and interactive review. Within
the Purcell laboratory, Lunascope is used for PSG review, quality
assessment, and exploratory analysis in projects that also rely on
Luna's analytical pipeline, including work on sleep EEG spectral
variation [@kozhemiako2022slope], neurodevelopmental sleep architecture
[@kozhemiako2024], and the GRINS schizophrenia neurophysiology
consortium [@Wang:2024; @murphy2024; @kozhemiako2022].


# Availability

The primary entry point for the Luna ecosystem is https://zzz.nyspi.org/luna/.
Lunascope is distributed via PyPI [@lunascope_pypi] and as standalone macOS and Windows
binaries built via GitHub Actions.  The software is available at the
Lunascope source repository [@lunascope_repo], with user documentation
available separately [@lunascope_docs]. Luna documentation is
available online [@luna_docs], including worked examples, vignettes,
and detailed descriptions of command options and outputs. Luna and `lunapi` are
also distributed as Docker containers, and `lunapi` is accompanied by
interactive notebooks and tutorials [@lunapi_notebooks]. We have also
developed an extensive six-part walkthrough [@luna_walkthrough], built
around 20 hd-EEG recordings available through the NSRR Luna/GRINS
dataset [@luna_grins_dataset]. The Luna source repository is also
publicly available [@luna_repo].


# AI usage disclosure

Generative AI tools (Claude, Anthropic and Codex, OpenAI) were used to
assist in editing and formatting the text of this manuscript, and
during software development, primarily to help refactor code, generate
help-related functions and documentation, and assist with use of the
Qt library in Lunascope.  All content and code were reviewed,
verified, and revised by the authors.

# Acknowledgements

Luna development is supported by grants/contracts to S. M. Purcell (as
either PI or MPI) from NIH/NHLBI (R01HL146339, R21HL145492,
75N92019C00011 and OT2HL67310-01), NIH/NIMH (MH108908),
the Wellcome Trust (227108/Z/23/Z) as well as support from
the Stanley Center for Psychiatric Research (Broad Institute).


# References
