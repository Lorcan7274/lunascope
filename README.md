# Lunascope

**An interactive desktop application for sleep signal visualization, annotation, and analysis — built on the [Luna](http://zzz.nyspi.org/luna/) ecosystem.**

![Lunascope interface](paper/figure1c.png)

---

## Statement of Need

Computational analysis of polysomnographic (PSG) recordings has become central to sleep research, yet a practical gap exists between large-scale scripted pipelines and the detailed visual review that quality assessment and exploratory analysis require. Existing tools tend to separate signal viewing, algorithmic analysis, and cohort-scale exploration into different environments, creating friction for reproducible research.

Lunascope bridges this gap. It is a native desktop application that puts Luna's full analytical engine — the same one used in command-line and Python workflows — behind an interactive graphical interface. Signals, annotations, and derived outputs can be inspected, modified, and re-analyzed within a single session, with changes propagated immediately between the visual layer and the underlying data model.

Lunascope is aimed at sleep researchers, clinical scientists, and trainees who need to review PSG recordings, manage annotations, run automated staging, or explore cohort-level summaries without leaving the graphical environment — while remaining able to drop into scripted or batch analysis at any point.

---

## Features

- **Synchronized multichannel viewer** — pan and zoom across hours of EEG, EMG, EOG, and other channels with responsive decimated rendering
- **Hypnogram display** — color-coded sleep staging synchronized with the signal viewer
- **Spectral summaries** — per-channel power spectra and spectrograms updated on the fly
- **Annotation editor** — create, edit, and delete interval annotations; changes are immediately reflected in Luna's data model
- **Automated sleep staging** — POPS and SOAP models accessible directly from the interface
- **Cohort-level Explorer** — Hypnoscope alignment, annotation summaries, waveform displays, and table-based plots across a sample list
- **Moonbeam NSRR module** — import recordings directly from the National Sleep Research Resource into the analysis context
- **Embedded scripting console** — execute any Luna command and receive structured result tables without leaving the application
- **Command browser** — searchable Luna command and parameter reference with embedded documentation
- **Multiday / actigraphy views** — support for EDF+D gapped recordings and long-form waveform review
- **Session save/restore** — full application state (layout, loaded files, annotations) persists across launches

---

## Worked Example

The following shows a typical exploratory session using the Python interface (`lunapi`) alongside Lunascope, using a single EDF recording.

### 1. Install and launch

```bash
pip install lunascope
lunascope
```

### 2. Load a recording via the console

Open **Console** from the View menu and enter:

```
LOAD edf /path/to/recording.edf
LOAD annot /path/to/recording.annot
```

Or use **File → Open EDF** to load through the GUI.

### 3. Run automated staging

In the console:

```
POPS
```

The predicted hypnogram appears immediately in the Hypnogram dock.

### 4. Inspect a spectral summary

Select a channel in the signal viewer, then open **Spectral** from the View menu. The power spectrum for the current epoch is displayed and updates as you navigate.

### 5. Export results

```
WRITE-ANNOTS file=out.annot
```

Or use **File → Save Annotations** to export the current annotation set.

### Using `lunapi` in Python / notebooks

```python
import lunapi as lp

p = lp.inst()
p.attach_edf("/path/to/recording.edf")
p.attach_annot("/path/to/recording.annot")

# Run a Luna command and retrieve results as a DataFrame
p.eval("PSD sig=EEG dB=T")
df = p.table("PSD", "CH_F")
print(df.head())
```

Full documentation, vignettes, and interactive notebooks are available at [zzz.nyspi.org/luna/](https://zzz.nyspi.org/luna/).

---

## Installation

### Standalone binaries (no Python required)

Pre-built apps for macOS and Windows are available from the
[Latest Build release](https://github.com/Lorcan7274/lunascope/releases/tag/latest-build).

**macOS**
1. Download `Lunascope.app.zip` and unzip it.
2. Move `Lunascope.app` to **Applications** (or anywhere).
3. First launch: right-click → **Open** to bypass Gatekeeper, then click **Open**. If you see *"will damage your computer"* with no Open option, run once in Terminal:
   ```
   xattr -dr com.apple.quarantine /path/to/Lunascope.app
   ```

**Windows**
1. Download `Lunascope-Windows.zip` and unzip it.
2. Open the `Lunascope.dist` folder and double-click **Lunascope.exe**.
3. First launch: click **More info → Run anyway** if SmartScreen appears.

> These binaries are unsigned. Browser or antivirus warnings are expected. The source is fully open and auditable here.

### From PyPI

```bash
pip install lunascope
lunascope
```

**Using a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install lunascope
lunascope
```

**Using pipx**

```bash
pipx install lunascope
lunascope
```

### Requirements

- Python 3.9–3.14 (CPython; PyPy not supported)
- Supported platforms: macOS (Intel & Apple Silicon), Linux (x86_64), Windows 10/11 (x86_64)
- On macOS with Homebrew Python, use a virtual environment or `pipx` to avoid the "externally managed environment" error.
- On Windows, install Python from python.org and check **"Add Python to PATH"**.

### Updating / uninstalling

```bash
pip install --upgrade lunascope   # upgrade
pip uninstall lunascope           # remove
# or with pipx:
pipx upgrade lunascope
pipx uninstall lunascope
```

---

## Documentation

| Resource | URL |
|---|---|
| Luna ecosystem | https://zzz.nyspi.org/luna/ |
| Lunascope docs | https://zzz.nyspi.org/luna/lunascope/ |
| lunapi notebooks | https://github.com/remnrem/luna-api/tree/main/notebooks |
| NSRR | https://sleepdata.org |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on bug reports, feature requests, and pull requests.

## Questions / Support

Open an issue on GitHub or write to `luna.remnrem@gmail.com`.
