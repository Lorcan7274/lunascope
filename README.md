# lunascope

A viewer and frontend for Luna (http://zzz.nyspi.org/luna/)

Requirements:

 - Python 3.9 - 3.14

 - Compatible with CPython: PyPy not supported.

 - Depends on [lunapi](https://pypi.org/project/lunapi/), the Python
   version of the core [Luna](https://zzz.nyspi.org/luna/) library. This binary
   package is tested and supported on macOS (Intel and Apple Silicon), Linux (x86_64), and
   Windows 10/11 (x86_64) with Python 3.9 – 3.14.  Other platforms or
   Python versions may build successfully [from source](https://github.com/remnrem/luna-api)
   but are not officially supported.


## Standalone Binaries (no Python required)

Pre-built standalone apps for macOS and Windows are available from the
[Latest Build release](https://github.com/Lorcan7274/lunascope/releases/tag/latest-build)
— no Python installation needed.

### macOS

1. Download `Lunascope.app.zip` and unzip it.
2. Move `Lunascope.app` to your **Applications** folder (or anywhere you like).
3. **First launch only:** macOS will block the app because it is not from the App Store.
   Right-click (or Control-click) `Lunascope.app` and choose **Open**, then click **Open**
   in the dialog that appears. You only need to do this once; subsequent double-clicks will
   work normally.

   If you see *"Lunascope.app" will damage your computer* and no Open option appears,
   run this once in Terminal to remove the quarantine flag:
   ```
   xattr -dr com.apple.quarantine /path/to/Lunascope.app
   ```

### Windows

1. Download `Lunascope-Windows.zip` and unzip it (right-click → **Extract All**).
2. Open the extracted `Lunascope.dist` folder and double-click **Lunascope.exe**.
3. **First launch only:** Windows SmartScreen may show a blue *"Windows protected your PC"*
   dialog because the app is not code-signed. Click **More info**, then **Run anyway**.

> **Note:** These binaries are unsigned. Your browser or antivirus may warn you on
> download — this is expected. The source code is fully open and auditable above.

---

## Installation

`lunascope` is distributed via PyPI:

```bash
pip install lunascope
```

After installation, you can launch it from the command line:

```bash
lunascope
```

or equivalently:

```bash
python -m lunascope
```

---

## Recommended: Use a Virtual Environment

It is strongly recommended to install `lunascope` into a virtual environment rather than into your system Python.

### Option 1 — Using `venv` (Standard Python)

Create and activate a virtual environment:

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

Then install:

```bash
pip install lunascope
```

Run:

```bash
lunascope
```

To leave the environment:

```bash
deactivate
```

---

### Option 2 — Using `pipx` (Recommended for CLI Apps)

If you prefer to install `lunascope` as an isolated application:

```bash
pipx install lunascope
```

Then run:

```bash
lunascope
```

`pipx` keeps the application isolated from your main Python
installation and avoids dependency conflicts. This is often the
simplest option if you are not otherwise managing environments.

---

## Python vs `python3`

On some systems:

- `python` may refer to Python 2 (older systems)
- `python3` refers to Python 3

Check your version:

```bash
python --version
```

or:

```bash
python3 --version
```

Use whichever command reports Python **3.9–3.14**.

Similarly for pip:

```bash
pip3 install lunascope
```

or more explicitly:

```bash
python3 -m pip install lunascope
```

Using `python -m pip` ensures you install into the correct interpreter.

---

## Platform Notes

- Supported platforms: macOS (Intel & Apple Silicon), Linux (x86_64), Windows 10/11 (x86_64).
- On macOS with Homebrew Python, you may see an “externally managed environment” error. In that case, use a virtual environment or `pipx`.
- On Windows, install Python from python.org and ensure **“Add Python to PATH”** is checked during installation.

---

## First Launch

The first time you run `lunascope`, startup may take longer while dependencies and components initialize. Subsequent launches are typically much faster.

---

## Updating

If installed with pip:

```bash
pip install --upgrade lunascope
```

If installed with pipx:

```bash
pipx upgrade lunascope
```

---

## Uninstalling

With pip:

```bash
pip uninstall lunascope
```

With pipx:

```bash
pipx uninstall lunascope
```

---

## Questions

Please write to `luna.remnrem@gmail.com`.
