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
