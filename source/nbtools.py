import json, pathlib
HERE = pathlib.Path(__file__).parent

def _lines(s):
    s = s.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(s)}

def code(s, hidden=None):
    meta = {}
    if hidden or (hidden is None and s.lstrip().startswith("#@title")):
        meta["cellView"] = "form"
    return {"cell_type": "code", "metadata": meta, "execution_count": None,
            "outputs": [], "source": _lines(s)}

TOOLKIT_URL = "https://raw.githubusercontent.com/akvilesinkeviciute/colour-brings-people-together/v1.0/colour_masterclass.py"

def make_module(path):
    base = (HERE / "toolkit.py").read_text()
    base = base.replace('print("Toolkit ready ✓  " + ("(running in Colab)" if IN_COLAB else "(running locally)"))\n', "")
    head = ('"""Colour Brings People Together: toolkit for the BCS HCI 2026 masterclass notebooks.\n\n'
            'Colour maths (CIELAB and CIEDE2000), the Colour Spinner connection, AI helpers and the\n'
            'charts and tables used in Parts B and C. Imported by the notebooks with\n'
            '`from colour_masterclass import *`.\n"""\n')
    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    tail = (f'\nTOOLKIT_VERSION = "{stamp}"\n'
            'print(f"Toolkit ready ✓  version {TOOLKIT_VERSION}  " + ("(running in Colab)" if IN_COLAB else "(running locally)"))\n')
    pathlib.Path(path).write_text(head + base + (HERE / "helpers.py").read_text() + tail)

def loader_cell():
    return code(f"""#@title Run me first: load the masterclass toolkit
#@markdown Takes about 30 seconds the first time. You're ready when it says **Toolkit ready ✓**.
ToolkitURL = "{TOOLKIT_URL}"  #@param {{type:"string"}}
import os, importlib, urllib.request
try:                                   # always fetch the current version
    urllib.request.urlretrieve(ToolkitURL, "colour_masterclass.py")
except Exception as e:
    if not os.path.exists("colour_masterclass.py"):
        raise RuntimeError("Could not download the toolkit. Check your internet connection, or upload "
                           "colour_masterclass.py in Colab's Files panel (folder icon, left).") from e
    print("Couldn't refresh the toolkit, so using the copy already here.")
import colour_masterclass
importlib.reload(colour_masterclass)
from colour_masterclass import *""", hidden=False)

def toolkit_cell():
    src = (HERE / "toolkit.py").read_text()
    return code("#@title Run me first: load the toolkit (no need to read the code)\n" + src, hidden=True)

def write(cells, name, path):
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"name": name, "provenance": [], "toc_visible": True},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    pathlib.Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
