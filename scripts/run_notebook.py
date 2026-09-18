"""Execute the notebook from a fresh kernel, in order, and export HTML.

Usage: python scripts/run_notebook.py            (run from the project root)
Writes: kindergarten_menu_ml.ipynb (executed, in place) and kindergarten_menu_ml.html
"""
import os, sys, time
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
NB = "kindergarten_menu_ml.ipynb"
t0 = time.time()
nb = nbformat.read(NB, as_version=4)
if "--export-only" not in sys.argv:          # re-export the HTML from an already executed notebook
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": ROOT}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, NB)          # keep partial outputs for debugging if a cell fails
    print(f"executed in {time.time() - t0:.0f} s")
html, _ = HTMLExporter().from_notebook_node(nb)
# small readability tweak: wrap long code lines instead of clipping them (nbconvert default scrolls)
html = html.replace("</head>", "<style>.jp-CodeMirrorEditor pre, .highlight pre { white-space: pre-wrap !important; }</style></head>", 1)
open("kindergarten_menu_ml.html", "w", encoding="utf-8").write(html)
print("HTML bytes:", os.path.getsize("kindergarten_menu_ml.html"))
