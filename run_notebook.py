"""Execute experiments.ipynb using the exact interpreter running this script."""
import json
import os
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
os.chdir(root)
os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
notebook = nbformat.read(root / "experiments.ipynb", as_version=4)
with tempfile.TemporaryDirectory(prefix="recsys-kernel-") as temp:
    kernel = Path(temp) / "kernels" / "recsys"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "RecSys homework", "language": "python"}))
    previous = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = temp + (os.pathsep + previous if previous else "")
    client = NotebookClient(notebook, timeout=1800, kernel_name="recsys", resources={"metadata": {"path": str(root)}})
    def started(cell, cell_index, **kwargs):
        print(f"Executing cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    def checkpoint(cell, cell_index, **kwargs):
        nbformat.write(notebook, root / "experiments.ipynb")
    client.on_cell_start = started
    client.on_cell_executed = checkpoint
    try:
        client.execute()
    finally:
        nbformat.write(notebook, root / "experiments.ipynb")
print("Notebook completed; all code cells executed.")
