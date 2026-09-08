"""Static validation only: no downloads, training, or API requests."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in (root / "src").glob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"))
    print("PASS Python syntax:", path.name)
for path in (root / "notebooks").glob("*.ipynb"):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert not cell["outputs"]
            assert cell["execution_count"] is None
    print("PASS Notebook syntax and cleared outputs:", path.name)
print("Static checks only. Historical model performance was not reproduced.")
