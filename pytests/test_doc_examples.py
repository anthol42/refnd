"""Executes every ``Example::`` block found in refnd's .pyi docstrings.

Catches broken imports, renamed/removed API, and stale parameters in the
docs before they reach Sphinx (see py/docs/source/conf.py) or a user.
"""

import ast
import textwrap
from pathlib import Path

import pytest

_STUB_DIR = Path(__file__).parent.parent / "py" / "python" / "refnd"


def _extract_examples(docstring: str) -> list[str]:
    lines = docstring.splitlines()
    examples = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "Example::" or stripped.endswith("Example::"):
            marker_indent = len(lines[i]) - len(lines[i].lstrip())
            block = []
            i += 1
            while i < len(lines):
                if lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) <= marker_indent:
                    break
                block.append(lines[i])
                i += 1
            code = textwrap.dedent("\n".join(block)).strip()
            if code:
                examples.append(code)
        else:
            i += 1
    return examples


def _collect_docstrings(node: ast.AST) -> list[tuple[str, str]]:
    """Return (qualified_name, docstring) for module/classes/functions."""
    out = []
    doc = ast.get_docstring(node)
    name = getattr(node, "name", "<module>")
    if doc:
        out.append((name, doc))
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for cname, cdoc in _collect_docstrings(child):
                out.append((f"{name}.{cname}", cdoc))
    return out


def _discover_examples():
    cases = []
    for pyi in sorted(_STUB_DIR.rglob("*.pyi")):
        tree = ast.parse(pyi.read_text())
        for qualname, doc in _collect_docstrings(tree):
            for j, code in enumerate(_extract_examples(doc)):
                rel = pyi.relative_to(_STUB_DIR).as_posix()
                cases.append(pytest.param(code, id=f"{rel}::{qualname}#{j}"))
    return cases


# These examples are correct but read a placeholder file (PDB/FASTA/edge list)
# that doesn't exist in the test sandbox, or continue a variable from an
# earlier paragraph in the same docstring that our block-based extraction
# doesn't carry over. Not import/API bugs.
_NEEDS_FIXTURE = {
    "core/__init__.pyi::<module>.CsrGraph.subgraph#0",
    "core/__init__.pyi::<module>.EdgeStore.save#0",
    "kernels/structures/__init__.pyi::<module>.USAlignKernel#0",
    "utils/__init__.pyi::<module>.PdbStructure#0",
    "utils/__init__.pyi::<module>.read_fasta#0",
}


@pytest.mark.parametrize("code", _discover_examples())
def test_doc_example_runs(request, code):
    if request.node.callspec.id in _NEEDS_FIXTURE:
        pytest.xfail("references a placeholder file / prior-paragraph variable, not a doc bug")
    exec(compile(code, "<doc-example>", "exec"), {"__name__": "__main__"})
