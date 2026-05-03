import ast
import re
import sys
from pathlib import Path

# ── Project info ───────────────────────────────────────────────────────────────

project = "refnd"
author = "Anthony Lavertu, Jacob Côté"

# ── Paths ──────────────────────────────────────────────────────────────────────

_DOCS_DIR = Path(__file__).parent
_STUB_DIR = _DOCS_DIR.parents[1] / "python"  # py/python/

sys.path.insert(0, str(_STUB_DIR))
import refnd  # noqa: E402, F401 — must be importable before autodoc runs

# ── Extensions ────────────────────────────────────────────────────────────────

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# ── Theme ─────────────────────────────────────────────────────────────────────

html_theme = "furo"
html_static_path = ["_static"]
html_baseurl = "/refnd/"

html_js_files = [
    ("https://cdn.jsdelivr.net/npm/turndown@7.2.0/dist/turndown.js", {}),
    "js/copy_markdown.js",
]
html_css_files = ["css/copy_markdown.css"]

# ── Autodoc ───────────────────────────────────────────────────────────────────

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "signature"   # put types in the signature, not the body
autodoc_typehints_format = "short"

# ── Napoleon ──────────────────────────────────────────────────────────────────

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = False        # keep Args: block as-is (already nicely written)
napoleon_use_rtype = False

# ── Misc ──────────────────────────────────────────────────────────────────────

nitpicky = False   # compiled extension cross-refs are noisy

# ─────────────────────────────────────────────────────────────────────────────
# Stub-based signature injection
#
# PyO3 compiled extensions don't expose Python type annotations at runtime.
# We parse the pyo3_stub_gen-generated .pyi files with `ast` and inject the
# full typed signatures via the autodoc-process-signature event.
# ─────────────────────────────────────────────────────────────────────────────

def _unparse(node: ast.expr | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\bbuiltins\.", "", ast.unparse(node))


def _extract_sig(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str | None]:
    """Return (params_str, return_str) with self/cls stripped."""
    a = func.args
    skip = {"cls", "self"}

    # Map arg name → default AST node (defaults are right-aligned)
    defaults: dict[str, ast.expr] = {}
    all_pos = a.posonlyargs + a.args
    for arg, dflt in zip(reversed(all_pos), reversed(a.defaults)):
        defaults[arg.arg] = dflt
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        if dflt is not None:
            defaults[arg.arg] = dflt

    def fmt(arg: ast.arg) -> str:
        ann = f": {_unparse(arg.annotation)}" if arg.annotation else ""
        dflt = f" = {_unparse(defaults[arg.arg])}" if arg.arg in defaults else ""
        return f"{arg.arg}{ann}{dflt}"

    parts: list[str] = []

    for arg in a.posonlyargs:
        if arg.arg not in skip:
            parts.append(fmt(arg))
    if a.posonlyargs and any(a.arg not in skip for a in a.posonlyargs):
        parts.append("/")

    for arg in a.args:
        if arg.arg not in skip:
            parts.append(fmt(arg))

    if a.vararg:
        ann = f": {_unparse(a.vararg.annotation)}" if a.vararg.annotation else ""
        parts.append(f"*{a.vararg.arg}{ann}")
    elif a.kwonlyargs:
        parts.append("*")

    for arg in a.kwonlyargs:
        parts.append(fmt(arg))

    if a.kwarg:
        ann = f": {_unparse(a.kwarg.annotation)}" if a.kwarg.annotation else ""
        parts.append(f"**{a.kwarg.arg}{ann}")

    ret = _unparse(func.returns) if func.returns else None
    return "(" + ", ".join(parts) + ")", ret


def _collect(nodes: list[ast.stmt], prefix: str, out: dict[str, tuple[str, str | None]]) -> None:
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[f"{prefix}.{node.name}"] = _extract_sig(node)
        elif isinstance(node, ast.ClassDef):
            cls = f"{prefix}.{node.name}"
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{cls}.{child.name}"] = _extract_sig(child)


def _load_stub_sigs() -> dict[str, tuple[str, str | None]]:
    out: dict[str, tuple[str, str | None]] = {}
    for pyi in sorted(_STUB_DIR.rglob("*.pyi")):
        rel = pyi.relative_to(_STUB_DIR)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts)
        try:
            tree = ast.parse(pyi.read_text())
            _collect(tree.body, module, out)
        except Exception:
            pass
    return out


_STUB_SIGS = _load_stub_sigs()


def _process_signature(app, what, name, obj, options, signature, return_annotation):
    # Properties are documented as attributes — no signature needed.
    if what == "attribute":
        return signature, return_annotation
    # Classes: use their __new__ signature (PyO3 pattern).
    lookup = f"{name}.__new__" if what == "class" else name
    if lookup in _STUB_SIGS:
        sig, ret = _STUB_SIGS[lookup]
        return sig, (None if what == "class" else ret)
    return signature, return_annotation


def setup(app):
    app.connect("autodoc-process-signature", _process_signature)
