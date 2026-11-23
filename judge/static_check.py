"""Judge step 1 (§14.1): AST allow-list for agent-submitted kernel source.

Agent code may only import DSL names from `aliensilicon.ir`, define decorated
@kernel / @program functions, and call DSL primitives. No attributes, no
underscored names, no other imports or builtins: that closes the usual
escapes (`().__class__.__mro__`, `__import__`, `open`, reading references).
"""
from __future__ import annotations

import ast
from typing import Any

from aliensilicon import ir
from aliensilicon.ir import DSL_NAMES

MAX_SOURCE_BYTES = 100_000

# Statement and expression nodes kernel code may contain; everything else fails.
_ALLOWED_NODES = (
    ast.Module, ast.ImportFrom, ast.alias, ast.FunctionDef, ast.arguments, ast.arg,
    ast.Assign, ast.Expr, ast.For, ast.With, ast.withitem, ast.Pass,
    ast.Call, ast.keyword, ast.Name, ast.Constant, ast.Tuple, ast.Subscript, ast.Slice,
    ast.UnaryOp, ast.USub, ast.Load, ast.Store,
)


class Rejected(Exception):
    """Submission violates the allow-list. Message names the line and rule."""


def _fail(node: ast.AST, msg: str) -> None:
    raise Rejected(f"line {getattr(node, 'lineno', '?')}: {msg}")


def check_source(source: str) -> ast.Module:
    """Return the parsed module if it passes the allow-list, else raise Rejected."""
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise Rejected(f"source larger than {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise Rejected(f"syntax error: {e}") from None

    kernel_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module != "aliensilicon.ir" or node.level != 0:
                _fail(node, "only `from aliensilicon.ir import ...` is allowed")
            for a in node.names:
                if a.name not in DSL_NAMES or a.asname is not None:
                    _fail(node, f"cannot import {a.name!r} (DSL names only, no aliases)")
        elif isinstance(node, ast.FunctionDef):
            kind = _decorator_kind(node)
            if kind == "kernel":
                kernel_names.add(node.name)
        else:
            _fail(node, f"top level may only hold the import and @kernel/@program functions, not {type(node).__name__}")

    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if len(defined) != sum(isinstance(n, ast.FunctionDef) for n in tree.body):
        _fail(tree, "duplicate function names")

    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        _check_function(fn, kernel_names)
    return tree


def _decorator_kind(fn: ast.FunctionDef) -> str:
    if len(fn.decorator_list) != 1:
        _fail(fn, f"{fn.name}: needs exactly one decorator, @kernel(...) or @program")
    d = fn.decorator_list[0]
    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "kernel":
        return "kernel"
    if isinstance(d, ast.Name) and d.id == "program":
        return "program"
    _fail(fn, f"{fn.name}: decorator must be @kernel(...) or @program")
    return ""


def _check_function(fn: ast.FunctionDef, kernel_names: set[str]) -> None:
    a = fn.args
    if a.posonlyargs or a.vararg or a.kwonlyargs or a.kwarg or a.defaults or fn.returns:
        _fail(fn, f"{fn.name}: plain annotated params only (no defaults, *args, **kw, return annotation)")
    if any(isinstance(n, ast.FunctionDef) for n in ast.walk(fn) if n is not fn):
        _fail(fn, f"{fn.name}: nested functions are not allowed")

    local = {p.arg for p in a.args}
    local |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    readable = DSL_NAMES | kernel_names | local
    callable_ = DSL_NAMES | kernel_names

    for node in ast.walk(fn):
        if not isinstance(node, _ALLOWED_NODES) or isinstance(node, (ast.ImportFrom, ast.Module)):
            _fail(node, f"{type(node).__name__} is not allowed in kernel code")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                _fail(node, f"underscored name {node.id!r} is not allowed")
            if isinstance(node.ctx, ast.Store) and node.id in callable_:
                _fail(node, f"cannot rebind DSL/kernel name {node.id!r}")
            if isinstance(node.ctx, ast.Load) and node.id not in readable:
                _fail(node, f"unknown name {node.id!r}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in callable_:
                _fail(node, "only DSL primitives and kernels defined here can be called")
            if any(k.arg is None for k in node.keywords):
                _fail(node, "**kwargs calls are not allowed")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str, bool)) and node.value is not None:
                _fail(node, f"constant of type {type(node.value).__name__} is not allowed")
        elif isinstance(node, ast.For):
            it = node.iter
            if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "serial"):
                _fail(node, "loops must be `for i in serial(n):`")
            if not isinstance(node.target, ast.Name) or node.orelse:
                _fail(node, "loop target must be one name, no else clause")
        elif isinstance(node, ast.With):
            for item in node.items:
                e = item.context_expr
                is_if = isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "if_"
                if not is_if or item.optional_vars:
                    _fail(node, "`with` is only allowed as `with if_(cond):`")
        elif isinstance(node, ast.Assign):
            if any(not isinstance(t, ast.Name) for t in node.targets):
                _fail(node, "assign to plain names only")


def build_submission(source: str) -> dict[str, Any]:
    """Allow-list the source, then build every kernel/program into IR.

    Returns {function name: Kernel | Program}. Raises Rejected or ir.DSLError.
    """
    # shortcut: in-process exec after the allow-list; move into the no-network Docker sandbox (§22) before agent runs.
    tree = check_source(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    env: dict[str, Any] = {name: getattr(ir, name) for name in DSL_NAMES}
    env["__builtins__"] = {}
    exec(compile(tree, "<submission>", "exec"), env)  # noqa: S102 - allow-listed above
    return {n.name: env[n.name].build() for n in tree.body if isinstance(n, ast.FunctionDef)}
