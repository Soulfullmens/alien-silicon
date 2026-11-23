"""Judge mutation score: break honest kernels in small ways, count how many the judge catches.

A mutant that survives is either a hole in the judge or an equivalent mutant
(the change does not alter the result). Both are worth a look; the score is a
number for the paper on how hard the judge is to fool.

    python -m judge.mutate          # score every task baseline
"""
from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Iterator
from typing import Any

_MUFU_SWAP = {"exp2": "log2", "log2": "exp2", "rcp": "rsqrt", "rsqrt": "rcp", "sin": "cos", "cos": "sin"}
_REDUCE_SWAP = {"max": "min", "min": "max", "sum": "max"}
_CMP_SWAP = {"lt": "ge", "le": "gt", "gt": "le", "ge": "lt", "eq": "ne", "ne": "eq"}


def mutants(source: str) -> Iterator[tuple[str, str]]:
    """Yield (description, mutated source). Every site is mutated on its own."""
    tree = ast.parse(source)
    sites = [n for n in ast.walk(tree) if isinstance(n, (ast.Call, ast.stmt, ast.Constant, ast.keyword))]
    for i in range(len(sites)):
        for desc, fn in _OPERATORS:
            t = copy.deepcopy(tree)
            node = [n for n in ast.walk(t) if isinstance(n, (ast.Call, ast.stmt, ast.Constant, ast.keyword))][i]
            parent = _parents(t).get(node)
            label = fn(node, parent)
            if label:
                yield f"line {getattr(node, 'lineno', '?')}: {desc} {label}", ast.unparse(t)


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}


def _call_name(node: Any) -> str | None:
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def _delete_stmt(node: Any, parent: Any) -> str | None:
    """Delete one statement inside a kernel body (never a def, the import, or a whole loop)."""
    if not isinstance(node, (ast.Expr, ast.Assign)) or parent is None or not hasattr(parent, "body"):
        return None
    if isinstance(parent, ast.Module) or len(parent.body) < 2:
        return None
    parent.body.remove(node)
    return ast.unparse(node)[:50]


def _bump_int(node: Any, parent: Any) -> str | None:
    if not (isinstance(node, ast.Constant) and type(node.value) is int and _call_name(parent) in ("iadd", "imul")):
        return None
    node.value += 1
    return f"{node.value - 1} -> {node.value}"


def _scale_float(node: Any, parent: Any) -> str | None:
    if not (isinstance(node, ast.Constant) and type(node.value) is float and node.value != 0.0):
        return None
    node.value *= 1.01
    return f"x1.01 ({node.value / 1.01:g})"


def _swap_op(node: Any, parent: Any) -> str | None:
    name = _call_name(node)
    table = {"mufu": _MUFU_SWAP, "reduce": _REDUCE_SWAP, "cmp": _CMP_SWAP}.get(name or "")
    if not table or not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    old = node.args[0].value
    node.args[0].value = table[old]
    return f"{name} {old} -> {table[old]}"


def _swap_args(node: Any, parent: Any) -> str | None:
    if _call_name(node) not in ("sub", "isub") or len(node.args) != 2:
        return None
    node.args.reverse()
    return f"{node.func.id} operands"


def _drop_keyword(node: Any, parent: Any) -> str | None:
    if not (isinstance(node, ast.keyword) and node.arg in ("init", "transpose_b", "keepdims", "clear")):
        return None
    parent.keywords.remove(node)
    return f"{node.arg}="


_OPERATORS: list[tuple[str, Callable[[Any, Any], str | None]]] = [
    ("delete", _delete_stmt), ("bump int", _bump_int), ("scale float", _scale_float),
    ("swap", _swap_op), ("swap", _swap_args), ("drop", _drop_keyword),
]


def score(task: Any, judge: Callable[[str, Any], bool] | None = None) -> dict[str, Any]:
    """Run every mutant of the task's baseline through `judge` (default: the full judge)."""
    if judge is None:
        from . import evaluate

        def judge(src: str, t: Any) -> bool:
            return evaluate(src, t, mufu_ulp_error=2.0)["correct"]
    assert judge(task.baseline_source(), task), "the honest baseline must pass first"
    survivors, total = [], 0
    for desc, src in mutants(task.baseline_source()):
        total += 1
        if judge(src, task):
            survivors.append(desc)
    return {"task": task.id, "mutants": total, "killed": total - len(survivors),
            "score": (total - len(survivors)) / total, "survivors": survivors}


if __name__ == "__main__":
    from aliensilicon.tasks import TASKS

    for t in TASKS.values():
        r = score(t)
        print(f"{t.id}: {r['killed']}/{r['mutants']} killed ({r['score']:.0%})")
        for s in r["survivors"]:
            print(f"   survived: {s}")
