import pytest

from judge.static_check import Rejected, build_submission, check_source

HEADER = "from aliensilicon.ir import GMEM, RMEM, alloc, f16, kernel, load, program_id, store\n"
OK_BODY = '''
@kernel(grid=("rows",), threads=128)
def k(X: GMEM[f16, ("rows", "cols")], Y: GMEM[f16, ("rows", "cols")]):
    r = program_id(0)
    x = alloc(RMEM, ("cols",), f16)
    load(X[r, :], x)
    store(x, Y[r, :])
'''


def test_honest_kernel_passes():
    assert build_submission(HEADER + OK_BODY)["k"].name == "k"


ATTACKS = {
    "import os": "import os\n" + OK_BODY,
    "other module": "from numpy import zeros\n" + OK_BODY,
    "aliased import": "from aliensilicon.ir import kernel as kk\n",
    "non-DSL ir name": "from aliensilicon.ir import Kernel\n",
    "dunder escape": OK_BODY.replace("r = program_id(0)", "r = ().__class__"),
    "builtin call": OK_BODY.replace("r = program_id(0)", "r = open('/etc/passwd')"),
    "eval": OK_BODY.replace("r = program_id(0)", "r = eval('1')"),
    "underscore name": OK_BODY.replace("r = program_id(0)", "_r = program_id(0)"),
    "while loop": OK_BODY.replace("r = program_id(0)", "while True:\n        pass"),
    "python range loop": OK_BODY.replace("r = program_id(0)", "for i in range(3):\n        pass"),
    "lambda": OK_BODY.replace("r = program_id(0)", "r = (lambda: 0)()"),
    "comprehension": OK_BODY.replace("r = program_id(0)", "r = [i for i in X]"),
    "global stmt": OK_BODY.replace("r = program_id(0)", "global Y"),
    "try": OK_BODY.replace("r = program_id(0)", "try:\n        pass\n    except Exception:\n        pass"),
    "top-level code": OK_BODY + "\nprint(1)\n",
    "undecorated fn": "def helper():\n    pass\n",
    "nested fn": OK_BODY.replace("r = program_id(0)", "def g():\n        pass\n    r = 0"),
    "rebind DSL": OK_BODY.replace("r = program_id(0)", "load = 1"),
    "bytes const": OK_BODY.replace("r = program_id(0)", "r = b'x'"),
    "f-string": OK_BODY.replace("r = program_id(0)", "r = f'{X}'"),
    "kwargs splat": OK_BODY.replace("r = program_id(0)", "r = program_id(**X)"),
    "default arg": OK_BODY.replace('Y: GMEM[f16, ("rows", "cols")])', 'Y: GMEM[f16, ("rows", "cols")] = 1)'),
    "binop": OK_BODY.replace("r = program_id(0)", "r = 1 + 2"),
    "unknown name": OK_BODY.replace("r = program_id(0)", "r = secret_reference"),
    "with open": OK_BODY.replace("r = program_id(0)", "with X:\n        pass"),
}


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_attacks_rejected(name):
    with pytest.raises(Rejected):
        check_source(HEADER + ATTACKS[name] if not ATTACKS[name].startswith(("import", "from")) else ATTACKS[name])


def test_size_limit():
    with pytest.raises(Rejected, match="larger"):
        check_source("#" * 200_000)


def test_syntax_error_is_rejected_not_raised():
    with pytest.raises(Rejected, match="syntax"):
        check_source("def (:")
