"""Child-side build: DSL source on stdin -> IR JSON on stdout. Runs only inside a sandbox."""
import json
import sys

from aliensilicon.ir import DSLError
from judge.static_check import Rejected, build_submission


def main() -> None:
    try:
        built = build_submission(sys.stdin.read())
        print(json.dumps({"ok": {name: ir.to_json() for name, ir in built.items()}}))
    except (Rejected, DSLError) as e:
        print(json.dumps({"rejected": str(e)}))


if __name__ == "__main__":
    main()
