#!/usr/bin/env python3
"""
Print the source file of a Python module in the current environment.

Usage:
    python get_pym_src.py <module.name> [<module.name> ...]

Examples:
    python get_pym_src.py ai2_kit.tool.ase
    python get_pym_src.py mace.calculators ase.io
"""

import sys
import importlib
import importlib.util


def print_module_source(module_name: str) -> None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"# ERROR: module '{module_name}' not found in current environment.",
              file=sys.stderr)
        sys.exit(1)

    origin = spec.origin
    if origin is None or origin == "built-in":
        print(f"# ERROR: '{module_name}' is a built-in module with no source file.",
              file=sys.stderr)
        sys.exit(1)

    print(f"# Source: {origin}\n", file=sys.stderr)
    with open(origin, "r", encoding="utf-8") as fh:
        print(fh.read())


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    for name in sys.argv[1:]:
        print_module_source(name)


if __name__ == "__main__":
    main()
