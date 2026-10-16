"""Regression checks for the two preprocessing bugs that were fixed.

``Xenium_pami.py`` and ``visium.py`` are scripts, not modules: importing either
one runs the whole pipeline. So these checks read the source and assert on its
structure rather than executing it, which is what a regression guard needs to do
here -- the two bugs were structural, and both are silent until a slide happens
to hit the branch.

    python test_preprocessing_fixes.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

failures = []


def check(label: str, condition: bool, detail="") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        failures.append(label)
        print(f"  FAIL  {label} {detail}")


def source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def imported_names(src: str) -> set[str]:
    """Top-level module names imported by *src*, including inside functions."""
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def parse_ok(label: str, rel: str) -> None:
    try:
        ast.parse(source(rel))
        check(f"{label} parses", True)
    except SyntaxError as exc:
        check(f"{label} parses", False, exc)


print("== files parse ==")
parse_ok("Xenium_pami.py", "xenium/Xenium_pami.py")
parse_ok("visium.py", "visium/visium.py")

print("== Xenium_pami.py: shutil is imported ==")
xenium = source("xenium/Xenium_pami.py")
names = imported_names(xenium)
check("shutil imported", "shutil" in names, sorted(names))
# remove_all_file recurses with shutil.rmtree; without the import that path
# raises NameError the first time a tile folder contains a subdirectory.
check("shutil is actually used", "shutil.rmtree" in xenium)
check("remove_all_file is defined", "def remove_all_file" in xenium)

print("== Xenium_pami.py: a blank tile is skipped, not dereferenced ==")
lines = xenium.splitlines()
save_line = next((i for i, line in enumerate(lines)
                  if "WSI_320=rgba_image_320_save" in line.replace(" ", "")), None)
check("found the line that reads the sub-patch stack", save_line is not None)

guard_line = next((i for i, line in enumerate(lines)
                   if line.strip() == "if count == 0:"), None)
check("found a count == 0 guard", guard_line is not None)

# There are two: the original one inside the sub-patch loop, which stores the
# first accepted patch, and the guard added before the stack is read. The one
# that matters is the last before the read.
if save_line is not None:
    guards = [i for i, line in enumerate(lines[:save_line])
              if line.strip() == "if count == 0:"]
    check("found a guard before the read", bool(guards), guards)
    guard_line = guards[-1] if guards else None

if save_line is not None and guard_line is not None:
    check("the guard comes before the read", guard_line < save_line,
          f"guard at {guard_line}, read at {save_line}")
    # The guard has to leave the iteration, otherwise the unbound name is still
    # reached on the next line. Comments are skipped so that the explanation
    # above the `continue` does not hide it.
    following = None
    for line in lines[guard_line + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        following = stripped
        break
    check("the guard continues out of the loop", following == "continue", following)

print("== visium.py: no local json2 helper ==")
visium = source("visium/visium.py")
visium_names = imported_names(visium)
check("json2 not imported", "json2" not in visium_names, sorted(visium_names))
check("json imported instead", "json" in visium_names)
check("json2 no longer referenced", "json2" not in visium)
check("the scale factors go through json.load", "json.load(" in visium)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("preprocessing fixes still in place")
