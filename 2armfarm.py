#!/usr/bin/env python3
"""`python 2armfarm.py` — a shim for `simulation/mujoco/two_arm_farm.py`.

⚠️ **The real module is `two_arm_farm.py` and this file is not importable.** A
module whose name starts with a digit is not a legal Python identifier, so
nothing in this repo could ever write `import 2armfarm` — `from farm import
duo` and the rest all have to reach the real thing. Python will happily *run* a
file with this name, which is why the shim works at all, and will refuse to
import it, which is why it must stay a shim rather than hold any code.

Everything lives in `simulation/mujoco/two_arm_farm.py`. This forwards argv to
it unchanged, so every flag documented there works here:

    python 2armfarm.py --seed 7
    python 2armfarm.py --truth --stops 2
"""

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "simulation" / "mujoco" / \
    "two_arm_farm.py"

if __name__ == "__main__":
    if not TARGET.exists():
        raise SystemExit(f"cannot find {TARGET}")
    # `run_path` with __main__ so the target's `if __name__ == "__main__"` fires
    # and `raise SystemExit(main())` propagates its exit code, exactly as if the
    # real file had been run. argv[0] is rewritten so --help prints the real
    # module's usage rather than this shim's name.
    sys.argv[0] = str(TARGET)
    sys.path.insert(0, str(TARGET.parent))
    runpy.run_path(str(TARGET), run_name="__main__")
