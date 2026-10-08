#!/usr/bin/env python3
"""`python 2armfarmtruss.py` — a shim for `simulation/mujoco/two_arm_farm_truss.py`.

⚠️ **The real module is `two_arm_farm_truss.py` and this file is not
importable.** A module whose name starts with a digit is not a legal Python
identifier, so nothing in this repo could ever write `import 2armfarmtruss`.
Python will happily *run* a file with this name, which is why the shim works at
all, and will refuse to import it, which is why it must stay a shim rather than
hold any code. `2armfarm.py` is the same shim for the loose-fruit sim and
carries the same warning.

Everything lives in `simulation/mujoco/two_arm_farm_truss.py`. This forwards
argv to it unchanged, so every flag documented there works here:

    python 2armfarmtruss.py --seed 7
    python 2armfarmtruss.py --truth --stops 2
    python 2armfarmtruss.py --threshold 0.33
"""

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "simulation" / "mujoco" / \
    "two_arm_farm_truss.py"

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
