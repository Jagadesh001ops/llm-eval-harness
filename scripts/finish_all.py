"""Run the remaining pipeline end to end, unattended: finish the grid, run transfer, run the analysis.

Each step is resumable (finished loops and cached scores are reused), so this can be re-run after any interruption.
Meant to be launched detached so it survives the launching terminal closing; progress goes to stdout.

    python scripts/finish_all.py
"""

import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/run_grid.py", "--intent-judge", "--workers", "4"],
    ["scripts/run_transfer.py", "--workers", "4"],
    ["scripts/run_sensitivity.py", "--workers", "3"],  # test-scores the no-judge picks the analysis needs
    ["analysis/analyze_grid.py"],
]


def main():
    env = dict(os.environ)
    env.pop("SSLKEYLOGFILE", None)  # antivirus-set key-log path can make Python's ssl import fail at random
    for step in STEPS:
        print(f"[{time.strftime('%H:%M:%S')}] START {' '.join(step)}", flush=True)
        rc = subprocess.run([sys.executable, *step], cwd=ROOT, env=env, stdin=subprocess.DEVNULL).returncode
        print(f"[{time.strftime('%H:%M:%S')}] END   {step[0]} rc={rc}", flush=True)
        if rc != 0:
            print("stopping: a step failed; re-run this script to resume", flush=True)
            sys.exit(rc)
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
