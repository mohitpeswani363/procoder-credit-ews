#!/usr/bin/env python3
"""Rebuild the v2 submission results end to end.

    python run_v2.py            # benchmark v2 (all horizons) + Indian replay + report
    python run_v2.py --no-train # reuse saved v2 models; rebuild replay and report only
"""
import argparse, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'credit-ews-matplotlib'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--no-train', action='store_true')
    ap.add_argument('--skip-tests', action='store_true')
    a = ap.parse_args()
    if not a.no_train:
        from ews.benchmark_v2 import run as bench
        bench(ROOT)
    from ews.replay import run as replay
    replay(ROOT)
    if not a.skip_tests:
        subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q'], cwd=ROOT, check=True)
    from ews.final_report import build
    build(ROOT)
    print('Wrote submission/Zeckathon_PS2_procoder_Final_Report.pdf')


if __name__ == '__main__':
    main()
