#!/usr/bin/env python3
"""Regenerate the two final compact hardware vector PDFs from archived reports."""
import argparse, os, subprocess, sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=root/'Images')
    p.add_argument('--qa-dir',type=Path,default=root/'qa')
    args=p.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FROST_FIGURE_OUTPUT=str(args.output_dir.resolve()),FROST_FIGURE_QA=str(args.qa_dir.resolve()))
    for script in ['draw_resources.py','draw_tradeoffs.py']:
        subprocess.run([sys.executable,str(root/'source'/script)],env=env,check=True)
    print('Hardware figures use Stage4/5 report JSON. No new hardware experiment was run.')

if __name__=='__main__':
    main()
