"""Wait for the complete measured pool, then verify, summarize and author."""
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args();root=a.root.resolve()
    while len(list((root/'runs').glob('image_*.json')))<200:
        if not a.watch:raise RuntimeError('The 200-image evaluation is incomplete')
        time.sleep(5)
    for name in ['summarize_bsds500.py','write_manuscript.py','plot_bsds500.py']:
        print('RUN',name,flush=True)
        subprocess.run([sys.executable,str(Path(__file__).resolve().parent/name),
                        '--root',str(root)],check=True)
    print('VERIFIED AND AUTHORED all 200-image statistics, LaTeX, and vector figure',flush=True)


if __name__=='__main__':main()
