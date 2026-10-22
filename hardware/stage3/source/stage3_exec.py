"""Execute an existing experiment command with the local toolchain."""
import sys
import subprocess
from stage2_flow import ROOT, ENV

if __name__ == '__main__':
    raise SystemExit(subprocess.call(sys.argv[1:], cwd=ROOT, env=ENV))
