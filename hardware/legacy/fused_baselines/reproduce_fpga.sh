#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BUILD_DIR="${BUILD_DIR:-$PWD/build}"
python3 source/setup_flow.py
python3 source/flow.py build --build "$BUILD_DIR/sim"
python3 source/verify.py --build "$BUILD_DIR/sim"
python3 source/flow.py synth
python3 source/flow.py route --freq 8 --seeds 1 2 3
