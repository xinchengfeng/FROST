#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
BUILD_DIR="${BUILD_DIR:-$PWD/build}"
: "${STA:?Set STA to your OpenSTA executable}"
python3 source/run_asic_synthesis.py
python3 source/flatten_gate_netlists.py
python3 source/build_gate_simulators.py --build "$BUILD_DIR/gate"
python3 source/run_gate_power.py --gate-build "$BUILD_DIR/gate" --rtl-build "$BUILD_DIR/sim" --sta "$STA" --lengths 64 256 1024
