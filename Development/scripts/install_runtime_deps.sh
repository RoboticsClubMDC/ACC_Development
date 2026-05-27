#!/usr/bin/env bash
# Idempotent runtime Python deps for the qcar2_autonomy stack.
# Safe to call from ~/.bashrc — pip exits 0 quickly if already installed.

set -e

PKGS=(
  tqdm        # pit.YOLO.nets (yolo_detector)
)

python3 -m pip install --quiet --disable-pip-version-check "${PKGS[@]}" >/dev/null 2>&1 || \
  python3 -m pip install --quiet --disable-pip-version-check --user "${PKGS[@]}"

echo "[install_runtime_deps] OK (${PKGS[*]})"
