#!/usr/bin/env bash
# save_screenshot.sh — grab the latest screenshot and stage it for Easy_Start.md
#
# Usage:
#   save_screenshot.sh <slug>
#
# Example:
#   save_screenshot.sh bev-cropped-after-undistort
#   → copies ~/Pictures/Screenshots/<latest>.png
#     to   docs/screenshots/2026-05-27-bev-cropped-after-undistort.png
#
# Override the source folder via SCREENSHOT_SRC env var if your distro
# puts screenshots elsewhere (Ubuntu GNOME default is ~/Pictures/Screenshots).
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $(basename "$0") <slug>"
    echo "  Copies the most recent screenshot to docs/screenshots/<date>-<slug>.png"
    exit 1
fi

SLUG="$1"
DATE="$(date +%Y-%m-%d)"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${REPO_ROOT}/docs/screenshots"
SRC_DIR="${SCREENSHOT_SRC:-${HOME}/Pictures/Screenshots}"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source folder not found: $SRC_DIR"
    echo "Set SCREENSHOT_SRC=<your_folder> and re-run."
    exit 1
fi

# Most recent PNG by modification time
LATEST="$(ls -t "${SRC_DIR}"/*.png 2>/dev/null | head -1)"
if [ -z "${LATEST:-}" ]; then
    echo "ERROR: no PNGs found in $SRC_DIR"
    exit 1
fi

mkdir -p "$DEST_DIR"
DEST="${DEST_DIR}/${DATE}-${SLUG}.png"
cp "$LATEST" "$DEST"
echo "Copied:"
echo "  src:  $LATEST"
echo "  dest: $DEST"
echo ""
echo "Reference in Easy_Start.md as:"
echo "  ![${SLUG}](docs/screenshots/${DATE}-${SLUG}.png)"
