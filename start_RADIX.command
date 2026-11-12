#!/bin/bash
# ============================================================
#  RADIX - Launch the segmentation GUI (macOS)
#  Double-click this file (or a Dock shortcut) to start.
#  Your browser will open automatically at http://localhost:8501
#  Run install_RADIX.command first if you haven't already.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=radix

# -- Find conda --------------------------------------------------
CONDA_EXE=""
for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/miniconda3/bin/conda" \
    "/opt/anaconda3/bin/conda" \
    "/opt/homebrew/Caskroom/miniconda3/base/bin/conda" \
    "/opt/homebrew/anaconda3/bin/conda"
do
    if [ -x "$candidate" ]; then
        CONDA_EXE="$candidate"
        break
    fi
done

if [ -z "$CONDA_EXE" ]; then
    echo ""
    echo " ERROR: conda not found."
    echo " Run install_RADIX.command first, or install Miniconda from:"
    echo " https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    read -p " Press Enter to close..."
    exit 1
fi

cd "$SCRIPT_DIR"

echo " Starting RADIX..."
"$CONDA_EXE" run -n "$ENV_NAME" streamlit run GUI.py --browser.gatherUsageStats false

if [ $? -ne 0 ]; then
    echo ""
    echo " RADIX exited with an error. See output above."
    echo " If this is the first run, make sure you ran install_RADIX.command first."
    read -p " Press Enter to close..."
fi
