#!/bin/bash
# ============================================================
#  RADIX - one-time environment setup (macOS)
#  Double-click this file ONCE on a new Mac, then use
#  start_RADIX.command to launch the app.
#
#  Requires: Miniconda or Anaconda installed
#  Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=radix
PYTHON_VERSION=3.11.15
TORCH_VERSION=2.5.1
TORCHVISION_VERSION=0.20.1
TORCHAUDIO_VERSION=2.5.1

echo ""
echo " RADIX - Installing environment..."
echo " This will take a few minutes on first run."
echo ""

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
    echo " ERROR: conda not found. Please install Miniconda first:"
    echo " https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    read -p " Press Enter to close..."
    exit 1
fi

echo " Found conda: $CONDA_EXE"

# -- Accept Anaconda's Terms of Service for the default channels --
# -- Newer conda versions refuse to install anything from the -----
# -- defaults channels until this is accepted; without it, the ----
# -- next step fails with a ToS/certificate-looking error. Older ---
# -- conda versions don't have the "tos" command at all, so -------
# -- failures here are silently ignored (best-effort, non-fatal). --
"$CONDA_EXE" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1
"$CONDA_EXE" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1

# -- Create environment (skip if it already exists) ---------------
if "$CONDA_EXE" env list | grep -qE "^${ENV_NAME} "; then
    echo " Environment \"$ENV_NAME\" already exists - reusing it."
else
    # --override-channels -c conda-forge avoids Anaconda's defaults
    # channels entirely, so the ToS gate never comes up regardless of
    # whether the "tos accept" calls above worked on this conda version.
    "$CONDA_EXE" create -n "$ENV_NAME" --override-channels -c conda-forge python=$PYTHON_VERSION -y
    if [ $? -ne 0 ]; then
        echo ""
        echo " ERROR: Environment creation failed. Check the output above."
        read -p " Press Enter to close..."
        exit 1
    fi
fi

# -- Install PyTorch (CPU/MPS build — Apple Silicon uses MPS automatically) --
echo ""
echo " Installing PyTorch..."
"$CONDA_EXE" run -n "$ENV_NAME" pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION torchaudio==$TORCHAUDIO_VERSION
if [ $? -ne 0 ]; then
    echo ""
    echo " ERROR: PyTorch install failed. Check the output above."
    read -p " Press Enter to close..."
    exit 1
fi

# -- Install remaining dependencies --------------------------------
echo ""
echo " Installing remaining packages..."
"$CONDA_EXE" run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt"
if [ $? -ne 0 ]; then
    echo ""
    echo " ERROR: Installation failed. Check the output above."
    read -p " Press Enter to close..."
    exit 1
fi

echo ""
echo " ============================================================"
echo "  Installation complete!"
echo "  Double-click start_RADIX.command to launch the application."
echo " ============================================================"
echo ""
read -p " Press Enter to close..."
