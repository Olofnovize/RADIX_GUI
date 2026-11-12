RADIX — Setup & Usage Guide
===========================

WHAT THIS IS
------------
RADIX segments plant root cross-section microscopy images into 7 anatomical
regions (whole root, epidermis, exodermis, endodermis, vascular, aerenchyma,
background) using a trained deep learning model (DINOv2 + DPT decoder). You
give it your raw TIFF images, it runs the model, and gives you back:
  - Per-sample prediction masks
  - Overlay preview images
  - A CSV of per-tissue fluorescence intensity measurements

No internet connection or coding is required after setup. Everything runs
locally on your computer.


ONE-TIME SETUP
---------------
1. Install Miniconda if you don't already have Python/conda:
   https://docs.conda.io/en/latest/miniconda.html
   Accept all defaults during installation. Restart your computer after.

2. Run the installer for your operating system:
     Windows:  double-click  install.bat
     Mac:      double-click  install_RADIX.command
   Wait for it to finish (5-15 minutes — it downloads packages, including
   PyTorch). You will see "Installation complete!" when done.

   Mac only: if double-clicking does nothing, right-click the file ->
   Open (you may need to do this once to bypass Gatekeeper), or run it
   from Terminal: bash install_RADIX.command


PREPARING YOUR IMAGES
----------------------
Put your images inside the "data" folder next to this file. Each sample
needs its own subfolder containing 3 TIFF files — one per fluorescence
channel:

    data\
      my_sample_01\
        my_sample_01_DAPI.tif
        my_sample_01_FITC.tif
        my_sample_01_TRITC.tif
      my_sample_02\
        my_sample_02_DAPI.tif
        ...

The channel is matched by filename (case-insensitive), and these
alternate names are also recognized:
    DAPI  channel: DAPI, CW, CALCOFLUOR
    FITC  channel: FITC, GFP
    TRITC channel: TRITC, RFP

A sample is skipped if all 3 channels aren't found — the app will tell you
which ones are missing. If your imaging setup produces a different channel
(e.g. brightfield) instead of one of the three above, contact Stefan before
running a full batch — the model expects exactly these three channel types
as input and a substitute channel may need to be mapped in first.

The trained model checkpoint is already included in the "model" folder —
you do not need to provide one.


RUNNING RADIX
--------------
     Windows:  double-click  start_RADIX.bat
     Mac:      double-click  start_RADIX.command

Your browser opens automatically to the app (http://localhost:8501). If it
doesn't, open that address manually in Chrome/Firefox/Safari.

In the app:
  1. Confirm the "Image folder" (left sidebar) points to the "data" folder
     (this is the default — only change it if your images are elsewhere. 
     You can copy the path ending in ./image into the left sidebar).
  2. Open the "Segmentation inference" panel and click "Run segmentation".
     This processes every sample in your data folder. Progress is shown
     live. A GPU (NVIDIA) will finish in seconds per image; without one,
     expect roughly 30-90 seconds per image on CPU.
     If predictions already exist and you moved them, you can copy the 
     path to the predictions into the left sidebar under 'Prediction folder'.
  3. Once finished, use the "Sample" dropdown to browse each image with its
     predicted tissue overlays (FITC/TRITC), and the "Barrier intensity
     quantification" panel to measure and export intensities per tissue.


WHERE YOUR RESULTS GO
-----------------------
Everything is written next to your "data" folder, in a sibling "ml" folder:
    ml\predictions\   <- one .npy mask file per sample
    ml\vis\           <- overlay preview PNGs (if enabled)
    ml\measurements.csv

You can copy this whole "ml" folder back to share results.


TROUBLESHOOTING
----------------
install.bat / install_RADIX.command crashes the FIRST time you run it,
with an error mentioning "Terms of Service" (or something about
certificates/channels) - this is a one-time Anaconda requirement on newer
conda installs, unrelated to RADIX itself. The installer already tries to
accept this automatically, but if it still fails, open a terminal
(Command Prompt on Windows, Terminal on Mac) and run:
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
Then re-run install.bat / install_RADIX.command. (If your conda is old
enough not to have the "tos" command at all, you don't need this step —
that error won't happen on older conda.)

"conda not found" - Miniconda isn't installed, or wasn't added to PATH.
  Reinstall Miniconda and make sure to restart your computer afterward.

"No prediction files found" - you haven't run segmentation yet. Open the
  "Segmentation inference" panel and click "Run segmentation".

App is very slow - normal on a computer without an NVIDIA GPU; each image
  is a large deep-learning model pass. Let it run in the
  background; it will finish.

Anything else - copy the error message shown in the black terminal window
  (or the "Show command" / error box in the app) and send it to Stefan.


CONTACT
-------
Questions or unexpected results: Stefan Sanow (ssanow@ucdavis.edu or stefansanow7@gmail.com or +1 (530)761-3364)
