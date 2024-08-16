# Segmentation

## Install

- `python -m venv --system-site-packages .venv` (for pi Camera)
- `source .venv/bin/activate`
- `python stream_masks.py`



Notes:
  - Must use old version of Raspbian (Bullseye or Buster)
  - Must use Python 3.8
  - Must install some annoying dependencies
  - Must have good power source
  - Models must be speficially compiled for TPU, and maybe also Raspbian(?)

TODO:
  - get command line segmentation model running
  - get single frame segmentation
  - segment video stream