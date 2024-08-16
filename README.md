# Segmentation

## Install

- Must use old version of Raspbian (Bullseye or Buster)
- Must use Python 3.8
- Must have good power source

- `python -m venv --system-site-packages .venv` (for pi Camera)
- `source .venv/bin/activate`
- `python stream_masks.py`


## Notes

- Force the monitor to turn on: `xset dpms force on`


## TODO's

- Compile TF -> TFLite models for the TPU
- Get labels on detected objects
