#   Dataset Creation
##  Tracking FLAME params

First download the FLAME model, and place the `generic_model.pkl` in `assets/FLAME/FLAME2020`. Copy and rename the `generic_model.pkl` to `flame_generic_model.pkl` and place it at `assets/SMPLX/`

Secondly, create folder `pretrained` and run `assets/Docs/run_download_pretrained.sh` 
Run:
```python
python tracking_flame_only --in_root /path/to/video.mp4 --output_dir /path/to/save_dir
```