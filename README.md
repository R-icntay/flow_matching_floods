# flow_matching_floods

Flow-matching experiments for conditional flood-map generation using PyTorch.

This repository is centered around
`simple_flood_simulator_debug.py`,
which contains a notebook-style, end-to-end pipeline for:

- loading monthly GeoTIFF flood maps,
- converting flood masks to signed-distance-function (SDF) targets,
- training a conditional U-Net vector field with flow matching,
- simulating trajectories from noise to flood maps,
- generating marginal flood probability and uncertainty maps.

## Repository contents

- `simple_flood_simulator_debug.py`  
  Main reference implementation (notebook-export style script).
- `simple_flood_simulator.ipynb`  
  Notebook variant.
- `creating_flood_maps.ipynb`  
  Additional data/map creation notebook.

## Data format expected by the simulator

The debug pipeline expects flood maps under:

`flood_data/monthly_flood_maps`

It recursively loads `.tif` files and infers the month from filenames that begin with:

`YYYY-MM...`

Example stem:

`2018-04_N00E039_flood_map`

Within loaded rasters, class value `2` is treated as flooded and converted to a binary mask.

## Environment

Python 3.10+ is required (`pyproject.toml`).

Core libraries used by `simple_flood_simulator_debug.py` include:

- `torch`, `torchvision`
- `numpy`, `scipy`, `matplotlib`
- `rasterio`, `natsort`, `tqdm`
- `scikit-image`

Install project dependencies as needed for your environment (for example via your preferred Python package manager).

## High-level training flow

The debug pipeline:

1. Builds `FloodDataset` and DataLoader from monthly flood maps.
2. Optionally dilates masks and converts them to SDF (`mask_to_sdf`).
3. Defines a Gaussian conditional probability path with linear schedules.
4. Trains `FloodUNet` through `CFGTrainer` with classifier-free guidance-style label dropping.
5. Saves model weights to:
   `full_train_floodnet_cfg_sdf.pth`

## Sampling and analysis

After training, the script demonstrates:

- trajectory visualization (noise -> flood map),
- single-sample conditional simulation,
- Monte Carlo probability/uncertainty map generation (`generate_probability_map`),
- distribution comparisons between generated and real flood extents.

## Notes

- `simple_flood_simulator_debug.py` is a notebook-export script and contains notebook cell markers / IPython calls.
- For iterative experimentation, running the notebook (`.ipynb`) is generally more convenient.
