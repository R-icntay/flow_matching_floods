from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, distance_transform_edt
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_FLOOD_VALUES = (2,)
DATE_PATTERN = re.compile(r"\d{4}[-_]\d{2}[-_]\d{2}")
TILE_PATTERN = re.compile(r"[NS]\d{2}[EW]\d{3}")
TARGET_SUFFIX = "_flood_map"
DEFAULT_TARGET_GLOB = "**/*.tif"


@dataclass
class FeatureSpec:
    name: str
    stem_templates: tuple[str, ...]
    band_names: tuple[str, ...] = ()
    categorical: bool = False
    required: bool = True
    fill_value: float = 0.0
    path_column: str = ""

    def __post_init__(self) -> None:
        if not self.path_column:
            self.path_column = f"{self.name}_path"

    def candidate_stems(self, tile_name: str, acquisition_date: str) -> list[str]:
        date_token = acquisition_date.replace("-", "_")
        return [
            template.format(
                tile_name=tile_name,
                acquisition_date=acquisition_date,
                date_token=date_token,
            )
            for template in self.stem_templates
        ]


@dataclass(frozen=True)
class TargetGrid:
    crs: Any
    transform: Affine
    height: int
    width: int


def default_feature_specs(
    *,
    include_chirps_30d: bool = False,
    include_era5_sm_30d: bool = False,
) -> list[FeatureSpec]:
    specs = [
        FeatureSpec("dem", ("{tile_name}_DEM",), ("dem",)),
        FeatureSpec("slope", ("{tile_name}_Slope",), ("slope",)),
        FeatureSpec(
            "merit_flow_accumulation",
            ("{tile_name}_MERIT_Hydro_flow_accumulation",),
            ("flow_acc_log10",),
        ),
        FeatureSpec(
            "merit_hand",
            ("{tile_name}_MERIT_Hydro_hand",),
            ("hand_m",),
        ),
        FeatureSpec(
            "soil_clay",
            ("{tile_name}_Soil_Static_clay",),
            ("clay_b10", "clay_b30", "clay_b60"),
        ),
        FeatureSpec(
            "esa_worldcover",
            ("{tile_name}_ESA_WorldCover_v200",),
            ("land_cover",),
            categorical=True,
        ),
        FeatureSpec(
            "chirps_daily",
            ("{tile_name}_CHIRPS_precip_{date_token}",),
            ("chirps_daily",),
        ),
        FeatureSpec(
            "chirps_3d_sum",
            ("{tile_name}_CHIRPS_precip_3d_sum_{date_token}",),
            ("chirps_3d_sum",),
        ),
        FeatureSpec(
            "chirps_7d_sum",
            ("{tile_name}_CHIRPS_precip_7d_sum_{date_token}",),
            ("chirps_7d_sum",),
        ),
        FeatureSpec(
            "chirps_14d_sum",
            ("{tile_name}_CHIRPS_precip_14d_sum_{date_token}",),
            ("chirps_14d_sum",),
        ),
        FeatureSpec(
            "era5_sm_daily",
            ("{tile_name}_ERA5_SM_daily_mean_{date_token}",),
            ("era5_sm_daily_l1", "era5_sm_daily_l2", "era5_sm_daily_l3"),
        ),
        FeatureSpec(
            "era5_sm_7d_mean",
            ("{tile_name}_ERA5_SM_7d_mean_{date_token}",),
            ("era5_sm_7d_l1", "era5_sm_7d_l2", "era5_sm_7d_l3"),
        ),
        FeatureSpec(
            "era5_sm_14d_mean",
            ("{tile_name}_ERA5_SM_14d_mean_{date_token}",),
            ("era5_sm_14d_l1", "era5_sm_14d_l2", "era5_sm_14d_l3"),
        ),
        FeatureSpec(
            "era5_temp_daily",
            ("{tile_name}_ERA5_TEMP_daily_mean_{date_token}",),
            ("era5_temp_daily_mean",),
        ),
        FeatureSpec(
            "era5_temp_7d_mean",
            ("{tile_name}_ERA5_TEMP_7d_mean_{date_token}",),
            ("era5_temp_7d_mean",),
        ),
        FeatureSpec(
            "era5_runoff_daily",
            ("{tile_name}_ERA5_Runoff_daily_sum_{date_token}",),
            ("surface_runoff_daily", "runoff_daily"),
        ),
        FeatureSpec(
            "era5_runoff_3d_sum",
            ("{tile_name}_ERA5_Runoff_3d_sum_{date_token}",),
            ("surface_runoff_3d_sum", "runoff_3d_sum"),
        ),
        FeatureSpec(
            "era5_runoff_7d_sum",
            ("{tile_name}_ERA5_Runoff_7d_sum_{date_token}",),
            ("surface_runoff_7d_sum", "runoff_7d_sum"),
        ),
        FeatureSpec(
            "era5_runoff_14d_sum",
            ("{tile_name}_ERA5_Runoff_14d_sum_{date_token}",),
            ("surface_runoff_14d_sum", "runoff_14d_sum"),
        ),
        FeatureSpec(
            "modis_ndvi_recent",
            ("{tile_name}_MODIS_NDVI_recent_32d_{date_token}",),
            ("modis_ndvi_recent",),
        ),
    ]

    if include_chirps_30d:
        specs.append(
            FeatureSpec(
                "chirps_30d_sum",
                ("{tile_name}_CHIRPS_precip_30d_sum_{date_token}",),
                ("chirps_30d_sum",),
            )
        )
    if include_era5_sm_30d:
        specs.append(
            FeatureSpec(
                "era5_sm_30d_mean",
                ("{tile_name}_ERA5_SM_30d_mean_{date_token}",),
                ("era5_sm_30d_l1", "era5_sm_30d_l2", "era5_sm_30d_l3"),
            )
        )
    return specs


def parse_target_identity(path: str | Path) -> tuple[str, str]:
    path = Path(path)
    stem = path.stem
    if TARGET_SUFFIX not in stem:
        raise ValueError(f"Target file is missing expected suffix {TARGET_SUFFIX}: {path}")

    date_match = DATE_PATTERN.search(stem)
    tile_match = TILE_PATTERN.search(stem)
    if date_match is None or tile_match is None:
        raise ValueError(f"Could not parse tile/date from target file name: {path}")

    acquisition_date = date_match.group(0).replace("_", "-")
    tile_name = tile_match.group(0)
    return tile_name, acquisition_date


def build_raster_index(search_roots: Iterable[str | Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for raster_path in root_path.rglob("*.tif"):
            index.setdefault(raster_path.stem, raster_path)
    return index


def build_flood_manifest(
    data_root: str | Path,
    *,
    target_root: str | Path | None = None,
    target_glob: str = DEFAULT_TARGET_GLOB,
    feature_specs: Sequence[FeatureSpec] | None = None,
    manifest_path: str | Path | None = None,
    compute_flood_pixel_count: bool = False,
    drop_incomplete: bool = True,
    flood_values: Sequence[int] = DEFAULT_FLOOD_VALUES,
    assign_splits: bool = True,
    gap_days: int = 14,
    temporal_train_ratio: float = 0.70,
    temporal_val_ratio: float = 0.15,
    random_seed: int = 0,
) -> pd.DataFrame:
    specs = list(feature_specs or default_feature_specs())
    data_root = Path(data_root)
    target_root = Path(target_root) if target_root is not None else data_root

    raster_index = build_raster_index([data_root])
    rows: list[dict[str, Any]] = []
    for target_path in sorted(target_root.glob(target_glob)):
        if not target_path.is_file() or target_path.suffix.lower() != ".tif":
            continue
        try:
            tile_name, acquisition_date = parse_target_identity(target_path)
        except ValueError:
            continue

        date_ts = pd.Timestamp(acquisition_date)
        row: dict[str, Any] = {
            "tile_name": tile_name,
            "acquisition_date": acquisition_date,
            "year": int(date_ts.year),
            "month": int(date_ts.month),
            "month_idx": int(date_ts.month - 1),
            "target_path": str(target_path),
        }
        missing_required: list[str] = []
        missing_any: list[str] = []
        for spec in specs:
            resolved_path: Path | None = None
            for stem in spec.candidate_stems(tile_name, acquisition_date):
                resolved_path = raster_index.get(stem)
                if resolved_path is not None:
                    break
            row[spec.path_column] = str(resolved_path) if resolved_path is not None else None
            missing_column = f"missing_{spec.name}"
            row[missing_column] = resolved_path is None
            if resolved_path is None:
                missing_any.append(spec.name)
                if spec.required:
                    missing_required.append(spec.name)

        row["missing_features"] = ",".join(missing_any)
        row["missing_required_features"] = ",".join(missing_required)
        row["is_complete"] = not missing_required
        if compute_flood_pixel_count:
            row["flood_pixel_count"] = compute_target_flood_pixel_count(target_path, flood_values=flood_values)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(
            f"No date-aligned targets were found under {target_root} using glob {target_glob}"
        )

    manifest = add_event_ids(manifest, gap_days=gap_days)
    if assign_splits:
        manifest = assign_default_splits(
            manifest,
            gap_days=gap_days,
            temporal_train_ratio=temporal_train_ratio,
            temporal_val_ratio=temporal_val_ratio,
            random_seed=random_seed,
        )

    if drop_incomplete:
        manifest = manifest.loc[manifest["is_complete"]].reset_index(drop=True)
    else:
        manifest = manifest.reset_index(drop=True)

    if manifest_path is not None:
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    return manifest


def compute_target_flood_pixel_count(
    target_path: str | Path,
    *,
    flood_values: Sequence[int] = DEFAULT_FLOOD_VALUES,
) -> int:
    with rasterio.open(target_path) as src:
        arr = src.read(1)
    flood_mask = np.isin(arr, tuple(flood_values))
    return int(flood_mask.sum())


def add_event_ids(manifest: pd.DataFrame, *, gap_days: int = 14) -> pd.DataFrame:
    required_columns = {"tile_name", "acquisition_date"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns required for event grouping: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["acquisition_date"] = pd.to_datetime(manifest["acquisition_date"])
    manifest = manifest.sort_values(["tile_name", "acquisition_date", "target_path"]).reset_index(drop=True)

    event_ids: list[str] = []
    event_counter_by_tile: dict[str, int] = {}
    previous_date_by_tile: dict[str, pd.Timestamp] = {}

    for row in manifest.itertuples(index=False):
        tile_name = row.tile_name
        acquisition_date = row.acquisition_date
        previous = previous_date_by_tile.get(tile_name)
        if previous is None or (acquisition_date - previous).days > gap_days:
            event_counter_by_tile[tile_name] = event_counter_by_tile.get(tile_name, 0) + 1
        event_ids.append(f"{tile_name}_event_{event_counter_by_tile[tile_name]:03d}")
        previous_date_by_tile[tile_name] = acquisition_date

    manifest["event_id"] = event_ids
    event_bounds = manifest.groupby("event_id", as_index=True)["acquisition_date"].agg(["min", "max"])
    manifest["event_start_date"] = manifest["event_id"].map(event_bounds["min"])
    manifest["event_end_date"] = manifest["event_id"].map(event_bounds["max"])
    manifest["acquisition_date"] = manifest["acquisition_date"].dt.strftime("%Y-%m-%d")
    manifest["event_start_date"] = pd.to_datetime(manifest["event_start_date"]).dt.strftime("%Y-%m-%d")
    manifest["event_end_date"] = pd.to_datetime(manifest["event_end_date"]).dt.strftime("%Y-%m-%d")
    return manifest


def assign_default_splits(
    manifest: pd.DataFrame,
    *,
    gap_days: int = 14,
    temporal_train_ratio: float = 0.70,
    temporal_val_ratio: float = 0.15,
    random_seed: int = 0,
) -> pd.DataFrame:
    required_columns = {"tile_name", "event_id", "event_start_date", "event_end_date"}
    missing = required_columns.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns required for splitting: {sorted(missing)}")

    manifest = manifest.copy()
    if "flood_pixel_count" not in manifest.columns:
        manifest["flood_pixel_count"] = 0

    tile_stats = (
        manifest.groupby("tile_name", as_index=False)
        .agg(
            event_count=("event_id", "nunique"),
            flood_pixels=("flood_pixel_count", "sum"),
        )
        .sort_values(["event_count", "flood_pixels", "tile_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    tile_groups = _balanced_tile_split(tile_stats, random_seed=random_seed)
    manifest["tile_split"] = manifest["tile_name"].map(tile_groups)

    temporal_split_by_event = _assign_temporal_splits(
        manifest,
        gap_days=gap_days,
        train_ratio=temporal_train_ratio,
        val_ratio=temporal_val_ratio,
    )
    manifest["temporal_split"] = manifest["event_id"].map(temporal_split_by_event)

    final_split: list[str | None] = []
    eval_group: list[str] = []
    for row in manifest.itertuples(index=False):
        if row.tile_split == "train":
            if row.temporal_split == "train":
                final_split.append("train")
                eval_group.append("train")
            elif row.temporal_split == "val":
                final_split.append("val")
                eval_group.append("temporal_seen_tiles")
            elif row.temporal_split == "test":
                final_split.append("test")
                eval_group.append("temporal_seen_tiles")
            else:
                final_split.append(None)
                eval_group.append("purged")
        elif row.tile_split == "val":
            final_split.append("val")
            eval_group.append("spatial_unseen_tiles")
        elif row.tile_split == "test":
            final_split.append("test")
            eval_group.append("spatial_unseen_tiles")
        else:
            final_split.append(None)
            eval_group.append("unassigned")

    manifest["split"] = final_split
    manifest["eval_group"] = eval_group
    return manifest


def validate_split_manifest(manifest: pd.DataFrame, *, lookback_days: int = 14) -> dict[str, Any]:
    manifest = manifest.copy()
    checks: dict[str, Any] = {}
    train_tiles = set(
        manifest.loc[
            (manifest["split"] == "train") & (manifest["eval_group"] == "train"),
            "tile_name",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    spatial_val_tiles = set(
        manifest.loc[
            (manifest["split"] == "val") & (manifest["eval_group"] == "spatial_unseen_tiles"),
            "tile_name",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    spatial_test_tiles = set(
        manifest.loc[
            (manifest["split"] == "test") & (manifest["eval_group"] == "spatial_unseen_tiles"),
            "tile_name",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    checks["spatial_tile_overlap"] = {
        "train_val": sorted(train_tiles.intersection(spatial_val_tiles)),
        "train_test": sorted(train_tiles.intersection(spatial_test_tiles)),
        "val_test": sorted(spatial_val_tiles.intersection(spatial_test_tiles)),
    }

    event_sets = {
        split: set(manifest.loc[manifest["split"] == split, "event_id"].dropna().unique().tolist())
        for split in ("train", "val", "test")
    }
    checks["event_overlap"] = {
        "train_val": sorted(event_sets["train"].intersection(event_sets["val"])),
        "train_test": sorted(event_sets["train"].intersection(event_sets["test"])),
        "val_test": sorted(event_sets["val"].intersection(event_sets["test"])),
    }

    manifest["event_start_date"] = pd.to_datetime(manifest["event_start_date"])
    temporal_rows = manifest.loc[manifest["eval_group"] == "temporal_seen_tiles"].copy()
    train_rows = manifest.loc[(manifest["tile_split"] == "train") & (manifest["temporal_split"] == "train")].copy()
    train_ranges = train_rows[["tile_name", "event_end_date"]].drop_duplicates()
    train_ranges["event_end_date"] = pd.to_datetime(train_ranges["event_end_date"])

    purge_violations: list[str] = []
    for row in temporal_rows.itertuples(index=False):
        candidate = train_ranges.loc[train_ranges["tile_name"] == row.tile_name]
        if candidate.empty:
            continue
        if ((pd.to_datetime(row.event_start_date) - candidate["event_end_date"]).dt.days <= lookback_days).any():
            purge_violations.append(row.event_id)
    checks["purge_violations"] = sorted(set(purge_violations))
    return checks


class FloodConditioningDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        *,
        split: str | None = None,
        feature_specs: Sequence[FeatureSpec] | None = None,
        target_size: tuple[int, int] = (256, 256),
        target_mode: str = "sdf",
        sdf_threshold: float = 2.0,
        dilate_target: bool = False,
        target_dilation_structure: np.ndarray | None = None,
        flood_values: Sequence[int] = DEFAULT_FLOOD_VALUES,
        derived_features: Sequence[str] = (),
        normalization_stats: Mapping[str, tuple[float, float]] | None = None,
        allow_missing_features: bool = False,
        tile_to_idx: Mapping[str, int] | None = None,
    ) -> None:
        self.feature_specs = list(feature_specs or default_feature_specs())
        self.feature_specs_by_name = {spec.name: spec for spec in self.feature_specs}
        self.target_size = target_size
        self.target_mode = target_mode
        self.sdf_threshold = sdf_threshold
        self.dilate_target = dilate_target
        self.target_dilation_structure = (
            target_dilation_structure if target_dilation_structure is not None else np.ones((3, 3), dtype=np.uint8)
        )
        self.flood_values = tuple(flood_values)
        self.derived_features = tuple(derived_features)
        self.normalization_stats = normalization_stats or {}
        self.allow_missing_features = allow_missing_features

        full_manifest = _coerce_manifest(manifest)
        manifest_df = full_manifest.copy()
        if split is not None:
            manifest_df = manifest_df.loc[manifest_df["split"] == split].reset_index(drop=True)
        else:
            manifest_df = manifest_df.reset_index(drop=True)

        if manifest_df.empty:
            raise ValueError("The manifest is empty after applying the requested split filter.")

        self.manifest = manifest_df
        self.split = split
        self.tile_to_idx = dict(tile_to_idx) if tile_to_idx is not None else {
            tile_name: idx
            for idx, tile_name in enumerate(sorted(full_manifest["tile_name"].dropna().unique().tolist()))
        }
        self.channel_names = self._build_channel_names()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]
        target_path = Path(row["target_path"])
        grid = _build_target_grid(target_path, self.target_size)
        target = self._load_target(target_path)

        conditioning_tensors: list[np.ndarray] = []
        channel_names: list[str] = []
        for spec in self.feature_specs:
            feature_path = row.get(spec.path_column)
            if pd.isna(feature_path) or feature_path is None:
                if not self.allow_missing_features and spec.required:
                    raise FileNotFoundError(
                        f"Missing required feature {spec.name} for sample {row['tile_name']} {row['acquisition_date']}"
                    )
                feature_array = np.full((len(spec.band_names) or 1, *self.target_size), spec.fill_value, dtype=np.float32)
            else:
                feature_array = _load_aligned_raster(
                    Path(feature_path),
                    grid,
                    categorical=spec.categorical,
                    fill_value=spec.fill_value,
                )
            conditioning_tensors.append(feature_array)
            channel_names.extend(_band_names_for_array(spec, feature_array.shape[0]))

        conditioning = np.concatenate(conditioning_tensors, axis=0).astype(np.float32, copy=False)
        conditioning, channel_names = self._append_derived_channels(conditioning, channel_names)
        conditioning = self._normalize(conditioning, channel_names)

        item = {
            "target": torch.from_numpy(target).to(torch.float32),
            "conditioning": torch.from_numpy(conditioning).to(torch.float32),
            "month_idx": int(row["month_idx"]),
            "tile_idx": int(self.tile_to_idx[row["tile_name"]]),
            "date": str(row["acquisition_date"]),
            "event_id": str(row["event_id"]),
            "tile_name": str(row["tile_name"]),
            "split": row.get("split"),
            "eval_group": row.get("eval_group"),
            "channel_names": channel_names,
            "target_path": str(target_path),
        }
        return item

    def _load_target(self, target_path: Path) -> np.ndarray:
        with rasterio.open(target_path) as src:
            target = src.read(1, out_shape=self.target_size, resampling=Resampling.nearest)

        target = np.isin(target, self.flood_values).astype(np.uint8)
        if self.dilate_target:
            target = binary_dilation(target, structure=self.target_dilation_structure).astype(np.uint8)
        if self.target_mode == "mask":
            return target[None, :, :].astype(np.float32)
        if self.target_mode == "sdf":
            return self.mask_to_sdf(target, truncation_threshold=self.sdf_threshold)[None, :, :].astype(np.float32)
        raise ValueError(f"Unsupported target_mode: {self.target_mode}")

    def _build_channel_names(self) -> list[str]:
        names: list[str] = []
        for spec in self.feature_specs:
            names.extend(_band_names_for_array(spec, len(spec.band_names) or 1))
        if "sm_anomaly" in self.derived_features:
            names.extend(("sm_anomaly_l1", "sm_anomaly_l2", "sm_anomaly_l3"))
        if "precip_intensity_ratio" in self.derived_features:
            names.append("precip_intensity_ratio")
        if "runoff_ratio" in self.derived_features:
            names.extend(("surface_runoff_ratio", "runoff_ratio"))
        return names

    def _append_derived_channels(
        self,
        conditioning: np.ndarray,
        channel_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        if not self.derived_features:
            return conditioning, channel_names

        derived_arrays: list[np.ndarray] = []
        derived_names: list[str] = []
        name_to_idx = {name: idx for idx, name in enumerate(channel_names)}
        epsilon = 1e-6

        if "sm_anomaly" in self.derived_features:
            daily_names = ("era5_sm_daily_l1", "era5_sm_daily_l2", "era5_sm_daily_l3")
            mean_names = ("era5_sm_14d_l1", "era5_sm_14d_l2", "era5_sm_14d_l3")
            if all(name in name_to_idx for name in (*daily_names, *mean_names)):
                sm_anomaly = conditioning[[name_to_idx[name] for name in daily_names]] - conditioning[
                    [name_to_idx[name] for name in mean_names]
                ]
                derived_arrays.append(sm_anomaly)
                derived_names.extend(("sm_anomaly_l1", "sm_anomaly_l2", "sm_anomaly_l3"))

        if "precip_intensity_ratio" in self.derived_features:
            if "chirps_daily" in name_to_idx and "chirps_14d_sum" in name_to_idx:
                ratio = conditioning[name_to_idx["chirps_daily"]] / (
                    (conditioning[name_to_idx["chirps_14d_sum"]] / 14.0) + epsilon
                )
                derived_arrays.append(ratio[None, :, :])
                derived_names.append("precip_intensity_ratio")

        if "runoff_ratio" in self.derived_features:
            required = ("surface_runoff_daily", "runoff_daily", "chirps_daily")
            if all(name in name_to_idx for name in required):
                chirps_daily = conditioning[name_to_idx["chirps_daily"]]
                runoff_ratio = np.stack(
                    [
                        conditioning[name_to_idx["surface_runoff_daily"]] / (chirps_daily + epsilon),
                        conditioning[name_to_idx["runoff_daily"]] / (chirps_daily + epsilon),
                    ],
                    axis=0,
                )
                derived_arrays.append(runoff_ratio)
                derived_names.extend(("surface_runoff_ratio", "runoff_ratio"))

        if not derived_arrays:
            return conditioning, channel_names
        conditioning = np.concatenate([conditioning, *derived_arrays], axis=0)
        return conditioning, channel_names + derived_names

    def _normalize(self, conditioning: np.ndarray, channel_names: Sequence[str]) -> np.ndarray:
        if not self.normalization_stats:
            return conditioning

        normalized = conditioning.copy()
        for channel_idx, channel_name in enumerate(channel_names):
            stats = self.normalization_stats.get(channel_name)
            if stats is None:
                continue
            mean, std = stats
            if std == 0:
                continue
            normalized[channel_idx] = (normalized[channel_idx] - mean) / std
        return normalized

    @staticmethod
    def mask_to_sdf(binary_mask: np.ndarray, truncation_threshold: float) -> np.ndarray:
        binary_mask = np.asarray(binary_mask).astype(np.float32)
        dist_outside = distance_transform_edt(1 - binary_mask)
        dist_inside = distance_transform_edt(binary_mask)
        sdf = dist_outside - dist_inside
        sdf_truncated = np.clip(sdf, -truncation_threshold, truncation_threshold)
        return sdf_truncated / truncation_threshold


def flood_collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first = batch[0]
    collated = {
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "conditioning": torch.stack([item["conditioning"] for item in batch], dim=0),
        "month_idx": torch.tensor([item["month_idx"] for item in batch], dtype=torch.long),
        "tile_idx": torch.tensor([item["tile_idx"] for item in batch], dtype=torch.long),
        "date": [item["date"] for item in batch],
        "event_id": [item["event_id"] for item in batch],
        "tile_name": [item["tile_name"] for item in batch],
        "split": [item["split"] for item in batch],
        "eval_group": [item["eval_group"] for item in batch],
        "target_path": [item["target_path"] for item in batch],
        "channel_names": first["channel_names"],
    }
    return collated


def create_flood_dataloader(
    dataset: FloodConditioningDataset,
    *,
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    if shuffle is None:
        shuffle = dataset.split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=flood_collate_fn,
    )


def _coerce_manifest(manifest: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    manifest_path = Path(manifest)
    if manifest_path.suffix.lower() == ".parquet":
        return pd.read_parquet(manifest_path)
    return pd.read_csv(manifest_path)


def _build_target_grid(target_path: Path, target_size: tuple[int, int]) -> TargetGrid:
    with rasterio.open(target_path) as src:
        height, width = target_size
        scale_x = src.width / float(width)
        scale_y = src.height / float(height)
        transform = src.transform * Affine.scale(scale_x, scale_y)
        return TargetGrid(
            crs=src.crs,
            transform=transform,
            height=height,
            width=width,
        )


def _load_aligned_raster(
    raster_path: Path,
    grid: TargetGrid,
    *,
    categorical: bool,
    fill_value: float,
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        destination = np.full((src.count, grid.height, grid.width), fill_value, dtype=np.float32)
        resampling = Resampling.nearest if categorical else Resampling.bilinear
        for band_idx in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=destination[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                src_nodata=src.nodata,
                dst_nodata=fill_value,
                resampling=resampling,
            )
    return destination


def _band_names_for_array(spec: FeatureSpec, count: int) -> list[str]:
    if spec.band_names and len(spec.band_names) == count:
        return list(spec.band_names)
    if spec.band_names and len(spec.band_names) > count:
        return list(spec.band_names[:count])
    if count == 1:
        return [spec.name]
    return [f"{spec.name}_{band_idx + 1}" for band_idx in range(count)]


def _balanced_tile_split(tile_stats: pd.DataFrame, *, random_seed: int) -> dict[str, str]:
    del random_seed
    tile_names = tile_stats["tile_name"].tolist()
    quotas = _tile_group_quotas(len(tile_names))
    flood_max = max(float(tile_stats["flood_pixels"].max()), 1.0)
    event_max = max(float(tile_stats["event_count"].max()), 1.0)
    tile_stats = tile_stats.copy()
    tile_stats["balance_weight"] = (
        tile_stats["event_count"].astype(float) / event_max
    ) + (
        tile_stats["flood_pixels"].astype(float) / flood_max
    )

    group_loads = {group: 0.0 for group in quotas}
    group_counts = {group: 0 for group in quotas}
    tile_groups: dict[str, str] = {}
    for row in tile_stats.itertuples(index=False):
        available_groups = [group for group, quota in quotas.items() if group_counts[group] < quota]
        selected_group = min(
            available_groups,
            key=lambda group: (group_loads[group], group_counts[group], group),
        )
        tile_groups[row.tile_name] = selected_group
        group_counts[selected_group] += 1
        group_loads[selected_group] += row.balance_weight
    return tile_groups


def _tile_group_quotas(n_tiles: int) -> dict[str, int]:
    if n_tiles < 3:
        raise ValueError("At least three tiles are required to build train/val/test spatial splits.")
    if n_tiles == 12:
        return {"train": 8, "val": 2, "test": 2}

    n_train = max(1, round(n_tiles * 0.67))
    n_val = max(1, round(n_tiles * 0.165))
    n_test = n_tiles - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    return {"train": n_train, "val": n_val, "test": n_test}


def _assign_temporal_splits(
    manifest: pd.DataFrame,
    *,
    gap_days: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str | None]:
    manifest = manifest.copy()
    manifest["event_start_date"] = pd.to_datetime(manifest["event_start_date"])
    manifest["event_end_date"] = pd.to_datetime(manifest["event_end_date"])

    event_splits: dict[str, str | None] = {}
    train_tiles = sorted(manifest.loc[manifest["tile_split"] == "train", "tile_name"].dropna().unique().tolist())
    for tile_name in train_tiles:
        tile_events = (
            manifest.loc[manifest["tile_name"] == tile_name, ["event_id", "event_start_date", "event_end_date"]]
            .drop_duplicates()
            .sort_values(["event_start_date", "event_id"])
            .reset_index(drop=True)
        )
        if tile_events.empty:
            continue

        quotas = _temporal_group_quotas(len(tile_events), train_ratio=train_ratio, val_ratio=val_ratio)
        labels = (["train"] * quotas["train"]) + (["val"] * quotas["val"]) + (["test"] * quotas["test"])
        if len(labels) < len(tile_events):
            labels.extend(["test"] * (len(tile_events) - len(labels)))
        labels = labels[: len(tile_events)]
        assignment = {row.event_id: labels[idx] for idx, row in enumerate(tile_events.itertuples(index=False))}
        assignment = _apply_purge_gap(tile_events, assignment, gap_days=gap_days)
        event_splits.update(assignment)

    return event_splits


def _temporal_group_quotas(
    n_events: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, int]:
    if n_events == 1:
        return {"train": 1, "val": 0, "test": 0}
    if n_events == 2:
        return {"train": 1, "val": 0, "test": 1}

    n_train = max(1, int(round(n_events * train_ratio)))
    n_val = max(1, int(round(n_events * val_ratio)))
    n_test = n_events - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
    while n_train + n_val + n_test > n_events:
        if n_train > max(1, n_val):
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return {"train": n_train, "val": n_val, "test": n_test}


def _apply_purge_gap(
    tile_events: pd.DataFrame,
    assignment: dict[str, str | None],
    *,
    gap_days: int,
) -> dict[str, str | None]:
    assignment = dict(assignment)
    ordered_events = list(tile_events.itertuples(index=False))
    for left_label, right_label in (("train", "val"), ("val", "test")):
        left_events = [row for row in ordered_events if assignment.get(row.event_id) == left_label]
        right_events = [row for row in ordered_events if assignment.get(row.event_id) == right_label]
        if not left_events or not right_events:
            continue

        left_boundary = left_events[-1]
        for right_event in right_events:
            gap = (right_event.event_start_date - left_boundary.event_end_date).days
            if gap <= gap_days:
                assignment[right_event.event_id] = None
            else:
                break
    return assignment


__all__ = [
    "DEFAULT_FLOOD_VALUES",
    "FeatureSpec",
    "FloodConditioningDataset",
    "add_event_ids",
    "assign_default_splits",
    "build_flood_manifest",
    "build_raster_index",
    "create_flood_dataloader",
    "default_feature_specs",
    "flood_collate_fn",
    "parse_target_identity",
    "validate_split_manifest",
]