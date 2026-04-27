# Data Download Plan: Conditioning Variables for Flood Extent Simulator

Flood extent simulation using conditional flow matching

We develop a generative model for flood extent simulation that learns the probabilistic relationship between hydrometeorological forcing conditions and inundation patterns across East Africa. Using 10 years of Sentinel-1 SAR flood detections from Misra et al. (2025) as ground truth, we train a flow matching model to generate plausible flood extent masks conditioned on a multiresolution feature stack comprising static topographic susceptibility layers — elevation, slope, Height Above Nearest Drainage (HAND), flow accumulation, and soil clay content — and dynamic forcing variables including antecedent precipitation aggregates (CHIRPS, 3–14 day), soil moisture state (ERA5-Land, daily and 7–30 day means), surface runoff (ERA5-Land, 3–14 day sums), and recent vegetation conditions (MODIS NDVI). Because the SAR dataset's 6–12 day revisit cycle biases detections toward riverine and pluvial events rather than flash floods, the model learns to route slow-onset inundation driven by multi-day rainfall accumulation and pre-event soil saturation through terrain. The result is a stochastic simulator: given a set of conditions at observation time t, the model generates an ensemble of flood extent realizations, capturing both the spatial structure of flooding constrained by topography and the uncertainty inherent in the relationship between atmospheric forcing and surface inundation.

## Dataset origin

Flood masks sourced from Misra et al. (2025), *"Mapping Global Floods with 10 Years of Satellite Radar Data"*
(arXiv:2411.01411). Detection method: Sentinel-1 SAR change detection (pre/post image pairs, VV/VH backscatter).

## Critical temporal constraint

Sentinel-1 revisit time: **6–12 days** (typically 12-day repeat cycle).

The paper explicitly notes: *"Flash floods are challenging because of the short duration. Unless there is an
observation taken during the time of the flood, our model will not capture any flooding."*

**Implication**: the flood dates in the parquet files correspond to SAR overpass dates, not necessarily the
moment flooding began. The actual flood-triggering rainfall likely occurred 1–7 days *before* the detection
date. The dataset is biased toward **riverine/pluvial events** that persist long enough to coincide with a
satellite pass. This makes **medium-term antecedent windows (7–14 days) more important than 1–3 day windows**.

Each parquet row = one acquisition-date detection. Consecutive dates for the same tile may be the same
prolonged event re-observed, or separate events. No event-tracking is performed in the dataset.

---

## Post-processing filter interpretation (from paper)

| Filter | Value used | Paper basis |
|---|---|---|
| `dem_metric_2 < 10` | Max slope within 240 m radius | *"slopes greater than 10° excluded"* |
| `soil_moisture_sca > 1` | SCA soil moisture valid flag | AMSR2 LPRM, ensures valid retrieval |
| `soil_moisture_zscore > 1` | SM anomaly above normal | Above-normal moisture required |
| `soil_moisture > 20` | LPRM SM floor | Low SM = likely dry false positive |
| `temp > 0` | ERA5 daily min temp | *"freeze/thaw cycles... filtered as false positives"* |
| `land_cover != 60` | Bare Ground excluded | ESA WorldCover class 60 |
| `edge_false_positives == 0` | Edge artifacts removed | Tile-boundary processing |

Note: the paper's own post-processing used **AMSR2 LPRM** soil moisture (10 km, daily). The GEE downloads
in this notebook use ERA5-Land (~11 km). These serve the same purpose at similar resolution.

---

## Conditioning variable recommendations

### Static layers (no change recommended)

| Layer | Source | Resolution | Importance |
|---|---|---|---|
| DEM (elevation) | SRTM | 30 m | High — elevation gradient drives routing |
| Slope | Derived from SRTM | 30 m | High — steep = fast runoff, low flood risk |
| HAND | MERIT Hydro `hnd` | ~90 m | **Highest** — vertical distance to nearest drainage; best single predictor |
| Flow accumulation | MERIT Hydro `upa` (log10) | ~90 m | **Very High** — identifies where water concentrates |
| Clay content (4 depths) | OpenLandMap | 250 m | Medium — controls infiltration capacity |

These are well-chosen. No additions or removals recommended.

---

### Dynamic layers: revised window recommendations

#### CHIRPS Precipitation — ~5.5 km (0.05°)

| Export | Window | Recommended | Rationale |
|---|---|---|---|
| Daily snapshot | 1 day | **Keep** | Captures rainfall on detection day itself |
| Rolling sum | 3 days | **Keep — but lower priority** | Less critical than originally thought given 12-day revisit; still captures late-event forcing |
| Rolling sum | 7 days | **Keep — now highest priority** | Most likely to capture the triggering rainfall event for riverine/pluvial floods with 6–12 day revisit |
| Rolling sum | 14 days | **Keep — now very important** | Antecedent pre-saturation; distinguishes a wet spell from an isolated event |
| Rolling sum | **30 days** | **Consider adding** | Seasonal wetness context (Kenya long rains: March–May); distinguishes climatological anomalies |

**Revised priority order**: 7-day > 14-day > 1-day > 3-day. The 7-day sum is now the most important
CHIRPS feature, not the 1-day. For a 12-day revisit sensor, the triggering event typically falls in the
7-day window before detection.

```python
chirps_windows = ((3, "sum"), (7, "sum"), (14, "sum"))           # Current — acceptable
# Consider: ((3, "sum"), (7, "sum"), (14, "sum"), (30, "sum"))   # Add 30-day seasonal context
```

---

#### ERA5-Land Soil Moisture — ~11 km (0.1°)

| Export | Window | Recommended | Rationale |
|---|---|---|---|
| Daily mean | 1 day | **Keep** | State of SM at detection time |
| Rolling mean | 7 days | **Keep** | Is SM trending up (wetting front)? |
| Rolling mean | 14 days | **Keep** | Pre-event baseline saturation |
| Rolling mean | **30 days** | **ADD** | Seasonal/climatological baseline; very important given that this dataset captures slow-onset events where antecedent SM over weeks is a dominant control |

**Key derived feature (no extra GEE export needed):** compute SM delta = `sm_daily - sm_14d_mean` after
downloading. Encodes whether the catchment is anomalously wet at detection time relative to the recent mean.
This directly mirrors the paper's own `soil_moisture_zscore` filter.

```python
era5_sm_windows = ((7, "mean"), (14, "mean"))                          # Current
# Recommended: era5_sm_windows = ((7, "mean"), (14, "mean"), (30, "mean"))
```

---

#### ERA5-Land Temperature — ~11 km (0.1°)

| Export | Window | Recommended | Rationale |
|---|---|---|---|
| Daily mean | 1 day | **Keep** | ET demand on detection day |
| Rolling mean | 3 days | **Drop** | Redundant given daily + 7-day; saves ~30% of temp GEE tasks |
| Rolling mean | 7 days | **Keep** | Best ET trend window; temperature matters for SM recovery between events |
| Rolling mean | 14 days | **Drop** | Minimal additional information for tropical East Africa (no snowmelt) |

The paper uses temperature only as a freeze/thaw filter (temp > 0°C). For Kenya (equatorial), temperatures
rarely approach 0°C, so temperature's utility in the model is primarily as an evapotranspiration proxy.
One rolling window (7-day) alongside the daily mean is sufficient.

```python
era5_temp_windows = ((3, "mean"), (7, "mean"), (14, "mean"))  # Current
# Recommended: era5_temp_windows = ((7, "mean"),)              # Simplified
```

---

#### ERA5-Land Runoff — ~11 km (0.1°)

| Export | Window | Recommended | Rationale |
|---|---|---|---|
| Daily sum | 1 day | **Keep** | Immediate hydrological response |
| Rolling sum | 3 days | **Keep** | Short compound events |
| Rolling sum | 7 days | **Keep — highest priority** | Primary window: cumulative runoff over ~1 Sentinel-1 revisit period |
| Rolling sum | 14 days | **Keep** | Pre-event runoff loading for riverine events |

ERA5 runoff integrates precipitation × soil saturation and is a strong signal for riverine/pluvial events.
The 7-day rolling sum closely mirrors the Sentinel-1 revisit window and is the most informative aggregate.

```python
era5_runoff_windows = ((3, "sum"), (7, "sum"), (14, "sum"))  # Current — keep as-is
```

---

#### MODIS NDVI — 250 m, 16-day composites

| Export | Window | Recommended | Rationale |
|---|---|---|---|
| Most-recent composite | 32-day lookback | **Keep** | Guarantees ≥1 MODIS composite before detection; 32 days covers two 16-day cycles |

No change needed.

```python
modis_ndvi_lookback = 32  # Keep
```

---

## Summary of changes vs. current defaults

| Parameter | Current | Recommended | Impact |
|---|---|---|---|
| `chirps_windows` | `((3,"sum"),(7,"sum"),(14,"sum"))` | Add `(30,"sum")` optional | +1 task/date |
| `era5_sm_windows` | `((7,"mean"),(14,"mean"))` | **Add `(30,"mean")`** | +1 task/date |
| `era5_temp_windows` | `((3,"mean"),(7,"mean"),(14,"mean"))` | **Simplify to `((7,"mean"),)`** | −2 tasks/date |
| `era5_runoff_windows` | `((3,"sum"),(7,"sum"),(14,"sum"))` | No change | — |
| `modis_ndvi_lookback` | `32` | No change | — |

Net change: −2 tasks/date (temp simplification) + 1 (SM 30d) = **−1 task/date** vs. current.
If also adding CHIRPS 30-day: net-neutral vs. current (16 tasks/date).

---

## Feature importance ranking (revised for riverine/pluvial flood type)

1. **HAND** — topographic susceptibility; strongest static predictor
2. **CHIRPS 7-day sum** — most likely window to contain triggering rainfall (primary trigger given 12-day revisit)
3. **CHIRPS 14-day sum** — pre-saturation context for slow-onset events
4. **ERA5 SM daily** — antecedent saturation state at detection time
5. **Log flow accumulation** — drainage network concentration
6. **ERA5 SM 14-day mean** — medium-term soil wetness baseline
7. **ERA5 runoff 7-day sum** — integrated hydrological response over Sentinel-1 revisit period
8. **CHIRPS 3-day sum** — late-event forcing (lower priority than previously thought)
9. **ERA5 SM 30-day mean** (new) — seasonal baseline; distinguishes climatological anomalies
10. **ERA5 runoff daily + 3-day**
11. **DEM + slope** — routing controls
12. **Clay content** — infiltration capacity
13. **CHIRPS 1-day** — day-of-acquisition rainfall (less predictive given detection lag)
14. **NDVI** — canopy interception and root-zone storage
15. **ERA5 temperature** — ET demand proxy (lowest priority in tropical Kenya)

---

## Post-download derived features (no extra GEE exports)

These can be computed cheaply during preprocessing after downloading:

| Feature | Formula | Why |
|---|---|---|
| SM anomaly | `sm_daily - sm_14d_mean` | Mimics the paper's `soil_moisture_zscore`; encodes wetting trajectory |
| Precip intensity ratio | `chirps_1d / (chirps_14d_sum / 14)` | Daily rainfall relative to 2-week mean; flags anomalous events |
| Runoff ratio | `era5_runoff_daily / chirps_1d` | Fraction of rainfall becoming runoff; proxy for saturation excess |

---

## GEE task count estimate (per tile, per date)

With current defaults:
- CHIRPS: 1 daily + 3 aggregates = 4 tasks/date
- ERA5 SM: 1 daily + 2 aggregates = 3 tasks/date
- ERA5 Temp: 1 daily + 3 aggregates = 4 tasks/date
- ERA5 Runoff: 1 daily + 3 aggregates = 4 tasks/date
- MODIS NDVI: 1 task/date
- **Total dynamic: 16 tasks/date**

With recommended changes (drop 2 temp windows, add 1 SM window):
- ERA5 Temp: 1 daily + 1 aggregate = 2 tasks/date
- ERA5 SM: 1 daily + 3 aggregates = 4 tasks/date
- **Total dynamic: 15 tasks/date** (−1 task/date, −6%)

If also adding CHIRPS 30-day: 16 tasks/date (net-neutral vs. current).
