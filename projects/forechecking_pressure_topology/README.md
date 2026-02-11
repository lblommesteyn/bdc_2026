# Forechecking Pressure Topology

This project turns forechecking pressure into a continuous spatial signal and builds episode-level features for clustering and outcome prediction.

## What it does

- Segments 5v5 forecheck episodes from event data.
- Aligns episodes to tracking frames.
- Computes frame-level pressure metrics around the puck carrier:
  - pressure intensity
  - pressure gradient
  - acceleration-constrained anisotropic reachability
  - escape corridor cost (middle, strong boards, weak boards, behind net)
  - compactness / local pressure / pinch indicators
- Adds role-aware support features:
  - inferred `F1/F2/F3` pressure and distance contributions
  - defense pinch support counts
- Adds score-state context at episode start:
  - score differential from forechecking perspective
  - score-state bucket (leading/tied/trailing)
- Aggregates to episode-level signatures.
- Runs:
  - archetype clustering (`k-means`)
  - predictive validation for D-zone turnovers vs baselines and score-state-adjusted variants

## Run

From repository root:

```bash
python projects/forechecking_pressure_topology/run_pipeline.py
```

Fast smoke test:

```bash
python projects/forechecking_pressure_topology/run_pipeline.py --max-games 2 --frame-stride 30
```

## Config

Default config is in `projects/forechecking_pressure_topology/config/default_config.json`.

Key parameters:

- `max_games`: limit games for quick iteration.
- `frame_stride`: sample every Nth tracking frame.
- `max_episode_seconds`: cap episode length.

## Outputs

Written to `projects/forechecking_pressure_topology/outputs`:

- `forecheck_episodes.csv`
- `forecheck_frame_metrics.csv`
- `forecheck_episode_features.csv`
- `clustering/cluster_summary.csv`
- `modeling/predictive_validation.csv`
- `run_summary.json`

## Notes

- Episode segmentation is rule-based and intentionally transparent.
- Pressure is currently ETA-weighted kernel pressure (interpretable baseline).
- The pipeline is designed so you can swap in richer pressure models later.
