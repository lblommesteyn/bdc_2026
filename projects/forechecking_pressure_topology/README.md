# Forechecking Pressure Topology

This directory contains the full Big Data Cup 2026 project: pipeline code, generated outputs, and the final written report.

## Quick Links

- [Final report PDF](report_v1/forechecking_pressure_topology_final_draft.pdf)
- [Manuscript source](report_v1/main.tex)
- [Report-specific README](report_v1/README.md)

## Project Summary

The project models defensive-zone retrieval denial as a spatial, time-dependent pressure problem rather than a simple count of nearby defenders. The core idea is that forechecking success depends on:

- how much pressure reaches the puck carrier
- which exit lane is cheapest at retrieval start
- how quickly that best lane closes over the next three seconds
- whether F2 and F3 arrive in time to remove secondary outlets

This produces two outputs at once:

- an interpretable coaching lens for clip review and system diagnosis
- a predictive feature set for defensive-zone turnover modeling

## Pipeline

The end-to-end pipeline does the following:

1. Segments 5v5 defensive-zone retrieval episodes from event data.
2. Aligns each episode to tracking frames.
3. Computes frame-level pressure and corridor metrics around the likely puck carrier.
4. Builds role-aware support features for the first, second, and third forecheckers.
5. Aggregates those signals into episode-level signatures.
6. Runs clustering and supervised validation against baseline models.

## Key Features

- pressure intensity and gradient at the puck carrier
- ETA-weighted directional pressure
- escape corridor cost for middle, strong-side boards, weak-side boards, and behind-net exits
- 3-second closure of the cheapest corridor
- compactness, local pressure, and D-pinch support
- F1/F2/F3 distance, contribution, and share-of-pressure features
- score-state context at episode start

## Headline Outputs

The current run processes:

- 10 public games
- 2,086 retrieval episodes
- 14,921 sampled tracking frames

Primary artifacts are written to `projects/forechecking_pressure_topology/outputs`:

- `forecheck_episodes.csv`
- `forecheck_frame_metrics.csv`
- `forecheck_episode_features.csv`
- `clustering/cluster_summary.csv`
- `modeling/predictive_validation.csv`
- `run_summary.json`

## Run

From the repository root:

```bash
python projects/forechecking_pressure_topology/run_pipeline.py
```

Fast smoke test:

```bash
python projects/forechecking_pressure_topology/run_pipeline.py --max-games 2 --frame-stride 30
```

## Config

Default config lives in `projects/forechecking_pressure_topology/config/default_config.json`.

Most useful parameters:

- `max_games`: limit games for quick iteration
- `frame_stride`: sample every Nth tracking frame
- `max_episode_seconds`: cap episode length
- `n_clusters`: number of archetypes for clustering

## Notes

- Episode segmentation is rule-based and intentionally auditable.
- Pressure is modeled as an ETA-weighted directional kernel, chosen for interpretability.
- The pipeline is modular enough to swap in richer pressure or transition-value models later.
