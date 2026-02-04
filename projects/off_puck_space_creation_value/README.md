# Off-Puck Space Creation Value

This project estimates player value from off-puck movement by measuring how movement changes an offensive opportunity signal.

## What it does

- Segments 5v5 offensive-zone possessions from event data.
- Aligns possessions to tracking frames.
- Computes a frame-level opportunity score from:
  - slot openness
  - seam/passing-lane openness
  - release space
- Estimates per-player **Space Creation Added (SCA)** with a leave-one-out freeze:
  - `SCA = Opportunity(all players) - Opportunity(player frozen at prior position)`
- Detects off-puck movement archetype actions:
  - cut
  - drag
  - screen
  - decoy
- Produces player leaderboards, archetypes, and possession-level validation.

## Run

From repository root:

```bash
python projects/off_puck_space_creation_value/run_pipeline.py
```

Fast smoke test:

```bash
python projects/off_puck_space_creation_value/run_pipeline.py --max-games 2 --frame-stride 30
```

## Config

Default config is in `projects/off_puck_space_creation_value/config/default_config.json`.

Key parameters:

- `max_games`: limit games for faster iteration.
- `frame_stride`: sample every Nth tracking frame.
- `max_possession_seconds`: cap possession windows.

## Outputs

Written to `projects/off_puck_space_creation_value/outputs`:

- `offensive_possessions.csv`
- `frame_contributions.csv`
- `player_sca_summary.csv`
- `possession_sca_summary.csv`
- `archetypes/archetype_summary.csv`
- `modeling/outcome_validation.csv`
- `run_summary.json`

## Notes

- Attribution is approximate by design (fast, explainable baseline).
- This is a foundation: you can swap in richer attention or interception models later.

