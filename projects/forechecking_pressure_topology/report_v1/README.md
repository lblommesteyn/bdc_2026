# Forechecking Report (Final Draft)

## Contents

- `main.tex`: full final manuscript synced to current pipeline outputs.
- `figures/`: generated PNG figures used in the report.
- `scripts/make_figures.py`: recreates report figures from pipeline outputs (uses `mplhockey` for rink panels when available).
- `build_pdf.ps1`: compiles LaTeX and creates:
  - `forechecking_pressure_topology_report_v1.pdf`
  - `forechecking_pressure_topology_report_v2.pdf`
  - `forechecking_pressure_topology_final_draft.pdf`

## Rebuild

From repository root:

```bash
python projects/forechecking_pressure_topology/report_v1/scripts/make_figures.py
powershell -ExecutionPolicy Bypass -File projects/forechecking_pressure_topology/report_v1/build_pdf.ps1
```
