# Forechecking Report

This folder contains the final written submission package for the Big Data Cup 2026 project.

## Open These First

- `forechecking_pressure_topology_final_draft.pdf`: final submission PDF
- `main.tex`: LaTeX source for the final manuscript

## Folder Contents

- `main.tex`: report source
- `figures/`: generated PNG figures used in the manuscript
- `scripts/make_figures.py`: figure-generation script
- `build_pdf.ps1`: LaTeX build script that creates:
  - `forechecking_pressure_topology_report_v1.pdf`
  - `forechecking_pressure_topology_report_v2.pdf`
  - `forechecking_pressure_topology_final_draft.pdf`

## Build

From the repository root:

```bash
python projects/forechecking_pressure_topology/report_v1/scripts/make_figures.py
powershell -ExecutionPolicy Bypass -File projects/forechecking_pressure_topology/report_v1/build_pdf.ps1
```

## Notes

- The report is synced to the current pipeline outputs in `projects/forechecking_pressure_topology/outputs`.
- Figures in this folder are generated assets intended for the manuscript, not hand-edited image files.
