# Forechecking Report v1

## Contents

- `main.tex`: 6-page rough draft report.
- `figures/`: generated PNG figures used in the report.
- `scripts/make_figures.py`: recreates report figures from pipeline outputs.
- `build_pdf.ps1`: compiles LaTeX and creates `forechecking_pressure_topology_report_v1.pdf`.

## Rebuild

From repository root:

```bash
python projects/forechecking_pressure_topology/report_v1/scripts/make_figures.py
powershell -ExecutionPolicy Bypass -File projects/forechecking_pressure_topology/report_v1/build_pdf.ps1
```

