# Off-Puck Report v1

## Contents

- `main.tex`: 6-page rough draft report.
- `figures/`: generated PNG figures used in the report.
- `scripts/make_figures.py`: recreates report figures from pipeline outputs.
- `build_pdf.ps1`: compiles LaTeX and creates `off_puck_space_creation_value_report_v1.pdf`.

## Rebuild

From repository root:

```bash
python projects/off_puck_space_creation_value/report_v1/scripts/make_figures.py
powershell -ExecutionPolicy Bypass -File projects/off_puck_space_creation_value/report_v1/build_pdf.ps1
```

