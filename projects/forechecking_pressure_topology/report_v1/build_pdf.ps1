$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
Copy-Item -Force main.pdf forechecking_pressure_topology_report_v1.pdf
Copy-Item -Force main.pdf forechecking_pressure_topology_report_v2.pdf
Copy-Item -Force main.pdf forechecking_pressure_topology_final_draft.pdf
Write-Host "Built forechecking_pressure_topology_report_v1.pdf"
Write-Host "Built forechecking_pressure_topology_report_v2.pdf"
Write-Host "Built forechecking_pressure_topology_final_draft.pdf"
