$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
Copy-Item -Force main.pdf off_puck_space_creation_value_report_v1.pdf
Write-Host "Built off_puck_space_creation_value_report_v1.pdf"

