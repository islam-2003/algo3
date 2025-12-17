# INFO-F310 TSP Project (MTZ / DFJ)

## Files
- tsp_solver.py : required CLI solver (f=0..4)
- run_all.py    : batch runner to generate results.csv
- report_template.tex : LaTeX template for the scientific report

## Setup
pip install pulp pandas

## Use
python3 tsp_solver.py instance_10_random_sym_1.txt 4

## Batch results
1) Unzip instances.zip into a folder `instances/`
2) Run:
   python3 run_all.py instances results.csv

## Notes
- DFJ enumerative is exponential: run only for n<=15 (as requested).
- Timing: solver time only (prob.solve()) is measured.
