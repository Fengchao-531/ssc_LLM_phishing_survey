## S8 Model-driven Automation

This stage mixes small reference inputs with large project-generated artifacts.
To keep the repository lightweight enough for normal GitHub browsing and cloning,
the largest generated outputs are stored as zip archives under [`archives/`](archives).

### Kept directly in Git

- `HW-B.csv`, `HW-P.csv`, `HW-combined-shuffled.csv`
- `candidate-prompts.csv`
- generation, evaluation, and post-processing scripts

### Archived in zip form

- `archives/S8-runs.zip`
  Contains the full benchmark run logs, chunk manifests, and intermediate outputs.
- `archives/S8-models-output.zip`
  Contains model-generated phishing outputs and fill-bracket post-processing outputs.
- `archives/S8-evaluation-results.zip`
  Contains subject/body scoring outputs and evaluation summaries.

### Restore commands

Run these from `Datasets/sublist/S8-Model-driven Automation/` after cloning:

```bash
unzip archives/S8-runs.zip
unzip archives/S8-models-output.zip
unzip archives/S8-evaluation-results.zip
```

After extraction, the restored directories are:

- `runs/`
- `Models-Output/`
- `Evaluation results/`

These archived artifacts are project-generated outputs rather than upstream public
benchmarks. The benchmark entry point is
`benchmark_llm_generation_against_reference.py`.
