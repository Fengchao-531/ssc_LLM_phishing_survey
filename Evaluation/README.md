# Evaluation

This public evaluation folder contains a single reproducible interface for the detector stage-transfer trend analysis.

Benchmark detector outputs are not committed. To reproduce the table, place local detector result CSVs anywhere on your machine, describe them in a manifest, then run:

```bash
python Evaluation/compute_stage_transfer_trend.py \
  --manifest Evaluation/examples/stage_transfer_manifest.csv \
  --output-dir Evaluation/stage-transfer-trend
```

## Manifest Format

The manifest is a CSV with these columns:

```csv
dataset_family,detector_family,stage,path,detector,prediction_column,label_column
```

- `dataset_family`: `hw` or `llm`.
- `detector_family`: `academic` or `industry`.
- `stage`: stage label such as `S1`, `S2`, `S6-fuzzer`, or `S8-llama`.
- `path`: local detector result CSV path, relative to the manifest directory unless absolute.
- `detector`: detector name. Use this for academic outputs that store predictions in `model_prediction`.
- `prediction_column`: optional explicit prediction column. If blank, all `*_prediction` columns are used.
- `label_column`: optional label column; defaults to `label`.

## Outputs

The script writes:

- `per_stage_metrics.csv`: detector metrics by dataset family, detector family, stage, and detector.
- `detector_hw_baselines.csv`: each detector's HW overall baseline.
- `stage_transfer_trend_long.csv`: long-form stage-transfer trend values.
- `table_2_stage_transfer_trend.csv`: wide table used by the README.
- `README.md`: rendered stage-transfer trend table.

The public example uses toy data only. Use your local detector outputs for the real benchmark.
