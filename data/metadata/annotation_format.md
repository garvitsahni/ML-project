# Congestion annotation format (optional — Experiment E gate)

Experiment E (congestion classification) is **disabled** until credible labels
exist. The dashboard and `reports/results/exp_e_status.json` report
`not_supported` until a file matching this format is present.

## Where to put it

```
data/metadata/congestion_labels.csv
```

## Format

One row per labeled window, aligned with the feature table's windows:

| column | type | meaning |
|--------|------|---------|
| `sequence_id` | str | matches `ucsd_windows.parquet` (e.g. `ped1_Test003`) |
| `window_start` | int | 0-based first frame of the window |
| `window_end` | int | exclusive end frame |
| `congestion_level` | int {0,1,2} | 0 = free flow, 1 = dense, 2 = congested |

## Rules

- Labels must be **independent of the features being evaluated** (annotators
  must not see model outputs).
- Every `(sequence_id, window_start)` pair must exist in the feature table;
  extras are ignored with a warning, gaps cause a hard error.
- Inter-annotator agreement (e.g. Cohen's κ) should be reported before E runs.

Until such a file exists and passes these checks, no congestion metric will be
computed or displayed anywhere in this project.
