# CrowdSense AI — Machine Learning-Based Crowd Analysis and Congestion Risk Assessment

> **Status:** under active construction. See `docs/superpowers/specs/` for the approved design.

Research question: *Does incorporating temporal crowd-movement features improve
congestion or anomaly prediction compared with using static crowd-density features alone?*

## Quick start (PowerShell)

```powershell
# 1. Dependencies (Python 3.13 required — verified environment)
python -m pip install -r requirements.txt

# 2. Run tests
python -m pytest tests/ -v

# 3. Launch the dashboard
python -m streamlit run app.py
```

## Datasets

This project uses the **UCSD Pedestrian Anomaly Detection Dataset** (placed at
`data/raw/ucsd_anomaly/UCSD_Anomaly_Dataset.v1p2/`). Counting datasets
(UCF-QNRF, ShanghaiTech) were excluded by decision — crowd counting therefore runs
as a *detection-based estimate without ground-truth comparison* (see Limitations).

## Documentation

- Full README (dataset layouts, runbook, methodology, limitations): written last, after implementation.
