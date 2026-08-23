# Exchangeability-Aware Conformal Prediction for Multimodal ICU Mortality

Reproducibility code accompanying the MSc dissertation:

> Piyush Trivedi (25030313). *Exchangeability-Aware Conformal Prediction for
> Multimodal ICU Mortality Under Missingness and Repeated Admissions.*
> MSc Artificial Intelligence, Northumbria University, 2025-26.

## Overview

Four pipeline stages, run in order on Kaggle (MIMIC-IV credentialed access
required):

| Stage | Script | Purpose |
|---|---|---|
| 1 | `stage1_feature_extraction.py` | Extracts the cohort and 48-hour hourly feature tensors from raw MIMIC-IV tables (LOS ≥48h, age ≥18). |
| 2 | `stage2_fusion_transformer.py` | Trains the primary multimodal (vitals + labs) dual-stream fusion Transformer. |
| 2b | `stage2b_lstm_baseline.py` | Trains the LSTM baseline with a fixed 60-epoch schedule and best-checkpoint restoration. |
| 3 | `stage3_conformal_prediction.py` | Split conformal prediction (LAC, APS), Mondrian group-conditional calibration, patient-level exchangeability correction, and all coverage/fairness/robustness analyses reported in the dissertation. |

## Reproducing the results

1. Request MIMIC-IV access via [PhysioNet](https://physionet.org/content/mimiciv/)
   and attach the dataset in a Kaggle notebook.
2. Run `stage1_feature_extraction.py` first to build the cohort and feature
   tensors.
3. Run `stage2_fusion_transformer.py` to train the primary model.
4. Run `stage2b_lstm_baseline.py` (after Stage 2, in the same session, so it
   picks up the corrected patient-grouped split indices) to train the LSTM
   baseline.
5. Run `stage3_conformal_prediction.py` last to reproduce all conformal
   prediction, calibration, coverage, and fairness results.

All four scripts use a fixed random seed (`42`) throughout for cohort
sampling, model initialization, and data splitting. Exact split sizes and
patient counts are reported in Section 4 of the dissertation.

## Notes

- Narrative comments have been removed from the versions here for brevity;
  the logic is otherwise exactly as run to produce the results reported in
  the dissertation.
- The MIMIC-IV dataset itself is not redistributed here (credentialed
  PhysioNet access required); the extraction criteria in Stage 1 and the
  dissertation's Methods section are sufficient to reconstruct the cohort.
