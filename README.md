# AERO — Reproduction & Modular Implementation
> Automotive Ethernet Real-Time Observer for Anomaly Detection in In-Vehicle Networks

This repository contains a **modular PyTorch reproduction** of the AERO anomaly-detection pipeline for Automotive Ethernet.
It implements the full training/evaluation flow described in the paper (autoencoder pretraining → point‑mapper pretrain → criterion point → fine‑tune → threshold selection → per‑attack evaluation), and provides both the paper-faithful original data pipeline and a tsabilized variant(ver2).

---

## Key Contents
- Modular training scripts for all steps of Algorithm 2.
- Two dataset pipelines:
  - data_utils.py (original): paper-faithful, protocol inference by wirelen
  - data_utils_ver2.py : stabilized variant; can slightly change protocol coverage.
- **Caching** for feature generators (FG1/FG2/FG3) to speed up runs.
- **Evaluation**:
  - percentile sweep to pick tau from validation scores,
  - test metrics,
  - per-attack FNR table (Table IV)
- **Reproducibility**
  - Set a global seed (e.g., 42). Training scripts already expose a seed setter; see each scripts header.
  - Also, if you change the window_size, stride, or the active data_utils.py, **must clear caches** before re-running.
  - In this reproduction... => `Seed:42, window_size=2048, stride=1`
---

## Repository Structure

<details>
  <summary>Modularization structure (Click)</summary>
    
```bash
Modularization/
├── notebooks/
│   └── trial2_Step1-Autoencoder_training.ipynb # the original whole jupyter notebook (before modularization)
├── cache/original or ver2/                     # FG1/2/3 caches per split (auto-created)  
│   ├── train/ 
│   │   ├── T_idx[0]_ws2048_st1.pkl
│   │   └── ...
│   ├── valid/ 
│   │   ├── T_idx[1]_ws2048_st1.pkl
│   │   └── ...
│   ├── test/ 
│   │   ├── T_idx[2]_ws2048_st1.pkl
│   │   └── ...
│   │
├── saved_models/original or ver2/
│   ├── step1_best_model_encoder.pt           # the best model's encoder (epoch1=20)
│   ├── step2_best_model_point_mapper.pt      # the best pointmapper model (epoch2=10)
│   ├── step3_criterion_point_a.pt            # the criterion point a
│   ├── step4_finetuned_point_mapper.pt       # the finetuned pointmapper model (epoch3=150)
│   ├── step5_anomaly_scores.npy              # the anomaly score list
│   ├── step5.2_anomaly_scores_test.npy
│   └── step5.2_labels_test.npy
│
├── src/
│   ├── dataset/           # raw pcaps + y_*.csv 
│   ├── models/            
│   │   └── modeling.py    # Encoder/Decoder/AE/PointMapper   
│   │
│   ├── training/          # Algorithm2 training steps
│   │   ├── step1-autoencoder_training.py       
│   │   ├── step2-pointmapper_training.py       
│   │   ├── step3-determine_criterion_point.py  # step3 : determine criterion point a
│   │   ├── step4-pointmapper_finetune.py       # step4 : fine-tuning pointmapper 
│   │   ├── step5.1-obtain_anomaly_score.py     # step5-1 : obtain list l(anomaly_scores)
│   │   ├── step5.2-threshold_determining.py    # step5-2 : determining threshold w/ visulaization of p
│   │   └── check_epoch.py                      # checking the number of epoch of saved models from each steps
│   │
│   ├── evaluation/
│   │   ├── original/ or ver2/
│   │   │   └── table_IV_by_attack.csv          # evaluation output (table_IV)
│   │   └── eval_by_attack.py
│   └── utils/             # choose: original or ver2       
│       └── data_utils.py  # 0. Seed / 1. TimeSeriesGenerator / # 2. Load Dataset & FG1-3 / # 3. Create train/validation/test sets / # 4. AEGenerator(NEW) / 5. Generate DataLoader for train/validation/test
│   
└── README.md
```

> Note: Make sure to keep one data_utils.py active at a time. See the "Pipelines" section below.

</details>

---

## Descriptions for Each Pipelines (choose one)
### 1) `original` (paper-faithful)
- Protocol assignment uses wirelen buckets (as in the paper).
- Maximally consistent with reported tables/figures.
- Recommended for paper reproduction and per-attack FNR matching.
- However it was difficult to reach high performance as the paper when we use the exact same hyperparameter (epoch1=20, epoch2=10, epoch3=150), so the output performance in this case was derived from hyperparameters selected in the early stopping logic (epoch1=31, epoch2=10, epoch3=10)

> TABLE IV — PERFORMANCE EVALUATION BY ATTACK TYPE (data_utils.py)

| Attack type          | # of features | # of misses |    FNR |
| -------------------- | ------------: | ----------: | -----: |
| CAN DoS              |       267,383 |       2,308 | 0.0086 |
| CAN replay           |       208,669 |       8,874 | 0.0425 |
| CAM table overflow   |       161,105 |       5,323 | 0.0330 |
| AVTP frame injection |       205,689 |       4,988 | 0.0243 |
| PTP sync attack      |       264,811 |         117 | 0.0004 |


### `ver2` (stabilized)
- Engineering tweaks(pooling/initialization etc.) and stricter protocol parsing.
- Often yiels stable training and strong overall metrics, but some non-IP frames (e.g., CAM overflow) may be filtered unless explicitly handled.
- Good for robust deployment experiments; not identical to paper's protocol coverage.
- As we revised the initial data_utils.py code, the output reproduction was much stable in the same hyperparameter environment (epoch1=20, epoch2=10, epoch3=150). But there's a trade-off as below, which can only detect 4 attack types, missing the `CAM table overflow` attack instead.

> TABLE IV — PERFORMANCE EVALUATION BY ATTACK TYPE (data_utils_ver2.py)

| Attack type          | # of features | # of misses |      FNR |
| -------------------- | ------------: | ----------: | -------: |
| CAN DoS              |       266,907 |       2,694 | 0.010093 |
| CAN replay           |       208,171 |      15,641 | 0.075135 |
| AVTP frame injection |       205,224 |       8,289 | 0.040390 |
| PTP sync attack      |       264,282 |         47 | 0.000178 |

---

## Architecture
### End-to-End Usage

```bash
# 1) Step 1: Train Autoencoder
python src/training/step1-autoencoder_training.py

# 2) Step 2: Train PointMapper (pretrain with L_Pre)
python src/training/step2-pointmapper_training.py

# 3) Step 3: Compute criterion point a (mean of M over train)
python src/training/step3-determine_criterion_point.py

# 4) Step 4: Fine‑tune PointMapper toward a
python src/training/step4-pointmapper_finetune.py 

# 5) Step 5.1: Get validation anomaly scores (for τ selection)
python src/training/step5.1-obtain_anomaly_score.py

# 6) Step 5.2: Percentile sweep (p∈[0.9990, 1.0000)) → pick τ
python src/training/step5.2-threshold_determining.py

```

### Feature Generators
- FG1 (T): 3×3 protocol transition matrices (sliding window).
- FG2 (P): 9 payload bytes starting at 0x22 (zero‑padded, normalized).
- FG3 (S): protocol‑wise inter‑arrival statistics (mean/std/|skew|, log‑scaled).
- Data are cached per split in cache/{train,valid,test}/.

### Models
- Autoencoder:
  - T: 9 → 64 (MLP), S: 9 → 64 (MLP), P: separable‑Conv1d stack → 576 (global pooling)
  - Concatenate to latent h ∈ ℝ⁷⁰⁴, reconstruct T/P/S

- PointMapper:
  - MLP 704 → 16
  - Pretrain with `L_Pre = Σ‖m_i − m̄‖²` on train windows;
  - Fine‑tune toward fixed criterion point a = mean(M_train) with `L_M = Σ‖m − a‖²`

### Recommended threshold selection
- We sweep the extreme tail (≥0.9990) on validation scores and pick tau at the start of the F1 plateau (e.g., p≈0.975 in the original run / p≈0.9970 in the ver2 run).

<p align="center">
  <img src="https://github.com/sallysooo/AERO/blob/main/Modularization/saved_models/original_SOTA%20(E31%2C%2010%2C%20E10)/Fig_6_ver2_3.png" alt="Preprocessing Workflow" width="400">
  <img src="https://github.com/sallysooo/AERO/blob/main/Modularization/saved_models/ver2(20%2C%2010%2C%20150)/ver7.png" alt="Preprocessing Workflow" width="400">
</p>
