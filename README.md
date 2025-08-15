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





