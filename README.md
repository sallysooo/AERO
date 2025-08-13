# AERO
AERO : Automotive Ethernet Real-Time Observer for Anomaly Detection in In-Vehicle Networks

```
Modularization/
├── notebooks/
│   └── trial2_Step1-Autoencoder_training.ipynb # the original whole jupyter notebook (before modularization)
├── cache/                    # caches of T, P, S calculation results for each train/validation/test dataset 
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
├── saved_models/
│   ├── step1_autoencoder_best_model.pt          # the best entire model from step1 
│   ├── step1_best_model_encoder_42.pt           # the best model's encoder from step1
│   ├── step2_best_model_point_mapper_42.pt      # the best pointmapper model from step2
│   ├── step3_criterion_point_a_42.pt            # the criterion point a
│   ├── step4_finetuned_point_mapper_42.pt       # the finetuned pointmapper model (early stopping applied)
│   ├── step4_finetuned_point_mapper_ver2_42.pt  # early stopping removed, epoch3 = 150 ver.
│   └── step5_anomaly_scores_42.npy              # the anomaly score list from step5
│
├── src/
│   ├── dataset/           # dataset folder (raw)
│   ├── models/            # model definition
│   │   └── modeling.py    # encoder, decoder, SeparableConv1d, SeparableConvTranspose1d, autoencoder class    
│   │
│   ├── training/          # Algorithm2 training steps
│   │   ├── step1-autoencoder_training.py       # step1 : training
│   │   ├── step1-evaluate_metrics.py           # step1 : evaluation
│   │   ├── step2-pointmapper_training.py       # step2 : training
│   │   ├── step3-determine_criterion_point.py  # step3 : determine criterion point a
│   │   ├── step4-pointmapper_finetune.py       # step4 : fine-tuning pointmapper (early stopping applied)
│   │   ├── step4-pointmapper_finetune_ver2.py  # step4 : early stopping removed, epoch3 = 150 ver.
│   │   ├── step5.1-obtain_anomaly_score.py     # step5-1 : obtain list l(anomaly_scores)
│   │   ├── step5.2-threshold_determining.py    # step5-2 : determining threshold w/ visulaization of p
│   │   └── test.py        # checking the best model's result of step1
│   │
│   ├── evaluation/        # (yet)
│   └── utils/             
│       └── data_utils.py  # 0. Seed / 1. TimeSeriesGenerator / # 2. Load Dataset & FG1-3 / # 3. Create train/validation/test sets / # 4. AEGenerator(NEW) / 5. Generate DataLoader for train/validation/test
│   
├── tests/                 # (yet)
└── README.md
```
