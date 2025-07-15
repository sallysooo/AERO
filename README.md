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
│   └── step1_autoencoder_best_model.pt # the best model during step1 training
├── src/
│   ├── data/              # 데이터 로딩/전처리 코드 (yet)
│   ├── dataset/           # dataset folder (raw)
│   ├── models/            # 모델 클래스 definition 모음
│   │   └── modeling.py    # encoder, decoder, SeparableConv1d, SeparableConvTranspose1d, autoencoder 클래스 정의    
│   ├── training/          # Algorithm2 training steps
│   │   ├── step1-autoencoder_training.py   # step1
│   │   ├── step1-evaluate_metrics.py
│   │   └── test.py        # checking the best model's result of step1
│   ├── evaluation/        # 평가/추론 코드 모음 (yet)
│   └── utils/             # 공통 함수, 설정 등
│       └── data_utils.py  # 1. TimeSeriesGenerator / # 2. Load Dataset & FG1-3 / # 3. Create train/validation/test sets / # 4. AEGenerator(NEW) / 5. Generate DataLoader for train/validation/test
│   
├── tests/                 # (yet)
├── algorithm2.py          # (yet)
└── README.md
```
