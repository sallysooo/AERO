# file: src/evaluation/eval_table_iv_all_splits.py
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # .../Modularization/
SAVE_DIR = BASE_DIR / "saved_models"
sys.path.append(str(BASE_DIR / "src"))

import numpy as np
import torch
from tqdm import tqdm
from collections import Counter, defaultdict

from models.modeling import Encoder, PointMapper
from utils.data_utils import get_processed_dataloader, do, args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
window_size, stride = 2048, 1
p = 0.9996

# --- 공격 코드 → 통일된 공격명 매핑 (데이터 버전에 따라 보강)
ATTACK_TO_NAME = {
    # CAN 계열
    "M_D": "CAN DoS",
    "M_R": "CAN replay",
    "M_C": "CAM table overflow",
    # AVTP 주입
    "M_F": "AVTP frame injection",
    "F_I": "AVTP frame injection",
    # PTP sync
    "P_I": "PTP sync attack",
}
ATTACK_NORMAL_TOKENS = {"Normal", ""}

# 1) tau 로드 (validation에서 미리 저장했거나 npy 재활용)
anomaly_scores_val = np.load(SAVE_DIR / "step5_anomaly_scores.npy")
try:
    tau = np.quantile(anomaly_scores_val, p, method="higher")
except TypeError:
    tau = np.quantile(anomaly_scores_val, p)
print(f"[τ] p={p:.4f}  tau={tau:.10e}")

# 2) 모델/기준점
encoder = Encoder(window_size=window_size)
ckpt1 = torch.load(SAVE_DIR / "step1_best_model_encoder.pt", map_location=device)
encoder.load_state_dict(ckpt1["encoder_state_dict"])
encoder.to(device).eval()

point_mapper = PointMapper()
ckpt2 = torch.load(SAVE_DIR / "step4_finetuned_point_mapper.pt", map_location=device)
point_mapper.load_state_dict(ckpt2["point_mapper_state_dict"])
point_mapper.to(device).eval()

a = torch.load(SAVE_DIR / "step3_criterion_point_a.pt", map_location=device).to(device)

# 3) split별 창 단위 공격 코드와 점수/판정 계산
def window_codes_scores_for_split(split_idx):
    """args[split_idx]로 자른 PktDataset에 대해
       - 창 끝 인덱스 배열(Validation/Test 생성과 동일)
       - 해당 창의 '비정상 최빈 y_desc' (없으면 Normal)
       - 각 창의 anomaly score와 예측(>=tau)
       반환
    """
    _, dataset = do(*args[split_idx])  # PktDataset slice
    ydesc = dataset.df["y_desc"].values
    n = len(ydesc)

    # 윈도 개수: FG1 생성 길이에 맞춤(정합성)
    T = dataset.do_fg1_transition_matrix(window_size=window_size, stride=stride)
    num_windows = T.shape[0]

    ends = np.arange(window_size, n + 1, stride)[:num_windows]

    # 창별 y_desc → 공격 코드(비정상 최빈)
    win_codes = []
    for e in ends:
        w = ydesc[e - window_size : e]
        abnormal = w[~np.isin(w, list(ATTACK_NORMAL_TOKENS))]
        if abnormal.size == 0:
            win_codes.append("Normal")
        else:
            win_codes.append(Counter(abnormal).most_common(1)[0][0])

    # 창별 점수/예측
    scores = []
    preds  = []
    # 배치 크기 크게 잡아도 무방
    B = 128
    with torch.no_grad():
        # p 윈도 입력을 DataLoader 없이 직접 순회 (I/O 절약을 위해 배치화)
        # AEGenerator 경로를 재사용해도 되지만 여기서는 간단화를 택함
        P = dataset.do_fg2_payload(window_size=window_size)    # (n, 9)
        S = dataset.do_fg3_statistics(window_size=window_size, stride=stride)  # (num_windows,3,3)
        # T는 위에서 이미 만듦

        # T/S는 창 단위, P는 패킷 단위 → AE 입력 형태 맞추기
        # 창 k의 끝은 ends[k], p-window는 ends[k]-ws : ends[k]
        for i in range(0, num_windows, B):
            j = min(i + B, num_windows)
            # t,s
            t_batch = torch.from_numpy(T[i:j].astype("float32")).flatten(1)  # (b,9)
            s_batch = torch.from_numpy(S[i:j].astype("float32")).flatten(1)  # (b,9)

            # p (b, ws, 9)
            p_list = []
            for e in ends[i:j]:
                pw = P[e - window_size : e]
                if pw.shape[0] < window_size:  # pad (방어적)
                    pad = np.zeros((window_size - pw.shape[0], P.shape[1]), dtype=np.float32)
                    pw = np.vstack([pad, pw])
                p_list.append(pw)
            p_batch = torch.from_numpy(np.stack(p_list).astype("float32"))

            t_batch = t_batch.to(device)
            s_batch = s_batch.to(device)
            p_batch = p_batch.to(device)

            h = encoder((t_batch, p_batch, s_batch))
            m = point_mapper(h)
            sc = ((m - a)**2).sum(dim=1)
            pr = (sc >= tau).int()

            scores.append(sc.cpu().numpy())
            preds.append(pr.cpu().numpy())

    return np.array(win_codes, dtype=object), np.concatenate(scores), np.concatenate(preds)

# 4) 모든 split(0~5) 합산
codes_all, scores_all, preds_all = [], [], []
for split_idx in range(6):
    print(f"[split {split_idx}] processing...")
    c, s, p_ = window_codes_scores_for_split(split_idx)
    codes_all.append(c); scores_all.append(s); preds_all.append(p_)
codes_all  = np.concatenate(codes_all)
scores_all = np.concatenate(scores_all)
preds_all  = np.concatenate(preds_all)

# 5) 공격 유형별 집계 (Normal 제외)
features_by, misses_by = defaultdict(int), defaultdict(int)

for code, yhat in zip(codes_all, preds_all):
    if code in ATTACK_NORMAL_TOKENS:
        continue
    # 통일된 공격명으로 묶기
    name = ATTACK_TO_NAME.get(code, code)
    features_by[name] += 1
    if yhat == 0:
        misses_by[name]  += 1

# 6) 출력
print("\nTABLE IV — PERFORMANCE EVALUATION BY ATTACK TYPE (ALL SPLITS)")
print(f"{'Attack type':<24} {'#of features':>14} {'#of misses':>12} {'FNR':>8}")
for name in sorted(features_by.keys()):
    feat = features_by[name]
    miss = misses_by.get(name, 0)
    fnr  = miss / feat if feat > 0 else 0.0
    print(f"{name:<24} {feat:14,d} {miss:12,d} {fnr:8.4f}")

# 저장
CSV_DIR = BASE_DIR / 'src' / 'evaluation' 
CSV_DIR.mkdir(parents=True, exist_ok=True)
out = CSV_DIR / "table_IV_by_attack.csv"
with out.open("w") as f:
    f.write("attack_type,features,misses,fnr\n")
    for name in sorted(features_by.keys()):
        feat = features_by[name]
        miss = misses_by.get(name, 0)
        fnr  = miss / feat if feat > 0 else 0.0
        f.write(f"{name},{feat},{miss},{fnr:.6f}\n")
print(f"\nSaved: {out}")

