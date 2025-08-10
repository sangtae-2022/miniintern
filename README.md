# miniintern

척추 협착증 관련 ai 학습 및 개발


## 시작

import pandas as pd
import ast

# 엑셀에서 라벨/텍스트 로딩
df = pd.read_excel("STEN_labeled_output.xlsx")

# 문자열 dict → 실제 dict
df["label_dict"] = df["auto_label"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 멀티라벨 벡터 생성
label_keys = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1", "need_check"]
df["label_vector"] = df["label_dict"].apply(lambda d: [int(d[k]) for k in label_keys])

# 학습에 쓰는 컬럼
df_model = df[["검사결과", "label_vector"]].rename(columns={"검사결과": "text"})

# train/val/test 분할 및 저장
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(df_model, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 = 0.1

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

train.to_json("train.jsonl", orient="records", lines=True, force_ascii=False)
val.to_json("val.jsonl", orient="records", lines=True, force_ascii=False)
test.to_json("test.jsonl", orient="records", lines=True, force_ascii=False)

import os, re, pickle, numpy as np, pandas as pd, warnings, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from scipy.sparse import hstack

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"  # 로그 즉시 출력

t0 = time.time()
def log(m): print(m, flush=True)

# =========================
# Config
# =========================
SAVE_DIR = "./final_model"; os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "tfidf_ovr_model.pkl")
THS_PATH   = os.path.join(SAVE_DIR, "best_ths.npy")
CAL_A_PATH = os.path.join(SAVE_DIR, "platt_A.npy")   # per-class A
CAL_B_PATH = os.path.join(SAVE_DIR, "platt_B.npy")   # per-class B

LOAD_EXISTING = False   # True면 저장된 아티팩트 로드 후 바로 추론/테스트

# 임계값 규칙
GLOBAL_MIN_PREC = 0.70  # 기본 클래스 정밀도 하한
FOCUS = {0: 0.80, 4: 0.80}  # ★ 클래스 0/4는 precision ≥ 0.80로 강화
BETA   = 0.5            # F_beta (precision 쪽 가중↑)

# =========================
# Utils
# =========================
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def platt_fit_1d(z, y):
    """
    z: (n,) 마진(decision_function)
    y: (n,) {0,1}
    1D 로지스틱 회귀로 p = sigmoid(A*z + B)를 학습.
    """
    z = z.reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200, class_weight="balanced")
    lr.fit(z, y)
    # lr.coef_.shape = (1,1), lr.intercept_.shape=(1,)
    A = float(lr.coef_[0, 0]); B = float(lr.intercept_[0])
    return A, B

def platt_apply(Z, A, B):
    # Z: (n,C), A/B: (C,)
    return sigmoid(Z * A + B)

def pick_thresholds(y_true: np.ndarray, prob: np.ndarray,
                    default_min_prec: float, beta: float,
                    class_prec_floor: dict | None = None) -> np.ndarray:
    """
    클래스별 임계값 선택:
      - 기본 min_prec = default_min_prec
      - 특정 클래스는 class_prec_floor[class]로 정밀도 하한 강화
      - F_beta 최대화 (beta<1이면 precision에 더 가중)
    """
    C = y_true.shape[1]
    ths = np.full(C, 0.6, dtype=np.float32)  # 못 찾으면 보수적으로 0.6
    grid = np.linspace(0.1, 0.9, 33)
    for c in range(C):
        min_prec = class_prec_floor.get(c, default_min_prec) if class_prec_floor else default_min_prec
        best, bt = -1.0, 0.6
        for t in grid:
            pred = (prob[:, c] >= t).astype(int)
            prec, rec, _, _ = precision_recall_fscore_support(
                y_true[:, c], pred, average="binary", zero_division=0
            )
            if prec < min_prec:
                continue
            b2 = beta * beta
            fbeta = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
            if fbeta > best:
                best, bt = fbeta, t
        ths[c] = bt
    return ths

def retune_class_threshold_by_f1(y_true, prob, ths, c, t_min=0.2, t_max=0.95, steps=40):
    """특정 클래스 c에 대해 F1 최대화 임계값 재탐색(정밀도 하한 없이 F1만)."""
    grid = np.linspace(t_min, t_max, steps)
    best_t, best_f = ths[c], -1.0
    for t in grid:
        pred = (prob[:, c] >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[:, c], pred, average="binary", zero_division=0
        )
        if f1 > best_f:
            best_f, best_t = f1, t
    ths[c] = best_t
    log(f"  - class {c} threshold retuned → {best_t:.3f} (val F1={best_f:.4f})")
    return ths

# =========================
# Load Data
# =========================
log("[1/6] Loading data ...")
df_tr = pd.read_json("train.jsonl", lines=True)
df_va = pd.read_json("val.jsonl",   lines=True)
df_te = pd.read_json("test.jsonl",  lines=True)
for d in (df_tr, df_va, df_te):
    d["text"] = d["text"].map(_clean)

Y_tr = np.array(df_tr["label_vector"].tolist(), dtype=int)
Y_va = np.array(df_va["label_vector"].tolist(), dtype=int)
Y_te = np.array(df_te["label_vector"].tolist(), dtype=int)

# =========================
# Train or Load
# =========================
if not LOAD_EXISTING:
    log("[2/6] Vectorizing (word & char n-grams) ...")
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                               min_df=3, max_df=0.9, max_features=200_000,
                               sublinear_tf=True, norm="l2")
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2,5),
                               min_df=3, max_df=1.0, max_features=200_000,
                               sublinear_tf=True, norm="l2")

    Xtr_w = word_vec.fit_transform(df_tr["text"])
    Xva_w = word_vec.transform(df_va["text"])
    Xte_w = word_vec.transform(df_te["text"])

    Xtr_c = char_vec.fit_transform(df_tr["text"])
    Xva_c = char_vec.transform(df_va["text"])
    Xte_c = char_vec.transform(df_te["text"])

    X_tr = hstack([Xtr_w, Xtr_c]).tocsr()
    X_va = hstack([Xva_w, Xva_c]).tocsr()
    X_te = hstack([Xte_w, Xte_c]).tocsr()
    log(f"    done. shapes: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")

    log("[3/6] Training OvR(LogReg, saga) ...")
    base_lr = LogisticRegression(
        solver="saga", penalty="l2", C=4.0, max_iter=2000,
        class_weight="balanced", n_jobs=-1, verbose=1
    )
    clf = OneVsRestClassifier(base_lr, n_jobs=None)
    clf.fit(X_tr, Y_tr)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"word_vec": word_vec, "char_vec": char_vec, "model": clf}, f)
    log(f"[Saved model] {MODEL_PATH}")

else:
    log("[2/6] Loading saved model/vectorizers ...")
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    word_vec, char_vec, clf = data["word_vec"], data["char_vec"], data["model"]

    Xte_w = word_vec.transform(df_te["text"])
    Xte_c = char_vec.transform(df_te["text"])
    X_te = hstack([Xte_w, Xte_c]).tocsr()

# =========================
# Per-class Platt calibration + Thresholds (0/4 강화) + (옵션) 0/4 재튜닝
# =========================
if not LOAD_EXISTING:
    log("[4/6] Per-class Platt calibration on validation margins ...")
    # 마진(결정함수) 추출
    Z_va = clf.decision_function(X_va)  # (n_val, C)
    Cnum = Z_va.shape[1]
    A = np.zeros(Cnum, dtype=np.float32)
    B = np.zeros(Cnum, dtype=np.float32)
    for c in range(Cnum):
        a, b = platt_fit_1d(Z_va[:, c], Y_va[:, c])
        A[c], B[c] = a, b
    np.save(CAL_A_PATH, A); np.save(CAL_B_PATH, B)
    log(f"    saved Platt params to {CAL_A_PATH}, {CAL_B_PATH}")

    # 보정 확률
    P_va_cal = platt_apply(Z_va, A, B)

    log("[5/6] Picking per-class thresholds with precision floors ...")
    ths = pick_thresholds(
        Y_va, P_va_cal,
        default_min_prec=GLOBAL_MIN_PREC,
        beta=BETA,
        class_prec_floor=FOCUS   # ★ 0/4 클래스 강화
    )

    # (선택) 0/4 F1만 더 올리려는 재튜닝
    for c in FOCUS.keys():
        ths = retune_class_threshold_by_f1(Y_va, P_va_cal, ths, c=c, t_min=0.2, t_max=0.95, steps=40)

    np.save(THS_PATH, ths)
    log(f"    saved thresholds to {THS_PATH}")
else:
    # 로드
    if not (os.path.exists(THS_PATH) and os.path.exists(CAL_A_PATH) and os.path.exists(CAL_B_PATH)):
        raise FileNotFoundError("Artifacts missing. Run once with LOAD_EXISTING=False.")
    ths = np.load(THS_PATH)
    A = np.load(CAL_A_PATH); B = np.load(CAL_B_PATH)
    log(f"[Loaded] thresholds & Platt params from {SAVE_DIR}")

# =========================
# Test
# =========================
log("[6/6] Testing ...")
Z_te = clf.decision_function(X_te)
P_te_cal = platt_apply(Z_te, A, B)
preds = (P_te_cal >= ths).astype(int)

print("\n[TEST Result]")
print(classification_report(Y_te, preds, zero_division=0, digits=4))
mi = f1_score(Y_te, preds, average="micro", zero_division=0)
ma = precision_recall_fscore_support(Y_te, preds, average="macro", zero_division=0)[2]
print({"micro_f1": round(mi,4), "macro_f1": round(ma,4)})

# =========================
# Summary
# =========================
log(f"\nArtifacts saved in: {os.path.abspath(SAVE_DIR)}")
for p in (MODEL_PATH, THS_PATH, CAL_A_PATH, CAL_B_PATH):
    log(f" - {p} ({'exists' if os.path.exists(p) else 'missing'})")

print(f"\nDone in {int(time.time()-t0)} sec.", flush=True)
