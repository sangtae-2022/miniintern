# miniintern
척추 협착증 관련 ai 학습 및 개발

# 데이터 정제
```python
import pandas as pd
import ast

df = pd.read_excel("STEN_labeled_output.xlsx")

df["label_dict"] = df["auto_label"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

label_keys = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1", "need_check"]
df["label_vector"] = df["label_dict"].apply(lambda d: [int(d[k]) for k in label_keys])

df_model = df[["검사결과", "label_vector"]].rename(columns={"검사결과": "text"})

df_model.head()
```

# 데이터 분할
```python

from sklearn.model_selection import train_test_split

train_val, test = train_test_split(df_model, test_size=0.1, random_state=42)

train, val = train_test_split(train_val, test_size=0.1111, random_state=42)  

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

train.to_json("train.jsonl", orient="records", lines=True, force_ascii=False)
val.to_json("val.jsonl", orient="records", lines=True, force_ascii=False)
test.to_json("test.jsonl", orient="records", lines=True, force_ascii=False)
```

# 학습 모델 코드

```python
import os, re, pickle, numpy as np, pandas as pd, warnings, time, ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

warnings.filterwarnings("ignore")
os.environ["PYTHONUNBUFFERED"] = "1"

t0 = time.time()
def log(m): print(m, flush=True)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))

def platt_fit_1d(z, y):
    z = z.reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200, class_weight="balanced")
    lr.fit(z, y)
    A = float(lr.coef_[0, 0]); B = float(lr.intercept_[0])
    return A, B

def platt_apply(Z, A, B):
    return sigmoid(Z * A + B)

def pick_thresholds(y_true: np.ndarray, prob: np.ndarray, default_min_prec: float, beta: float, class_prec_floor: dict | None = None) -> np.ndarray:
    C = y_true.shape[1]
    ths = np.full(C, 0.6, dtype=np.float32)
    grid = np.linspace(0.1, 0.9, 33)
    for c in range(C):
        min_prec = class_prec_floor.get(c, default_min_prec) if class_prec_floor else default_min_prec
        best, bt = -1.0, 0.6
        for t in grid:
            pred = (prob[:, c] >= t).astype(int)
            prec, rec, _, _ = precision_recall_fscore_support(y_true[:, c], pred, average="binary", zero_division=0)
            if prec < min_prec: continue
            b2 = beta * beta
            fbeta = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
            if fbeta > best:
                best, bt = fbeta, t
        ths[c] = bt
    return ths

def retune_class_threshold_by_f1(y_true, prob, ths, c, t_min=0.2, t_max=0.95, steps=40):
    grid = np.linspace(t_min, t_max, steps)
    best_t, best_f = ths[c], -1.0
    for t in grid:
        pred = (prob[:, c] >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true[:, c], pred, average="binary", zero_division=0)
        if f1 > best_f:
            best_f, best_t = f1, t
    ths[c] = best_t
    log(f"  - class {c} threshold retuned → {best_t:.3f} (val F1={best_f:.4f})")
    return ths

df = pd.read_excel("STEN_labeled_output.xlsx")
df["label_dict"] = df["auto_label"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
label_keys = ["L1/2", "L2/3", "L3/4", "L4/5", "L5/S1", "need_check"]
df["label_vector"] = df["label_dict"].apply(lambda d: [int(d[k]) for k in label_keys])
df_model = df[["검사결과", "label_vector"]].rename(columns={"검사결과": "text"})
train_val, test = train_test_split(df_model, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1111, random_state=42)
print("Train:", len(train)); print("Val:", len(val)); print("Test:", len(test))
train.to_json("train.jsonl", orient="records", lines=True, force_ascii=False)
val.to_json("val.jsonl", orient="records", lines=True, force_ascii=False)
test.to_json("test.jsonl", orient="records", lines=True, force_ascii=False)

SAVE_DIR = "./final_model"; os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, "tfidf_ovr_model.pkl")
THS_PATH   = os.path.join(SAVE_DIR, "best_ths.npy")
CAL_A_PATH = os.path.join(SAVE_DIR, "platt_A.npy")
CAL_B_PATH = os.path.join(SAVE_DIR, "platt_B.npy")
LOAD_EXISTING = False
GLOBAL_MIN_PREC = 0.70
FOCUS = {0: 0.80, 4: 0.80}
BETA = 0.5

log("[1/6] Loading data ...")
df_tr = pd.read_json("train.jsonl", lines=True)
df_va = pd.read_json("val.jsonl",   lines=True)
df_te = pd.read_json("test.jsonl",  lines=True)
for d in (df_tr, df_va, df_te):
    d["text"] = d["text"].map(_clean)
Y_tr = np.array(df_tr["label_vector"].tolist(), dtype=int)
Y_va = np.array(df_va["label_vector"].tolist(), dtype=int)
Y_te = np.array(df_te["label_vector"].tolist(), dtype=int)

if not LOAD_EXISTING:
    log("[2/6] Vectorizing (word & char n-grams) ...")
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_df=0.9, max_features=200_000, sublinear_tf=True, norm="l2")
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2,5), min_df=3, max_df=1.0, max_features=200_000, sublinear_tf=True, norm="l2")
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
    base_lr = LogisticRegression(solver="saga", penalty="l2", C=4.0, max_iter=2000, class_weight="balanced", n_jobs=-1, verbose=1)
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
    X_va = None

if not LOAD_EXISTING:
    log("[4/6] Per-class Platt calibration on validation margins ...")
    Z_va = clf.decision_function(X_va)
    Cnum = Z_va.shape[1]
    A = np.zeros(Cnum, dtype=np.float32); B = np.zeros(Cnum, dtype=np.float32)
    for c in range(Cnum):
        a, b = platt_fit_1d(Z_va[:, c], Y_va[:, c]); A[c], B[c] = a, b
    np.save(CAL_A_PATH, A); np.save(CAL_B_PATH, B)
    log(f"    saved Platt params to {CAL_A_PATH}, {CAL_B_PATH}")
    P_va_cal = platt_apply(Z_va, A, B)
    log("[5/6] Picking per-class thresholds with precision floors ...")
    ths = pick_thresholds(Y_va, P_va_cal, default_min_prec=GLOBAL_MIN_PREC, beta=BETA, class_prec_floor=FOCUS)
    for c in FOCUS.keys():
        ths = retune_class_threshold_by_f1(Y_va, P_va_cal, ths, c=c, t_min=0.2, t_max=0.95, steps=40)
    np.save(THS_PATH, ths)
    log(f"    saved thresholds to {THS_PATH}")
else:
    if not (os.path.exists(THS_PATH) and os.path.exists(CAL_A_PATH) and os.path.exists(CAL_B_PATH)):
        raise FileNotFoundError("Artifacts missing. Run once with LOAD_EXISTING=False.")
    ths = np.load(THS_PATH); A = np.load(CAL_A_PATH); B = np.load(CAL_B_PATH)
    log(f"[Loaded] thresholds & Platt params from {SAVE_DIR}")

log("[6/6] Testing ...")
Z_te = clf.decision_function(X_te)
P_te_cal = platt_apply(Z_te, A, B)
preds = (P_te_cal >= ths).astype(int)
print("\n[TEST Result]")
print(classification_report(Y_te, preds, zero_division=0, digits=4))
mi = f1_score(Y_te, preds, average="micro", zero_division=0)
ma = precision_recall_fscore_support(Y_te, preds, average="macro", zero_division=0)[2]
print({"micro_f1": round(mi,4), "macro_f1": round(ma,4)})

log(f"\nArtifacts saved in: {os.path.abspath(SAVE_DIR)}")
for p in (MODEL_PATH, THS_PATH, CAL_A_PATH, CAL_B_PATH):
    log(f" - {p} ({'exists' if os.path.exists(p) else 'missing'})")

print(f"\nDone in {int(time.time()-t0)} sec.", flush=True)
```



# 웹 페이지

```python
import os, pickle, numpy as np
from flask import Flask, request, render_template_string
from scipy.sparse import hstack

SAVE_DIR="./final_model"
MODEL_PATH=os.path.join(SAVE_DIR,"tfidf_ovr_model.pkl")
A_PATH=os.path.join(SAVE_DIR,"platt_A.npy")
B_PATH=os.path.join(SAVE_DIR,"platt_B.npy")
THS_PATH=os.path.join(SAVE_DIR,"best_ths.npy")

def _sigmoid(x):
    x=np.clip(x,-50,50)
    return 1/(1+np.exp(-x))

def _platt_apply(Z,A,B):
    return _sigmoid(Z*A+B)

def _clean(s):
    import re
    return re.sub(r"\s+"," ",str(s or "")).strip()

def _percent_tags(P,class_names):
    P=np.asarray(P,dtype=float)
    s=P.sum()
    if s<=0:
        return [f"{class_names.get(i,str(i))}:0%" for i in range(len(P))]
    raw=(P/s)*100.0
    rounded=np.floor(raw).astype(int)
    diff=int(100-rounded.sum())
    order=np.argsort(raw-rounded)[::-1]
    for i in range(min(abs(diff),len(P))):
        rounded[order[i]]+=1 if diff>0 else -1
    return [f"{class_names.get(i,str(i))}:{rounded[i]}%" for i in range(len(P))]

for p in [MODEL_PATH,A_PATH,B_PATH,THS_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"missing: {p}")

with open(MODEL_PATH,"rb") as f:
    data=pickle.load(f)
word_vec, char_vec, clf = data["word_vec"], data["char_vec"], data["model"]
A=np.load(A_PATH)
B=np.load(B_PATH)
ths=np.load(THS_PATH)
C=len(ths)

CLASS_NAMES={0:"정상",1:"경증 의심",2:"중등도 의심",3:"중증 의심",4:"협착 의심",5:"직접 확인 필요"}
message_map={0:"정상으로 판단됩니다.",1:"경증 의심이 의심됩니다.",2:"중등도 의심이 의심됩니다.",3:"중증 의심이 의심됩니다.",4:"척추관 협착증이 의심됩니다.",5:"직접 확인이 필요하며 추가 증상이 의심됩니다."}
detail_map={0:"현재로선 특별한 이상 소견이 낮습니다.",1:"증상이 지속되면 외래 방문을 고려하세요.",2:"통증·저림이 악화되면 검사 일정을 앞당기는 것이 좋습니다.",3:"일상 활동 제한이 크면 즉시 병원 진료를 받으세요.",4:"보행 시 통증/저림이 심하면 신경외과/정형외과 진료 권고.",5:"세부 판독은 추가 정보가 필요합니다."}

TEMPLATE="""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>spinal stenosis AI</title>
  <style>
    :root{--bg:#eaf4ff;--ink:#222;--muted:#6b7280;--line:#2f2f2f;--panel:#fff;--ring:#9ca3af}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Noto Sans KR,sans-serif}
    header{background:#2C2A27;color:#fff;padding:.9rem 1.2rem;font-weight:700}
    .wrap{padding:1.25rem;max-width:1100px;margin:0 auto}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
    @media (max-width:980px){.grid{grid-template-columns:1fr}}
    .card{background:var(--panel);border:2px solid var(--line);border-radius:14px;padding:1rem}
    .section{border:1px solid var(--ring);border-radius:12px;padding:.9rem;margin-bottom:.9rem}
    .row{display:flex;gap:.75rem;align-items:center;margin-bottom:.6rem}
    .label{width:110px;white-space:nowrap}
    input,select,textarea,button{font:inherit}
    input[type=text],input[type=number],select{flex:1;padding:.55rem .7rem;border:1px solid var(--line);border-radius:10px}
    textarea{width:100%;max-width:100%;padding:.75rem;border:1px solid var(--line);border-radius:10px;resize:vertical;min-height:120px}
    .muted{color:var(--muted);font-size:.9rem}
    .actions{display:flex;gap:.6rem;justify-content:flex-end}
    .btn{padding:.6rem 1rem;border:1px solid var(--line);border-radius:10px;background:#e0edff;cursor:pointer}
    .btn:hover{background:#d4e4ff}
    h5{margin:.1rem 0 1rem 0;font-size:1.05rem}
    .pills{margin:.25rem 0 .5rem 0}
    .pill{display:inline-block;padding:.22rem .55rem;border:1px solid var(--line);border-radius:999px;font-size:.85rem;margin-right:.35rem;margin-bottom:.35rem}
  </style>
</head>
<body>
  <header>spinal stenosis AI</header>
  <div class="wrap">
    <form method="post">
      <div class="grid">
        <div class="card">
          <h5>[환자 인적 사항]</h5>
          <div class="section">
            <div class="row"><div class="label">성명</div><input name="name" type="text" value="{{ request.form.get('name','') }}" placeholder="홍길동"></div>
            <div class="row"><div class="label">나이</div><input name="age" type="number" min="0" value="{{ request.form.get('age','') }}" placeholder="65"></div>
            <div class="row"><div class="label">성별</div>
              <select name="gender">
                <option value="">선택</option>
                <option value="M" {% if request.form.get('gender')=='M' %}selected{% endif %}>남</option>
                <option value="F" {% if request.form.get('gender')=='F' %}selected{% endif %}>여</option>
              </select>
            </div>
          </div>
          <div class="section">
            <div class="row" style="justify-content:space-between">
              <div class="label" style="flex:none">의사 소견/기타사항</div>
              <div class="muted" style="flex:none">텍스트 입력 기반 판독</div>
            </div>
            <textarea name="notes" rows="7" placeholder="예) 보행 시 둔통과 하지 저림, 굴곡 시 증상 악화...">{{ request.form.get('notes','') }}</textarea>
            <div class="actions"><button class="btn" type="submit">판독 실행</button></div>
          </div>
        </div>
        <div class="card">
          <h5>AI 판정 결과</h5>
          {% if result %}
            <div class="pills">
              {% for tag in result.tags %}
                <span class="pill">{{ tag }}</span>
              {% endfor %}
            </div>
            <p style="margin:.6rem 0 0 0">{{ result.summary }}</p>
            <div class="section" style="margin-top:1rem">
              <h5 style="margin:0 0 .5rem 0">주의사항</h5>
              <p style="margin:0">{{ result.name }}님, {{ result.detail_text }}</p>
            </div>
          {% else %}
            <p class="muted">좌측에 소견을 입력하고 <b>판독 실행</b>을 누르면 결과가 표시됩니다.</p>
          {% endif %}
        </div>
      </div>
    </form>
  </div>
</body>
</html>
"""

app=Flask(__name__)

def predict_from_text(text:str):
    txt=_clean(text)
    X=hstack([word_vec.transform([txt]),char_vec.transform([txt])]).tocsr()
    Z=clf.decision_function(X)
    P=_platt_apply(Z,A,B)
    P=np.asarray(P,dtype=float)[0]
    pred=(P>=ths).astype(int)
    return P,pred

@app.route("/",methods=["GET","POST"])
def index():
    result=None
    if request.method=="POST":
        notes=request.form.get("notes","")
        P,pred=predict_from_text(notes)
        top_idx=int(np.argmax(P))
        tags=_percent_tags(P,CLASS_NAMES)
        summary=message_map.get(top_idx,"판정 결과를 확인했습니다.")
        code=top_idx
        detail_text=detail_map.get(code,"추가 정보 입력 시 더 정확한 안내가 가능합니다.")
        result={
            "name":_clean(request.form.get("name","환자")),
            "tags":tags,
            "summary":summary,
            "detail_text":detail_text,
            "code":code,
        }
    return render_template_string(TEMPLATE,result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=False,use_reloader=False)
```
