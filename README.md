# miniintern
척추 협착증 관련 ai 학습 및 개발

# 데이터 정제
```python
import os
import pandas as pd
import matplotlib.pyplot as plt

RAW_CANDIDATES = ["labels_from_report.csv", "labels_from_report.xlsx"]
REFINED_PATH   = "refined_labels.csv"

def find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def to_bool(s, default=True):
    if not isinstance(s, (pd.Series, pd.Index)):
        s = pd.Series(s)
    s = s.replace({"True": True, "False": False, 1: True, 0: False})
    return s.astype(object).fillna(default).astype(bool)

def main():
    raw_path = find_first(RAW_CANDIDATES)
    if raw_path is None:
        raise FileNotFoundError("labels_from_report.csv(.xlsx) 파일을 같은 폴더에 두세요.")
    if not os.path.exists(REFINED_PATH):
        raise FileNotFoundError("refined_labels.csv 파일이 없습니다.")

    raw = pd.read_excel(raw_path) if raw_path.lower().endswith(".xlsx") else pd.read_csv(raw_path)
    ref = pd.read_csv(REFINED_PATH)

    if "row_idx" not in raw.columns:
        raw = raw.reset_index(drop=True).reset_index().rename(columns={"index":"row_idx"})
    if "row_idx" not in ref.columns:
        ref = ref.reset_index(drop=True).reset_index().rename(columns={"index":"row_idx"})

    if "need_check" not in raw.columns:
        raw["need_check"] = True
    raw["need_check"] = to_bool(raw["need_check"])

    if "need_check_before" not in ref.columns:
        ref = ref.merge(
            raw[["row_idx", "need_check"]].rename(columns={"need_check":"need_check_before"}),
            on="row_idx", how="left"
        )
    ref["need_check_before"] = to_bool(ref["need_check_before"])
    ref["need_check"] = to_bool(ref["need_check"]) if "need_check" in ref.columns else ref["need_check_before"].copy()

    before = ref["need_check_before"]
    after_raw = ref["need_check"]
    after_fixed = before & after_raw
    ref["need_check_fixed"] = after_fixed

    n = len(ref)
    before_true = int(before.sum())
    after_fix_true = int(after_fixed.sum())
    after_fix_false = n - after_fix_true

    plt.figure()
    plt.bar(["Before(True)", "After_fixed(True)"], [before_true, after_fix_true])
    plt.title("Need check true before after_fixed")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.pie([after_fix_true, after_fix_false],
            labels=["True", "False"], autopct="%.1f%%", startangle=90)
    plt.title("True vs False")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
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
import os, json, re, time, pickle
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    f1_score,
)

ARTI_DIR   = "artifacts"          
CONFIRM_CSV= os.path.join(ARTI_DIR, "confirmed_dataset.csv")
META_JSON  = os.path.join(ARTI_DIR, "meta.json")
SAVE_DIR   = "final_model"; os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "tfidf_ovr_model.pkl")
A_PATH     = os.path.join(SAVE_DIR, "platt_A.npy")
B_PATH     = os.path.join(SAVE_DIR, "platt_B.npy")
THS_PATH   = os.path.join(SAVE_DIR, "best_ths.npy")

SEED = 42
MAXF_WORD = 120_000
MAXF_CHAR = 80_000
MAX_ITER  = 800
C_PARAM   = 3.0

DEFAULT_PREC_FLOOR = 0.70           
CLASS_PREC_FLOOR   = {}             
BETA = 0.5                         


def _clean(s): return re.sub(r"\s+", " ", str(s or "")).strip()
def _sigmoid(x): x=np.clip(x,-50,50); return 1/(1+np.exp(-x))

def _platt_fit_1d(z, y):
    """z: (n,) decision margin, y: {0,1}"""
    z = z.reshape(-1,1)
    lr = LogisticRegression(solver="lbfgs", C=1.0, max_iter=200, class_weight="balanced")
    lr.fit(z, y)
    return float(lr.coef_[0,0]), float(lr.intercept_[0])

def _platt_apply(Z, A, B):
  
    return _sigmoid(Z*A + B)

def _pick_thresholds(y_true, prob, default_min_prec, beta=1.0, class_prec_floor=None):
    C = y_true.shape[1]
    ths = np.full(C, 0.5, dtype=np.float32)
    grid = np.linspace(0.1, 0.95, 57)
    for c in range(C):
        floor = (class_prec_floor or {}).get(c, default_min_prec)
        best, best_t = -1.0, 0.5
        for t in grid:
            pred = (prob[:,c] >= t).astype(int)
            prec, rec, _, _ = precision_recall_fscore_support(y_true[:,c], pred, average="binary", zero_division=0)
            if floor is not None and prec < floor:  
                continue
            b2 = beta*beta
            fbeta = (1+b2)*prec*rec/(b2*prec+rec+1e-12)
            if fbeta > best:
                best, best_t = fbeta, t
        ths[c] = best_t
    return ths

def main():
    t0 = time.time()
    if not os.path.exists(CONFIRM_CSV) or not os.path.exists(META_JSON):
        raise FileNotFoundError("artifacts/confirmed_dataset.csv 또는 artifacts/meta.json 이 없습니다. 먼저 전처리를 실행하세요.")

    meta = json.load(open(META_JSON, "r", encoding="utf-8"))
    text_col = meta.get("text_col") or "text"
    levels   = meta.get("levels")
    if not levels:

        sample = pd.read_csv(CONFIRM_CSV, nrows=5)
        levels = [c for c in sample.columns if str(c).startswith("L")]
        if not levels:
            raise ValueError("라벨 컬럼(levels)을 찾을 수 없습니다.")

    df = pd.read_csv(CONFIRM_CSV)
    if "need_check_fixed" not in df.columns:
        raise ValueError("confirmed_dataset.csv 에 need_check_fixed가 없습니다.")
    df = df[df["need_check_fixed"]==False].copy().reset_index(drop=True) 
    df[text_col] = df[text_col].map(_clean)

    for lv in levels: df[lv] = df[lv].fillna(0).astype(int)
    Y = df[levels].astype(int).values
    X_text = df[text_col].astype(str)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_text, Y, test_size=0.1, random_state=SEED, stratify=Y.sum(axis=1)
    )

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2),
                               min_df=3, max_df=0.95, max_features=MAXF_WORD,
                               sublinear_tf=True, norm="l2")
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(2,4),
                               min_df=3, max_df=1.0, max_features=MAXF_CHAR,
                               sublinear_tf=True, norm="l2")
    Xtr_w = word_vec.fit_transform(X_tr); Xva_w = word_vec.transform(X_va)
    Xtr_c = char_vec.fit_transform(X_tr); Xva_c = char_vec.transform(X_va)
    XTR = hstack([Xtr_w, Xtr_c]).tocsr()
    XVA = hstack([Xva_w, Xva_c]).tocsr()

    clf = OneVsRestClassifier(
        LogisticRegression(solver="saga", penalty="l2", C=C_PARAM,
                           max_iter=MAX_ITER, class_weight="balanced", n_jobs=-1, verbose=0)
    )
    clf.fit(XTR, y_tr)
    Z_va = clf.decision_function(XVA)              
    Cnum = Z_va.shape[1]
    A = np.zeros(Cnum, dtype=np.float32)
    B = np.zeros(Cnum, dtype=np.float32)
    for c in range(Cnum):
        a, b = _platt_fit_1d(Z_va[:,c], y_va[:,c])
        A[c], B[c] = a, b
    P_va = _platt_apply(Z_va, A, B)            

    ths = _pick_thresholds(
        y_va, P_va,
        default_min_prec=DEFAULT_PREC_FLOOR,
        beta=BETA,
        class_prec_floor=CLASS_PREC_FLOOR
    )
    preds = (P_va >= ths).astype(int)
    print("\n[VALID REPORT]")
    print(classification_report(y_va, preds, target_names=levels, zero_division=0, digits=4))
    mi = f1_score(y_va, preds, average="micro", zero_division=0)
    ma = precision_recall_fscore_support(y_va, preds, average="macro", zero_division=0)[2]
    print({"micro_f1": round(mi,4), "macro_f1": round(ma,4)})

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"word_vec":word_vec, "char_vec":char_vec, "model":clf,
                     "levels":levels, "text_col":text_col}, f)
    np.save(A_PATH, A); np.save(B_PATH, B); np.save(THS_PATH, ths)

    print("\n 완료 / 저장 위치:", os.path.abspath(SAVE_DIR))
    print(" -", MODEL_PATH)
    print(" -", A_PATH)
    print(" -", B_PATH)
    print(" -", THS_PATH)
    print()

if __name__ == "__main__":
    main()

```

# 이미지 학습 모델

```python
import os, time, json, random, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

DATA_ROOT = r"C:\python\minidata"
TRAIN_DIR = os.path.join(DATA_ROOT, "training")  
TEST_DIR  = os.path.join(DATA_ROOT, "testing")    
SAVE_DIR  = os.path.join(DATA_ROOT, "final_image_model_torch"); os.makedirs(SAVE_DIR, exist_ok=True)


SEED=42
IMG=224
BATCH=16                 
VAL_SPLIT=0.15
HEAD_EPOCHS=2            
FT_EPOCHS=10
HEAD_LR=3e-4             
UNFREEZE_BLOCKS=5       
WD=1e-4                 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP=(device.type=="cuda")
NUM_WORKERS=0           
PIN=(device.type=="cuda")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
set_seed()

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.95,1.0)),  
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(5, fill=0),
    transforms.ColorJitter(contrast=0.10),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)
])
eval_tf  = transforms.Compose([
    transforms.Resize(int(IMG*1.14)), transforms.CenterCrop(IMG),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD)
])
def build_model(nc:int):
    m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_f=m.classifier[1].in_features
    m.classifier=nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4), nn.Linear(in_f,nc))
    return m.to(device)

def unfreeze_last(m, n=UNFREEZE_BLOCKS):
    for p in m.parameters(): p.requires_grad=False
    for p in m.classifier.parameters(): p.requires_grad=True
    for blk in list(m.features.children())[-n:]:
        for p in blk.parameters(): p.requires_grad=True

def step_print(tag, ep, step, total, t0):
    if step==1 or step==total or step%(max(1,total//20))==0:
        dt=time.time()-t0; eta=dt/step*(total-step)
        print(f"  {tag} E{ep:02d} [{step:>4}/{total:<4}] {dt/step:.2f}s/it ETA {eta/60:.1f}m", flush=True)

def make_criterion(class_weights: torch.Tensor):
    try:
        return nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    except TypeError:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))

def make_loaders_and_weights(train_root, test_root):
    probe=datasets.ImageFolder(train_root)
    idx=np.arange(len(probe.samples)); y_all=np.array([t for _,t in probe.samples])
    tr,va=train_test_split(idx, test_size=VAL_SPLIT, random_state=SEED, stratify=y_all)
    cnt=np.bincount(y_all[tr], minlength=len(probe.classes)).astype(np.float32)
    w=1.0/(cnt+1e-9); w=w/w.sum()*len(w)
    class_weights=torch.tensor(w, dtype=torch.float32)

    train_ds=Subset(datasets.ImageFolder(train_root, transform=train_tf), tr.tolist())
    val_ds  =Subset(datasets.ImageFolder(train_root, transform=eval_tf),   va.tolist())
    test_ds =datasets.ImageFolder(test_root,  transform=eval_tf)
    tl=DataLoader(train_ds, batch_size=BATCH,   shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN, drop_last=False)
    vl=DataLoader(val_ds,   batch_size=BATCH*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN)
    tel=DataLoader(test_ds,  batch_size=BATCH*2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN)
    return tl, vl, tel, probe.classes, test_ds, class_weights

def train_epoch(m, dl, crit, opt, scaler, ep, tag, scheduler=None):
    m.train(); tot=0.0; n=len(dl); t0=time.time()
    if n==0: print("[WARN] empty loader"); return 0.0
    for i,(x,y) in enumerate(dl,1):
        x=x.to(device); y=y.to(device)
        with torch.cuda.amp.autocast(enabled=AMP):
            logit=m(x); loss=crit(logit,y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        if scheduler is not None: scheduler.step()  
        tot+=loss.item(); step_print(tag, ep, i, n, t0)
    return tot/n

@torch.no_grad()
def eval_metrics(m, dl):
    m.eval(); ys=[]; ps=[]
    for x,y in dl:
        x=x.to(device)
        with torch.cuda.amp.autocast(enabled=AMP): logit=m(x)
        ys.append(y.numpy()); ps.append(logit.argmax(1).cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    return accuracy_score(y,p), f1_score(y,p,average="macro"), y, p

def save_reports(save_dir, test_ds, acc, f1, y, p):
    os.makedirs(save_dir, exist_ok=True)
    rep=classification_report(y,p,target_names=test_ds.classes, digits=4)
    cm =confusion_matrix(y,p)
    with open(os.path.join(save_dir,"test_report.txt"),"w",encoding="utf-8") as f:
        f.write(f"Acc: {acc:.4f} | Macro-F1: {f1:.4f}\n\n"+rep)
    np.savetxt(os.path.join(save_dir,"confusion_matrix.csv"), cm, fmt="%d", delimiter=",")
    with open(os.path.join(save_dir,"classes.txt"),"w",encoding="utf-8") as f:
        for c in test_ds.classes: f.write(c+"\n")
    print(rep); print(cm)
def main():
    print(f"[Device] {device} | CUDA={torch.cuda.is_available()} | mode=single")
    tl,vl,tel, classes, test_ds, class_weights = make_loaders_and_weights(TRAIN_DIR, TEST_DIR)
    print(f"[Classes] {classes} | sizes train={len(tl.dataset)} val={len(vl.dataset)} test={len(test_ds)}")

    m=build_model(len(classes))
    crit=make_criterion(class_weights)
    scaler=torch.cuda.amp.GradScaler(enabled=AMP)

    for p in m.parameters(): p.requires_grad=False
    for p in m.classifier.parameters(): p.requires_grad=True
    opt_head=torch.optim.AdamW(filter(lambda p:p.requires_grad, m.parameters()),
                               lr=HEAD_LR, weight_decay=WD)
    print(f"[Head] {HEAD_EPOCHS} epochs, lr={HEAD_LR}")
    for ep in range(1, HEAD_EPOCHS+1):
        tr=train_epoch(m, tl, crit, opt_head, scaler, ep, "Head")
        vacc,vf1,_,_=eval_metrics(m, vl)
        print(f"    -> val_acc {vacc:.4f} | val_f1 {vf1:.4f}")

    unfreeze_last(m, UNFREEZE_BLOCKS)
    last_blocks = list(m.features.children())[-UNFREEZE_BLOCKS:]
    params_head = list(m.classifier.parameters())
    params_last = [p for blk in last_blocks for p in blk.parameters()]
    head_ids={id(p) for p in params_head}
    last_ids={id(p) for p in params_last}
    params_others = [p for p in m.parameters()
                     if p.requires_grad and (id(p) not in head_ids) and (id(p) not in last_ids)]
    print(f"[FT groups] head={len(params_head)} last={len(params_last)} others={len(params_others)}")

    opt = torch.optim.AdamW([
        {"params": params_head,   "lr": 1e-4},
        {"params": params_last,   "lr": 3e-5},
        {"params": params_others, "lr": 1e-5},
    ], weight_decay=WD)

    total_steps  = FT_EPOCHS * max(1, len(tl))
    warmup_steps = max(1, 3 * len(tl))
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    best_acc=-1.0
    area_dir=os.path.join(SAVE_DIR, "all"); os.makedirs(area_dir, exist_ok=True)
    best_path=os.path.join(area_dir, "best_full.pt")

    print(f"[FT] {FT_EPOCHS} epochs, lr(head=1e-4,last=3e-5,others=1e-5), unfreeze last {UNFREEZE_BLOCKS}")
    for ep in range(1, FT_EPOCHS+1):
        tr=train_epoch(m, tl, crit, opt, scaler, ep, "FT", scheduler)
        vacc,vf1,_,_=eval_metrics(m, vl)
        print(f"    -> val_acc {vacc:.4f} | val_f1 {vf1:.4f}")
        if vacc>best_acc:
            best_acc=vacc; torch.save(m.state_dict(), best_path)

    m.load_state_dict(torch.load(best_path, map_location=device))
    acc,f1,y,p=eval_metrics(m, tel)
    print("\n[TEST] Acc {:.4f} | Macro-F1 {:.4f}".format(acc, f1))
    save_reports(area_dir, test_ds, acc, f1, y, p)
    with open(os.path.join(area_dir,"train_summary.json"),"w",encoding="utf-8") as f:
        json.dump({"best_val_acc":float(best_acc), "test_acc":float(acc), "test_macro_f1":float(f1)}, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {best_path}")

if __name__=="__main__":
    main()

```

#  웹 페이지

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
