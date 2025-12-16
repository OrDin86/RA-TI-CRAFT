# ★★★★★★★ 정상학습 -------------------------------------
# train_craft_ra_ti_multi.py
# -----------------------------------------------------------
# CRAFT Base 멀티태스크 (Region / Affinity / Table / Image) 4채널 학습
# - 입력: gray / grad / highpass (3채널, 파일로 제공)
# - 라벨: region, affinity, table, image (각 디렉토리 단일 PNG 사용; 없으면 0으로 대체)
# - 손실: BCEWithLogitsLoss(채널별 pos_weight 지원) + 채널 가중치 cw_R/A/T/I
# - AMP(torch.cuda.amp), Windows DataLoader, 클래스별 Dice, 체크포인트/재개
# - basenet.vgg16_bn / init_weights 그대로 사용
# -----------------------------------------------------------
import os, glob, time, random, argparse
from pathlib import Path
from contextlib import nullcontext
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ★ 네 환경의 백본 유틸 (그대로 사용)
from basenet.vgg16_bn import vgg16_bn, init_weights


# =======================
# CONFIG
# =======================
def get_cfg():
    p = argparse.ArgumentParser()
    # 경로
    p.add_argument("--train_root", type=str, default=r"E:\Train\01")
    p.add_argument("--val_root",   type=str, default=r"E:\Train\02")
    p.add_argument("--save_dir",   type=str, default=r"E:\_origin_pdf")

    # 데이터
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--spe_train", type=int, default=2000)
    p.add_argument("--spe_val",   type=int, default=400)
    p.add_argument("--bs",        type=int, default=4)
    p.add_argument("--nw",        type=int, default=8)
    p.add_argument("--drop_last", action="store_true")

    # 학습
    p.add_argument("--epochs",    type=int, default=50)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--amp",       action="store_true")  # torch.cuda.amp
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--print_every",  type=int, default=50)
    p.add_argument("--seed",      type=int, default=42)

    # 손실 가중치 (채널별 곱해지는 coef; 예: affinity/테이블/이미지 강조)
    p.add_argument("--cw_region",   type=float, default=1.0)
    p.add_argument("--cw_affinity", type=float, default=1.0)
    p.add_argument("--cw_table",    type=float, default=1.0)
    p.add_argument("--cw_image",    type=float, default=1.0)

    # BCE pos_weight (희소 클래스 보정; 0이면 비활성)
    p.add_argument("--pw_region",   type=float, default=1.0)
    p.add_argument("--pw_affinity", type=float, default=1.0)
    p.add_argument("--pw_table",    type=float, default=1.0)
    p.add_argument("--pw_image",    type=float, default=1.0)

    # Dice threshold
    p.add_argument("--thr_region",   type=float, default=0.50)
    p.add_argument("--thr_affinity", type=float, default=0.40)
    p.add_argument("--thr_table",    type=float, default=0.40)
    p.add_argument("--thr_image",    type=float, default=0.50)

    # 크롭/양성유도 세팅
    p.add_argument("--pos_center_prob", type=float, default=1.0,
                   help="양성 좌표 중심 크롭을 시도할 확률")
    p.add_argument("--pcw_region",   type=float, default=1.0,
                   help="양성 중심 클래스 샘플링 가중치 (region)")
    p.add_argument("--pcw_affinity", type=float, default=1.0,
                   help="양성 중심 클래스 샘플링 가중치 (affinity)")
    p.add_argument("--pcw_table",    type=float, default=1.0,
                   help="양성 중심 클래스 샘플링 가중치 (table)")
    p.add_argument("--pcw_image",    type=float, default=1.0,
                   help="양성 중심 클래스 샘플링 가중치 (image)")
    p.add_argument("--min_frac_region",   type=float, default=0.0,
                   help="크롭 내 region 최소양성 비율")
    p.add_argument("--min_frac_affinity", type=float, default=0.0,
                   help="크롭 내 affinity 최소양성 비율")
    p.add_argument("--min_frac_table",    type=float, default=0.0,
                   help="크롭 내 table 최소양성 비율")
    p.add_argument("--min_frac_image",    type=float, default=0.0,
                   help="크롭 내 image 최소양성 비율")
    p.add_argument("--max_pos_per_item",  type=int,   default=4096,
                   help="페이지당 저장할 양성 좌표 상한")
    p.add_argument("--max_crop_attempts", type=int,   default=24)
    p.add_argument("--item_retries",      type=int,   default=4)

    # 재개
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--resume_strict", type=int, default=1)
    p.add_argument("--resume_reset_opt", action="store_true")
    p.add_argument("--lr_override", type=float, default=None)
    return p.parse_args()


# =======================
# Utils
# =======================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trn

def dice_from_logits(logit: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-6):
    prob = torch.sigmoid(logit)
    pred = (prob > thr).float()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return ((2*inter + eps) / (denom + eps)).mean()

def save_ckpt(model, optimizer, scaler=None, epoch=0, best=-1e9, path=None, cfg_dict=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best": float(best),
        "cfg": dict(cfg_dict) if cfg_dict is not None else {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[Save] {path}")

def _safe_load_state_dict(module, state, strict=True, name="model"):
    try:
        module.load_state_dict(state, strict=bool(strict))
    except RuntimeError as e:
        print(f"[WARN] {name}.load_state_dict(strict={strict}) 실패: {e}")

def load_ckpt_all(model, optimizer, scaler, path, device,
                  strict=True, reset_opt=False, lr_override=None):
    ckpt = torch.load(path, map_location=device)
    state_m = ckpt.get("model", ckpt)
    _safe_load_state_dict(model, state_m, strict=strict, name="model")
    if not reset_opt:
        if optimizer is not None and isinstance(ckpt.get("optimizer", None), dict):
            try: optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e: print(f"[WARN] optimizer state 복구 실패 → 새로 시작: {e}")
        if scaler is not None and isinstance(ckpt.get("scaler", None), dict):
            try: scaler.load_state_dict(ckpt["scaler"])
            except Exception as e: print(f"[WARN] GradScaler state 복구 실패 → 새로 시작: {e}")
    else:
        print("[INFO] --resume_reset_opt: optimizer/scaler 복구 안 함")
    if lr_override is not None:
        for g in optimizer.param_groups: g["lr"] = float(lr_override)
        print(f"[INFO] lr_override={lr_override} 적용")
    start = int(ckpt.get("epoch", 0)) + 1
    best  = float(ckpt.get("best", -1e9))
    print(f"[Resume] epoch={start}, best={best}, from {path}")
    return start, best

def find_auto_resume_path(save_dir: str):
    sd = Path(save_dir)
    cands = []
    if (sd / "last.pt").exists(): cands.append(sd / "last.pt")
    if (sd / "best.pt").exists(): cands.append(sd / "best.pt")
    cands += sorted(sd.glob("epoch_*.pt"))[::-1]
    return str(cands[0]) if cands else ""


# =======================
# Dataset (R/A/T/I) — 라벨 가공 없음
# =======================
class DocSegDatasetRA_TI(Dataset):
    """
    root/
      page_x/
        images/
          <stem>_gray.png
          <stem>_grad.png
          <stem>_highpass.png
        ann/
          region/*.png
          affinity/*.png
          table/*.png
          image/*.png

    * 크롭은 "클래스 먼저 선택 → 그 클래스를 가진 페이지에서" 양성 중심 크롭을 우선 시도
    * region은 게이트에서 제외하려면 gate_classes=("affinity","table","image")로 두면 됨(기본값)
    """
    CLS = ("region", "affinity", "separator", "image_heat")
    #CLS = ("separator", "image_heat")

    def __init__(self,
                 root_dir: str,
                 patch_size: int = 512,
                 samples_per_epoch: int = 2000,
                 rand_flip: bool = True,
                 rand_rot: bool = True,
                 max_crop_attempts: int = 24,
                 item_retries: int = 4,
                 pos_center_prob: float = 0.7,
                 pcw=(1.0, 2.0, 1.5, 1.5),               # 클래스 선택 확률 가중치 (region,affinity,table,image)
                 min_frac=(0.003, 0.005, 0.010, 0.010),  # 클래스별 최소 양성 비율
                 max_pos_per_item: int = 4096,
                 #gate_classes=("affinity", "table", "image")):
                 gate_classes=("region", "affinity","separator","image_heat")):

        self.root = root_dir
        self.patch = patch_size
        self.N = samples_per_epoch
        self.rand_flip = rand_flip
        self.rand_rot  = rand_rot
        self.max_crop_attempts = max_crop_attempts
        self.item_retries = item_retries
        self.pos_center_prob = pos_center_prob
        self.pcw = np.asarray(pcw, dtype=np.float32)
        self.pcw = self.pcw / max(1e-8, self.pcw.sum())
        self.min_frac = np.asarray(min_frac, dtype=np.float32)
        self.max_pos_per_item = max_pos_per_item
        self.gate = set(gate_classes)  # 양성 판정/타깃 후보로 고려할 클래스 집합 (region 제외 가능)

        self.items: List[Tuple[Dict[str,str], Dict[str,str]]] = []  # (triplet paths, label paths)
        self.pos_centers: List[Dict[str, np.ndarray]] = []          # per item: {cls: coords[K,2]}

        # 1) 아이템 인덱스 구성
        page_dirs = [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
        for pd in page_dirs:
            img_dir = os.path.join(pd, "images")
            ann_dir = os.path.join(pd, "ann")
            if not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
                continue

            gray_list = glob.glob(os.path.join(img_dir, "*_gray.png"))
            stems = {}
            for g in gray_list:
                base = os.path.basename(g)
                stem = base[: base.rfind("_gray.png")]
                stems[stem] = g

            for stem, gpath in stems.items():
                grad = os.path.join(img_dir, f"{stem}_grad.png")
                hip  = os.path.join(img_dir, f"{stem}_highpass.png")
                if not (os.path.exists(grad) and os.path.exists(hip)):
                    continue

                labels = {}
                for cname in self.CLS:
                    cdir = os.path.join(ann_dir, cname)
                    files = glob.glob(os.path.join(cdir, "*.png"))
                    if files:
                        labels[cname] = files[-1]  # 해당 페이지 라벨 PNG (없으면 이후 0으로 대체)
                self.items.append(({"gray":gpath,"grad":grad,"highpass":hip}, labels))

        # 2) 양성 중심 좌표 빌드 (각 클래스별)
        self._build_pos_centers()

        # 3) 클래스별로 "양성 좌표를 가진 페이지 인덱스" 리스트를 구축
        self.items_by_class = {c: [] for c in self.CLS}
        for i, centers in enumerate(self.pos_centers):
            for c in self.CLS:
                if len(centers[c]) > 0:
                    self.items_by_class[c].append(i)

        # (선택) 간단 커버리지 로그
        cov = {c: len(self.items_by_class[c]) for c in self.CLS}
        print(f"[DocSegDatasetRA_TI] items={len(self.items)} | pages with positives: {cov} | gate={self.gate}")

    def _build_pos_centers(self):
        self.pos_centers = []
        for trip, labels in self.items:
            centers = {}
            for cname in self.CLS:
                p = labels.get(cname, None)
                if (p is None) or (not os.path.exists(p)):
                    centers[cname] = np.zeros((0,2), dtype=np.int32); continue
                with Image.open(p) as im:
                    arr = (np.array(im.convert("L"), dtype=np.uint8) > 127)
                ys, xs = np.where(arr)
                K = len(ys)
                if K == 0:
                    centers[cname] = np.zeros((0,2), dtype=np.int32)
                else:
                    if K > self.max_pos_per_item:
                        sel = np.random.choice(K, self.max_pos_per_item, replace=False)
                        ys = ys[sel]; xs = xs[sel]
                    centers[cname] = np.stack([ys, xs], axis=1).astype(np.int32)
            self.pos_centers.append(centers)

    def __len__(self): return self.N

    def _random_crop(self, W, H):
        x0 = 0 if W <= self.patch else random.randint(0, W - self.patch)
        y0 = 0 if H <= self.patch else random.randint(0, H - self.patch)
        return x0, y0, x0+self.patch, y0+self.patch

    def _aug(self, imgs, masks):
        if self.rand_rot:
            k = random.randint(0, 3)
            if k:
                imgs  = [im.rotate(90*k, expand=False) for im in imgs]
                masks = [m.rotate(90*k, expand=False) for m in masks]
        if self.rand_flip:
            if random.random() < 0.5:
                imgs  = [im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs]
                masks = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in masks]
            if random.random() < 0.5:
                imgs  = [im.transpose(Image.FLIP_TOP_BOTTOM) for im in imgs]
                masks = [m.transpose(Image.FLIP_TOP_BOTTOM) for m in masks]
        return imgs, masks

    def __getitem__(self, idx):
        base_idx = idx % len(self.items)

        def _load_or_zero(path, W, H):
            if path and os.path.exists(path):
                return Image.open(path).convert("L")
            return Image.fromarray(np.zeros((H, W), dtype=np.uint8))

        # 클래스별 최소 양성 비율 dict
        mn = {c: self.min_frac[i] for i,c in enumerate(self.CLS)}

        for item_try in range(self.item_retries):
            # --------------------------
            # (A) 타깃 클래스 선택
            # --------------------------
            target_cls = None
            cur_idx = base_idx  # 기본값

            if random.random() < self.pos_center_prob:
                # gate에 포함되고, 실제로 양성 페이지가 있는 클래스만 후보에 올림
                cand = [c for c in self.CLS if (c in self.gate) and (len(self.items_by_class.get(c,[])) > 0)]
                if len(cand) > 0:
                    # 클래스별 선택 확률(PCW) 적용
                    w = np.array([self.pcw[self.CLS.index(c)] for c in cand], dtype=np.float32)
                    w = w / max(1e-8, w.sum())
                    target_cls = np.random.choice(cand, p=w)
                    # ★ 해당 클래스를 가진 페이지에서 cur_idx를 다시 선택
                    cur_idx = random.choice(self.items_by_class[target_cls])
                else:
                    # gate에 해당하는 양성 페이지가 하나도 없으면 타깃 비활성
                    target_cls = None

            trip, labels = self.items[cur_idx]

            # --------------------------
            # (B) 캔버스 크기 & 원본 마스크 로딩(이진)
            # --------------------------
            with Image.open(trip["gray"]) as g0:
                W, H = g0.size

            with _load_or_zero(labels.get("region"), W, H) as reg0: reg_pick = (np.asarray(reg0, dtype=np.uint8) > 127)
            with _load_or_zero(labels.get("affinity"), W, H) as aff0: aff_pick = (np.asarray(aff0, dtype=np.uint8) > 127)
            with _load_or_zero(labels.get("separator"), W, H) as tab0: tab_pick = (np.asarray(tab0, dtype=np.uint8) > 127)
            with _load_or_zero(labels.get("image_heat"), W, H) as img0: img_pick = (np.asarray(img0, dtype=np.uint8) > 127)

            def _class_ok(cname, y0, y1, x0, x1):
                if cname == "region":   return reg_pick[y0:y1, x0:x1].mean() >= mn["region"]
                if cname == "affinity": return aff_pick[y0:y1, x0:x1].mean() >= mn["affinity"]
                if cname == "separator":    return tab_pick[y0:y1, x0:x1].mean() >= mn["separator"]
                if cname == "image_heat":    return img_pick[y0:y1, x0:x1].mean() >= mn["image_heat"]
                return False

            found = False
            half = self.patch // 2

            # --------------------------
            # (C) 타깃 클래스 중심 시도 (타깃 클래스만 기준 충족해야 통과)
            # --------------------------
            if target_cls is not None:
                coords = self.pos_centers[cur_idx][target_cls]
                if len(coords) > 0:
                    for _ in range(self.max_crop_attempts):
                        cy, cx = coords[np.random.randint(len(coords))]
                        jx = np.random.randint(-16, 17); jy = np.random.randint(-16, 17)
                        x0 = int(np.clip(cx - half + jx, 0, max(0, W - self.patch)))
                        y0 = int(np.clip(cy - half + jy, 0, max(0, H - self.patch)))
                        x1, y1 = x0 + self.patch, y0 + self.patch
                        if _class_ok(target_cls, y0, y1, x0, x1):
                            found = True
                            break

            # --------------------------
            # (D) 랜덤 시도 (gate 내 아무 클래스나 충족하면 OK)
            # --------------------------
            if not found:
                for _ in range(self.max_crop_attempts):
                    x0, y0, x1, y1 = self._random_crop(W, H)
                    if any(_class_ok(c, y0, y1, x0, x1) for c in self.gate):
                        found = True
                        break

            if not found:
                x0, y0, x1, y1 = self._random_crop(W, H)

            # --------------------------
            # (E) 실제 크롭 & Aug & Tensor화
            # --------------------------
            with Image.open(trip["gray"]) as g0:
                g_c  = g0.convert("L").crop((x0, y0, x1, y1)).copy()
            with Image.open(trip["grad"]) as gx0:
                gx_c = gx0.convert("L").crop((x0, y0, x1, y1)).copy()
            with Image.open(trip["highpass"]) as hp0:
                hp_c = hp0.convert("L").crop((x0, y0, x1, y1)).copy()

            reg_c = Image.fromarray(reg_pick[y0:y1, x0:x1].astype(np.uint8) * 255)
            aff_c = Image.fromarray(aff_pick[y0:y1, x0:x1].astype(np.uint8) * 255)
            tab_c = Image.fromarray(tab_pick[y0:y1, x0:x1].astype(np.uint8) * 255)
            img_c = Image.fromarray(img_pick[y0:y1, x0:y1].astype(np.uint8) * 255) if False else Image.fromarray(img_pick[y0:y1, x0:x1].astype(np.uint8) * 255)  # 안전

            [g_c, gx_c, hp_c], [reg_c, aff_c, tab_c, img_c] = self._aug([g_c, gx_c, hp_c], [reg_c, aff_c, tab_c, img_c])

            g_np  = np.asarray(g_c,  dtype=np.uint8)
            gx_np = np.asarray(gx_c, dtype=np.uint8)
            hp_np = np.asarray(hp_c, dtype=np.uint8)
            img_np = np.stack([g_np, gx_np, hp_np], axis=0)  # [3,H,W]
            x = torch.from_numpy(img_np).to(dtype=torch.float32).div_(255.0)

            def _bin(pil_img): return torch.from_numpy((np.asarray(pil_img, dtype=np.uint8) > 127)).to(dtype=torch.float32)

            reg = _bin(reg_c).view(1, x.shape[1], x.shape[2])
            aff = _bin(aff_c).view(1, x.shape[1], x.shape[2])
            tab = _bin(tab_c).view(1, x.shape[1], x.shape[2])
            imc = _bin(img_c).view(1, x.shape[1], x.shape[2])

            y = torch.cat([reg, aff, tab, imc], dim=0)  # [4,H,W]
            return x, y

        # 폴백: item_retries 실패 시 랜덤 크롭
        trip, labels = self.items[base_idx]
        with Image.open(trip["gray"]) as g0, Image.open(trip["grad"]) as gx0, Image.open(trip["highpass"]) as hp0:
            W, H = g0.size
            x0, y0, x1, y1 = self._random_crop(W, H)
            g_c  = g0.convert("L").crop((x0, y0, x1, y1)).copy()
            gx_c = gx0.convert("L").crop((x0, y0, x1, y1)).copy()
            hp_c = hp0.convert("L").crop((x0, y0, x1, y1)).copy()

        def _load_or_zero2(path):
            if path and os.path.exists(path): return Image.open(path).convert("L")
            return Image.fromarray(np.zeros((H, W), dtype=np.uint8))
        with _load_or_zero2(labels.get("region")) as reg0: reg_c = reg0.crop((x0, y0, x1, y1)).copy()
        with _load_or_zero2(labels.get("affinity")) as aff0: aff_c = aff0.crop((x0, y0, x1, y1)).copy()
        with _load_or_zero2(labels.get("separator")) as tab0: tab_c = tab0.crop((x0, y0, x1, y1)).copy()
        with _load_or_zero2(labels.get("iimage_heat")) as img0: img_c = img0.crop((x0, y0, x1, y1)).copy()

        [g_c, gx_c, hp_c], [reg_c, aff_c, tab_c, img_c] = self._aug([g_c, gx_c, hp_c], [reg_c, aff_c, tab_c, img_c])

        g_np  = np.asarray(g_c,  dtype=np.uint8)
        gx_np = np.asarray(gx_c, dtype=np.uint8)
        hp_np = np.asarray(hp_c, dtype=np.uint8)
        img_np = np.stack([g_np, gx_np, hp_np], axis=0)
        x = torch.from_numpy(img_np).to(dtype=torch.float32).div_(255.0)

        reg_np = (np.asarray(reg_c, dtype=np.uint8) > 127)
        aff_np = (np.asarray(aff_c, dtype=np.uint8) > 127)
        tab_np = (np.asarray(tab_c, dtype=np.uint8) > 127)
        img_np2= (np.asarray(img_c, dtype=np.uint8) > 127)
        y = torch.from_numpy(np.stack([reg_np, aff_np, tab_np, img_np2], axis=0)).to(dtype=torch.float32)
        return x, y


# =======================
# Collate
# =======================
def collate_fn(batch):
    imgs, ys = [], []
    for img, y in batch:
        imgs.append(img); ys.append(y)
    imgs = torch.stack(imgs, dim=0)        # [B,3,H,W]
    ys   = torch.stack(ys,   dim=0)        # [B,4,H,W]
    return imgs, ys


# =======================
# Model (CRAFT Base 4ch)
# =======================
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class CRAFT_4ch(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()
        self.basenet = vgg16_bn(pretrained, freeze)
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512,  256, 128)
        self.upconv3 = double_conv(256,  128, 64)
        self.upconv4 = double_conv(128,   64, 32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4,   kernel_size=1),  # [region, affinity, table, image]
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1); y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1); y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1); y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        logits = self.conv_cls(feature)
        return {"raw": {"cls": logits}, "feature": feature}


# =======================
# Loss (4ch, BCE+pos_weight, 채널가중치)
# =======================
class Loss4(nn.Module):
    def __init__(self, cw=(1.0,1.8,1.0,1.0), pos_weight=(0,0,0,0)):
        super().__init__()
        # ★ buffer로 등록하면 criterion.to(device) 시 같이 이동
        cw_t = torch.tensor(cw, dtype=torch.float32).view(1,4,1,1)
        pw_t = torch.tensor([max(0.0, float(x)) for x in pos_weight],
                            dtype=torch.float32).view(1,4,1,1)
        self.register_buffer("cw", cw_t)  # [1,4,1,1]
        self.register_buffer("pw", pw_t)  # [1,4,1,1]

    @staticmethod
    def _resize(x, tgt):
        H, W = tgt.shape[-2:]
        return x if x.shape[-2:] == (H, W) else F.interpolate(x, (H, W), mode="bilinear", align_corners=False)

    def forward(self, out_raw, gt_4):
        logits = out_raw["cls"]
        logits = self._resize(logits, gt_4)  # logits, gt_4 둘 다 같은 device

        # BCE per-pixel
        loss = F.binary_cross_entropy_with_logits(logits, gt_4, reduction='none')

        # 양성 위치에만 pos_weight 근사 적용 (희소 보정)
        if (self.pw > 0).any():
            with torch.no_grad():
                posmask = (gt_4 > 0.5).float()
        # ★ self.pw/self.cw는 이미 같은 device (buffer)
            loss = loss * (1 + posmask * (self.pw - 1.0))

        # 채널 가중치
        loss = loss * self.cw
        l_ch = loss.mean(dim=(0,2,3))   # [4]
        total = loss.mean()
        logs = {
            "total": float(total.detach()),
            "region": float(l_ch[0].detach()),
            "affinity": float(l_ch[1].detach()),
            "separator": float(l_ch[2].detach()),
            "image_heat": float(l_ch[3].detach())
        }
        return total, logs


# =======================
# Train / Validate
# =======================
def train_one_epoch(model, loader, optimizer, scaler, device, criterion, epoch, cfg):
    model.train()
    t0 = time.time()
    meter = {k:0.0 for k in ["loss","region","affinity","separator","image_heat"]}

    use_amp = bool(cfg.amp and torch.cuda.is_available())
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else nullcontext()

    for it, (img, gt4) in enumerate(loader, 1):
        img  = img.to(device, non_blocking=True)
        gt4  = gt4.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(img)
            loss, logs = criterion(out["raw"], gt4)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        for k in meter: meter[k] += logs[k] if k in logs else logs["total"]

        if it % cfg.print_every == 0:
            dt = time.time() - t0
            cur = {k: meter[k]/it for k in meter}
            # 양성률 빠른 프린트
            pos_r = float(gt4[:,0].mean().detach().cpu())
            pos_a = float(gt4[:,1].mean().detach().cpu())
            pos_t = float(gt4[:,2].mean().detach().cpu())
            pos_i = float(gt4[:,3].mean().detach().cpu())
            print(
                f"[Train][Ep {epoch}][{it}/{len(loader)}] "
                f"loss {cur['loss']:.4f} | R {cur['region']:.4f} | A {cur['affinity']:.4f} "
                f"| T {cur['separator']:.4f} | I {cur['image_heat']:.4f} "
                f"| pos R {pos_r:.4f} A {pos_a:.4f} T {pos_t:.4f} I {pos_i:.4f} | {dt:.1f}s"
            )

    for k in meter: meter[k] /= len(loader)
    return meter


@torch.no_grad()
def validate_one_epoch(model, loader, device, criterion, cfg):
    model.eval()
    loss_sum = 0.0
    dice = {"region":0.0, "affinity":0.0, "separator":0.0, "image_heat":0.0}

    use_amp = bool(cfg.amp and torch.cuda.is_available())
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else nullcontext()

    def resize_like(x, tgt):
        H, W = tgt.shape[-2:]
        return x if x.shape[-2:]==(H,W) else F.interpolate(x, (H,W), mode="bilinear", align_corners=False)

    for img, gt4 in loader:
        img = img.to(device); gt4 = gt4.to(device)
        with amp_ctx:
            out = model(img)
            loss, _ = criterion(out["raw"], gt4)
        loss_sum += float(loss)

        cls = out["raw"]["cls"]
        cls = resize_like(cls, gt4)
        thr = [cfg.thr_region, cfg.thr_affinity, cfg.thr_table, cfg.thr_image]
        for c,(name) in enumerate(("region","affinity","separator","image_heat")):
            dice[name] += dice_from_logits(cls[:,c:c+1], gt4[:,c:c+1], thr=thr[c])

    n = len(loader)
    for k in dice: dice[k] /= n
    return {"loss": loss_sum/n, **{f"dice_{k}":v for k,v in dice.items()}}


# =======================
# Main
# =======================
def main():
    cfg = get_cfg()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 모델
    model = CRAFT_4ch(pretrained=False, freeze=False).to(device)
    tot, trn = count_params(model)
    print(f"[Model] total={tot:,} trainable={trn:,}")

    # 옵티마이저 / AMP
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = bool(cfg.amp and torch.cuda.is_available())
    scaler = (torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None)

    # 데이터
    train_ds = DocSegDatasetRA_TI(
        cfg.train_root,
        patch_size=cfg.patch,
        samples_per_epoch=cfg.spe_train,
        rand_flip=True, rand_rot=True,
        max_crop_attempts=1, item_retries=1,
        pos_center_prob=0.0, 
        pcw=(1,1,1,1),
        min_frac=(0,0,0,0),
        max_pos_per_item=0,
        gate_classes=()
    )
    val_ds = DocSegDatasetRA_TI(
        cfg.val_root,
        patch_size=cfg.patch,
        samples_per_epoch=cfg.spe_val,
        rand_flip=False, rand_rot=False,
        max_crop_attempts=1, item_retries=1,
        pos_center_prob=0.0,
        pcw=(1,1,1,1),
        min_frac=(0,0,0,0),
        max_pos_per_item=0,
        gate_classes=()
    )
    val_ds.N = cfg.spe_val  # 빠른 검증

    # DataLoader (Windows OK)
    def make_loader(ds, shuffle, nw, drop_last):
        kwargs = dict(
            batch_size=cfg.bs, shuffle=shuffle, num_workers=nw, pin_memory=True,
            persistent_workers=False, drop_last=drop_last, collate_fn=collate_fn
        )
        return DataLoader(ds, **kwargs)

    train_dl = make_loader(train_ds, True,  cfg.nw, cfg.drop_last)
    val_dl   = make_loader(val_ds,   False, max(1, cfg.nw//2), False)

    # 재개
    start_epoch, best_score = 1, -1e9
    resume_path = cfg.resume or (find_auto_resume_path(cfg.save_dir) if cfg.auto_resume else "")
    if resume_path:
        start_epoch, best_score = load_ckpt_all(
            model, optimizer, scaler, resume_path, device,
            strict=bool(cfg.resume_strict),
            reset_opt=cfg.resume_reset_opt,
            lr_override=cfg.lr_override,
        )

    save_dir = Path(cfg.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    criterion = Loss4(
        cw=(cfg.cw_region, cfg.cw_affinity, cfg.cw_table, cfg.cw_image),
        pos_weight=(cfg.pw_region, cfg.pw_affinity, cfg.pw_table, cfg.pw_image)
    ).to(device)

    for epoch in range(start_epoch, cfg.epochs + 1):
        _ = train_one_epoch(model, train_dl, optimizer, scaler, device, criterion, epoch, cfg)

        if (epoch % cfg.val_interval == 0) or (epoch == cfg.epochs):
            val = validate_one_epoch(model, val_dl, device, criterion, cfg)
            mean_dice = (val["dice_region"] + val["dice_affinity"] + val["dice_separator"] + val["dice_image_heat"]) / 4.0
            print(f"[Val][Ep {epoch}] loss {val['loss']:.4f} | "
                  f"dice R {val['dice_region']:.3f} A {val['dice_affinity']:.3f} "
                  f"T {val['dice_separator']:.3f} I {val['dice_image_heat']:.3f} | mean {mean_dice:.3f}")

            # best ← mean dice 기준(원하면 region/affinity 위주로 바꿔도 됨)
            if mean_dice > best_score:
                best_score = float(mean_dice)
                save_ckpt(model, optimizer, scaler, epoch, best_score, save_dir / "best.pt", cfg_dict=vars(cfg))

        save_ckpt(model, optimizer, scaler, epoch, best_score, save_dir / "last.pt", cfg_dict=vars(cfg))

    print(f"[Done] best mean dice = {best_score:.4f}")


if __name__ == "__main__":
    main()
