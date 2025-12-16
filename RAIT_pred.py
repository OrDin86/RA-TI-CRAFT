# infer_craft_ra_ti_single.py
# -----------------------------------------------------------
# CRAFT_4ch Inference (Region / Affinity / Table / Image)
# - 입력: gray / grad / highpass (3채널, 0~255 PNG)
# - 출력: 각 클래스 확률맵(.png, 0~255) + 이진 마스크(.png, 0/255)
# - 슬라이딩 윈도우 추론(타일/오버랩/배치), AMP 지원
# - 깨진 PNG 대비: 안전 로더 + (옵션) gray에서 grad/highpass 자동 계산
# -----------------------------------------------------------
import os, glob, argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 잘린 PNG도 읽도록

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 학습과 동일한 백본 유틸 (로컬 모듈)
# -----------------------
from basenet.vgg16_bn import vgg16_bn, init_weights  # ← 훈련 때 쓰던거 그대로

# -----------------------
# 모델 정의 (학습 코드와 동일)
# -----------------------
class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
        feat = self.upconv4(y)

        logits = self.conv_cls(feat)
        return {"raw": {"cls": logits}, "feature": feat}

# -----------------------
# 유틸
# -----------------------
CLS = ("region","affinity","table","image")

def seed_everything(seed=42):
    import random, os
    random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def to_uint8(x: np.ndarray):
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def read_L_u8_safe(path: str):
    """깨진 PNG도 최대한 읽고, 실패시 None"""
    try:
        with Image.open(path) as im:
            im.load()
            return np.array(im.convert("L"), dtype=np.uint8)
    except Exception as e:
        print(f"[WARN] cannot read image: {path} -> {e}")
        return None

def compute_grad_highpass_from_gray(gray_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """gray -> Sobel magnitude (grad), Laplacian abs (highpass) 0..255"""
    g = gray_u8.astype(np.float32) / 255.0
    # Sobel
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)
    G = torch.from_numpy(g).view(1,1,*g.shape)
    gx = F.conv2d(G, torch.from_numpy(kx).view(1,1,3,3), padding=1)
    gy = F.conv2d(G, torch.from_numpy(ky).view(1,1,3,3), padding=1)
    grad = torch.sqrt(gx**2 + gy**2).squeeze().numpy()
    if grad.max() > 0: grad = grad / grad.max()
    # Laplacian
    kL = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], np.float32)
    hp = F.conv2d(G, torch.from_numpy(kL).view(1,1,3,3), padding=1).abs().squeeze().numpy()
    if hp.max() > 0: hp = hp / hp.max()
    return to_uint8(grad), to_uint8(hp)

def collect_triplets(input_root: str, compute_from_gray: bool=False) -> List[Dict[str,str]]:
    """
    반환: [{"stem": "...", "gray": "...", "grad": "...", "highpass": "...", "folder_name": "..."}]
    - root/page_x/images/*.png 또는 평평한 폴더(*_gray/_grad/_highpass) 모두 지원
    """
    items = []
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"input_root not found: {root}")

    def scan_img_dir(img_dir: Path, folder_name: str):
        grays = sorted(img_dir.glob("*_gray.png"))
        for g in grays:
            stem = g.name[: g.name.rfind("_gray.png")]
            rec = {"stem": stem, "gray": str(g), "folder_name": folder_name}
            grad = img_dir / f"{stem}_grad.png"
            hip  = img_dir / f"{stem}_highpass.png"
            if grad.exists(): rec["grad"] = str(grad)
            if hip.exists():  rec["highpass"] = str(hip)
            items.append(rec)

    page_dirs = sorted([p for p in root.glob("*") if p.is_dir()])
    if any((p / "images").is_dir() for p in page_dirs):
        for pd in page_dirs:
            img_dir = pd / "images"
            if img_dir.is_dir():
                scan_img_dir(img_dir, folder_name=pd.name)
    else:
        scan_img_dir(root, folder_name=root.name)

    # grad/highpass 없는데 compute_from_gray False면 제외
    out = []
    for rec in items:
        have_grad = ("grad" in rec) and Path(rec["grad"]).exists()
        have_hip  = ("highpass" in rec) and Path(rec["highpass"]).exists()
        if not (have_grad and have_hip) and not compute_from_gray:
            print(f"[WARN] missing grad/highpass -> skip: {rec['stem']} (use --compute_from_gray)")
            continue
        out.append(rec)

    print(f"[INFO] found {len(out)} triplets")
    return out

# -----------------------
# 슬라이딩 윈도우 추론
# -----------------------
@torch.no_grad()
def predict_sliding(
    model: nn.Module,
    img3: torch.Tensor,      # [1,3,H,W], float(0..1)
    tile: int = 1024,
    overlap: int = 256,
    batch_size: int = 4,
    amp: bool = False
) -> torch.Tensor:
    """
    반환: prob [1,4,H,W] (0..1)
    """
    device = next(model.parameters()).device
    _, _, H, W = img3.shape
    step = max(1, tile - overlap)

    ys = list(range(0, max(1, H - tile + 1), step))
    xs = list(range(0, max(1, W - tile + 1), step))
    last_y = max(0, H - tile); last_x = max(0, W - tile)
    if not ys or ys[-1] != last_y: ys.append(last_y)
    if not xs or xs[-1] != last_x: xs.append(last_x)

    tiles, coords = [], []
    for y0 in ys:
        for x0 in xs:
            h = min(tile, H - y0); w = min(tile, W - x0)
            patch = img3[:, :, y0:y0+h, x0:x0+w]
            if h < tile or w < tile:
                pad = (0, tile-w, 0, tile-h)
                patch = F.pad(patch, pad, mode="reflect")
            tiles.append(patch)
            coords.append((y0, x0, h, w))

    prob_acc = torch.zeros(1, 4, H, W, device=device, dtype=torch.float32)
    weight   = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)

    use_amp = bool(amp and torch.cuda.is_available())
    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else torch.no_grad()

    model.eval()
    for i in range(0, len(tiles), batch_size):
        batch = torch.cat(tiles[i:i+batch_size], dim=0).to(device, non_blocking=True)  # [B,3,tile,tile]
        with amp_ctx:
            out = model(batch)
            logits = out["raw"]["cls"]  # [B,4,*,*]
            # 모델 출력 공간 크기 보정
            if logits.shape[-2:] != batch.shape[-2:]:
                logits = F.interpolate(logits, size=batch.shape[-2:], mode="bilinear", align_corners=False)
            probs  = torch.sigmoid(logits)

        for b in range(probs.size(0)):
            idx = i + b
            if idx >= len(coords): break
            y0, x0, h, w = coords[idx]
            prob_acc[:, :, y0:y0+h, x0:x0+w] += probs[b:b+1, :, :h, :w]
            weight[:, :, y0:y0+h, x0:x0+w]   += 1.0

    prob = prob_acc / torch.clamp_min(weight, 1.0)
    return prob

# -----------------------
# 저장
# -----------------------
def save_maps(out_dir: Path, folder: str, prob: torch.Tensor,
              thr: Tuple[float,float,float,float]):
    dst = out_dir / folder
    dst.mkdir(parents=True, exist_ok=True)
    prob = prob.squeeze(0).cpu().numpy()  # [4,H,W], 0..1

    for ci, name in enumerate(CLS):
        pm = prob[ci]
        pm_u8 = to_uint8(pm)
        Image.fromarray(pm_u8).save(dst / f"{name}_prob.png")
        bin_u8 = (pm > thr[ci]).astype(np.uint8) * 255
        Image.fromarray(bin_u8).save(dst / f"{name}_bin.png")

# -----------------------
# 메인
# -----------------------
def get_args():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--ckpt", type=str, default=r"E:\save_RATI_CRAFTbase\best.pt",
    #                 help="checkpoint path (best.pt/last.pt)")
    ap.add_argument("--ckpt", type=str, default=r"E:\last_rati_nopre\nopretrained\nothresh\weight\best.pt",
                    help="checkpoint path (best.pt/last.pt)")
    ap.add_argument("--input_root", type=str, default=r"E:\Train\19",
                    help="입력 루트 (page_x/images 또는 평평한 폴더)")
    ap.add_argument("--out_dir", type=str, default=r"E:\last_rati_nopre\nopretrained\nothresh\pred",
                    help="출력 폴더")

    # 타일/배치/AMP
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--amp", action="store_true")

    # threshold (저장용 bin)
    ap.add_argument("--thr_region", type=float, default=0.50)
    ap.add_argument("--thr_affinity", type=float, default=0.45)
    ap.add_argument("--thr_table", type=float, default=0.45)
    ap.add_argument("--thr_image", type=float, default=0.50)

    # 장치/기타
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--strict", type=int, default=1, help="load_state_dict strict")
    ap.add_argument("--compute_from_gray", action="store_true",
                    help="grad/highpass가 없거나 깨졌으면 gray에서 자동 계산")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def main():
    args = get_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[Device] {device}")

    # 모델 로드
    model = CRAFT_4ch(pretrained=False, freeze=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_m = ckpt.get("model", ckpt)
    try:
        model.load_state_dict(state_m, strict=bool(args.strict))
    except RuntimeError as e:
        print(f"[WARN] load_state_dict(strict={bool(args.strict)}) 실패: {e}")
    model.eval()

    # 입력 모으기
    triplets = collect_triplets(args.input_root, compute_from_gray=args.compute_from_gray)
    if len(triplets) == 0:
        raise RuntimeError("입력이 없습니다.")

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    thr = (args.thr_region, args.thr_affinity, args.thr_table, args.thr_image)

    for rec in triplets:
        stem = rec["stem"]; folder = rec["folder_name"]

        # 1) gray (필수)
        g = read_L_u8_safe(rec["gray"])
        if g is None:
            print(f"[SKIP] gray broken: {rec['gray']}")
            continue

        # 2) grad / highpass 안전 로딩
        gx = read_L_u8_safe(rec["grad"]) if "grad" in rec and Path(rec["grad"]).exists() else None
        hp = read_L_u8_safe(rec["highpass"]) if "highpass" in rec and Path(rec["highpass"]).exists() else None

        # 필요하면 gray에서 계산
        need_grad = gx is None
        need_hp   = hp is None
        if need_grad or need_hp:
            if args.compute_from_gray:
                print(f"[INFO] compute from gray: {stem} (need_grad={need_grad}, need_hp={need_hp})")
                g_grad, g_hp = compute_grad_highpass_from_gray(g)
                if need_grad: gx = g_grad
                if need_hp:   hp = g_hp
            else:
                print(f"[SKIP] missing/broken grad/highpass: stem={stem} (use --compute_from_gray)")
                continue

        # 3) 스택 & 정규화 (/255)
        img3 = np.stack([g, gx, hp], axis=0)  # [3,H,W]
        img_t = torch.from_numpy(img3).unsqueeze(0).to(dtype=torch.float32).div_(255.0)  # [1,3,H,W]

        # 4) 추론
        with torch.no_grad():
            prob = predict_sliding(
                model, img_t.to(device),
                tile=args.tile, overlap=args.overlap,
                batch_size=args.batch_size, amp=args.amp
            )

        # 5) 저장
        save_maps(out_root, folder, prob, thr)
        print(f"[OK] {folder}/{stem} -> {out_root / folder}")

    print("[Done] inference completed.")

if __name__ == "__main__":
    main()
