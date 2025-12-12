#!/usr/bin/env python3
import os
import glob
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

def build_model(checkpoint_path: str, device: str):
    # Must match the architecture used in training
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # (conf_logit, vx, vy, vz)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    model.to(device).eval()
    return model

def preprocess_jpg_exact(bgr_img: np.ndarray, device: str):
    """
    Mirror your normalize-only pipeline (no resize/crop).
    Input: OpenCV BGR uint8
    Output: torch (1,3,H,W)
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    chw = (chw - mean) / std
    return chw.unsqueeze(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True, help="Folder containing .jpg files")
    ap.add_argument("--out-dir", required=True, help="Folder to write .txt outputs")
    ap.add_argument("--ckpt", required=True, help="Path to .pth checkpoint")
    ap.add_argument("--glob", default="*.jpg", help="Pattern inside --img-dir (default: *.jpg)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.ckpt, device)

    paths = sorted(glob.glob(os.path.join(args.img_dir, args.glob)))
    if not paths:
        raise SystemExit(f"No images found in {args.img_dir} matching {args.glob}")

    use_amp = (device == "cuda")
    for img_path in paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.out_dir, base + ".txt")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[SKIP] Could not read {img_path}")
            continue

        inp = preprocess_jpg_exact(bgr, device)

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    out = model(inp)
            else:
                out = model(inp)

            conf = torch.sigmoid(out[:, :1]).item()  # scalar
            vec_raw = out[:, 1:]                      # (1,3)
            # normalize safely
            vec = F.normalize(vec_raw, p=2, dim=1, eps=1e-12)[0].detach().cpu().numpy()

        # Write: first line confidence, next 3 lines vector components
        with open(out_path, "w") as f:
            f.write(f"{conf:.8f}\n")
            f.write(f"{vec[0]:.8f}\n{vec[1]:.8f}\n{vec[2]:.8f}\n")

        print(f"[OK] {base}: conf={conf:.3f}, vec=({vec[0]:.3f},{vec[1]:.3f},{vec[2]:.3f}) -> {out_path}")

if __name__ == "__main__":
    main()
