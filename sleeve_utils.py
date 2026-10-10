"""Sleeve reader: SegFormer human parsing (no training needed).
Model: https://huggingface.co/mattmdjaga/segformer_b2_clothes (MIT license)
Labels used: 4 Upper-clothes, 7 Dress, 14 Left-arm, 15 Right-arm."""
import cv2
import numpy as np
import torch
from transformers import AutoModelForSemanticSegmentation

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
UPPER_IDS = (4, 7)
ARM_IDS = (14, 15)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SleeveParser:
    def __init__(self, parse_size=384):
        print("Loading SegFormer clothes parser (downloads on first run)...")
        self.model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).eval()
        self.size = parse_size

    def _letterbox(self, img):
        """Resize keeping aspect ratio, pad with grey to a square."""
        S = self.size
        h, w = img.shape[:2]
        s = S / max(h, w)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        r = cv2.resize(img, (nw, nh), interpolation=interp)
        canvas = np.full((S, S, 3), 128, np.uint8)
        canvas[:nh, :nw] = r
        content = np.zeros((S, S), np.uint8)
        content[:nh, :nw] = 1
        return canvas, content

    @torch.inference_mode()
    def parse(self, crops, batch_size=8):
        """crops: list of RGB uint8 arrays.
        Returns a list of dicts: upper, arm, content (pixel counts on the parse grid)."""
        out = []
        for i in range(0, len(crops), batch_size):
            chunk = crops[i:i + batch_size]
            imgs, contents = zip(*[self._letterbox(c) for c in chunk])
            x = torch.from_numpy(np.stack(imgs)).float().div(255).permute(0, 3, 1, 2)
            x = (x - MEAN) / STD
            logits = self.model(pixel_values=x).logits      # (N, classes, S/4, S/4)
            pred = logits.argmax(dim=1).numpy()
            g = pred.shape[1]
            for p, cm in zip(pred, contents):
                cmg = cv2.resize(cm, (g, g), interpolation=cv2.INTER_NEAREST).astype(bool)
                out.append({
                    "upper": int(np.isin(p, UPPER_IDS)[cmg].sum()),
                    "arm": int(np.isin(p, ARM_IDS)[cmg].sum()),
                    "content": max(1, int(cmg.sum())),
                })
        return out