import os

import joblib
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

MODEL_ID = "google/siglip-base-patch16-224"

LONG_PROMPTS = [
    "a photo of a person wearing a long-sleeve shirt",
    "a photo of a person wearing a jacket with long sleeves",
    "a person whose arms are fully covered by sleeves",
]
SHORT_PROMPTS = [
    "a photo of a person wearing a short-sleeve t-shirt",
    "a photo of a person with bare upper arms",
    "a person wearing a sleeveless or short-sleeve top",
]


def _tensor(x):
    """Different transformers versions return slightly different things."""
    if isinstance(x, torch.Tensor):
        return x
    for name in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(x, name, None)
        if v is not None:
            return v
    raise RuntimeError("Unexpected output type from SigLIP")


class SleeveClassifier:
    """Embeds person crops with SigLIP, then classifies long vs short sleeve.
    - If a trained classifier file exists (sleeve_clf.joblib), uses it (best).
    - Otherwise falls back to zero-shot text prompts (rough)."""

    def __init__(self, clf_path=None, unsure_below=0.6):
        print("Loading SigLIP...")
        self.model = AutoModel.from_pretrained(MODEL_ID).eval()
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.unsure_below = unsure_below
        self.clf = None
        if clf_path and os.path.exists(clf_path):
            self.clf = joblib.load(clf_path)
            print(f"Using trained sleeve classifier: {clf_path}")
        else:
            print("No trained classifier found -> using zero-shot prompts (less accurate).")
            self._prepare_text()

    def _prepare_text(self):
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        feats = []
        with torch.inference_mode():
            for prompts in (LONG_PROMPTS, SHORT_PROMPTS):
                t = tok(prompts, padding="max_length", max_length=64,
                        return_tensors="pt")
                f = _tensor(self.model.get_text_features(**t))
                f = f / f.norm(dim=-1, keepdim=True)
                m = f.mean(dim=0)
                feats.append(m / m.norm())
        self.text_feats = torch.stack(feats)  # (2, D)

    def embed(self, images, batch_size=16):
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                inp = self.proc(images=images[i:i + batch_size], return_tensors="pt")
                f = _tensor(self.model.get_image_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f)
        return torch.cat(out).numpy()

    def predict(self, crops):
        """crops: list of PIL images (or None). Returns [(label, confidence), ...]"""
        results = [("unsure", 0.0)] * len(crops)
        idx = [i for i, c in enumerate(crops) if c is not None]
        if not idx:
            return results
        feats = self.embed([crops[i] for i in idx])

        if self.clf is not None:
            probs = self.clf.predict_proba(feats)
            classes = list(self.clf.classes_)
            for k, i in enumerate(idx):
                j = int(np.argmax(probs[k]))
                label, conf = classes[j], float(probs[k][j])
                results[i] = (label if conf >= self.unsure_below else "unsure", conf)
        else:
            scale = self.model.logit_scale.exp().item()
            logits = scale * torch.from_numpy(feats) @ self.text_feats.T
            probs = logits.softmax(dim=-1).numpy()
            for k, i in enumerate(idx):
                p_long, p_short = probs[k]
                label = "long" if p_long >= p_short else "short"
                conf = float(max(p_long, p_short))
                results[i] = (label if conf >= self.unsure_below else "unsure", conf)
        return results