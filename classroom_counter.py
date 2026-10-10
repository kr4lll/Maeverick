"""
Classroom counter: heads -> body masks -> sleeve type. CPU only.

  python classroom_counter.py                       # webcam
  python classroom_counter.py --source test.jpg     # photo
  python classroom_counter.py --source clip.mp4     # video file
  python classroom_counter.py --tiles 2             # better for crowded rooms (slower)
  python classroom_counter.py --save-crops crops    # save masked crops for labelling
Press 'q' in the video window to quit.
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from ultralytics import SAM, YOLO

from sleeve_utils import SleeveClassifier

COLORS = {"long": (0, 200, 0), "short": (0, 140, 255), "unsure": (170, 170, 170)}


# ---------------------------------------------------------------- video input
class WebcamLatest:
    """Background thread keeps only the NEWEST webcam frame."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera: {src}")
        self.frame, self.lock, self.running = None, threading.Lock(), True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


class VideoFile:
    def __init__(self, path, interval):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.skip = max(1, int(fps * interval))

    def read(self):
        for _ in range(self.skip - 1):
            self.cap.grab()
        ok, f = self.cap.read()
        return f if ok else None

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------- head detection
def detect_heads(model, frame, conf, imgsz, tiles):
    """Full-frame pass, plus optional overlapping tiles to catch small heads."""
    H, W = frame.shape[:2]

    def run(img, ox, oy):
        r = model.predict(img, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        if len(b):
            b[:, [0, 2]] += ox
            b[:, [1, 3]] += oy
        return b, s

    all_b, all_s = [], []
    b, s = run(frame, 0, 0)
    all_b.append(b); all_s.append(s)

    if tiles > 1:
        tw, th, ov = W / tiles, H / tiles, 0.2
        for r_ in range(tiles):
            for c in range(tiles):
                x1 = int(max(0, c * tw - ov * tw)); x2 = int(min(W, (c + 1) * tw + ov * tw))
                y1 = int(max(0, r_ * th - ov * th)); y2 = int(min(H, (r_ + 1) * th + ov * th))
                b, s = run(frame[y1:y2, x1:x2], x1, y1)
                all_b.append(b); all_s.append(s)

    boxes = np.concatenate([b for b in all_b if len(b)] or [np.zeros((0, 4))])
    scores = np.concatenate([s for s in all_s if len(s)] or [np.zeros(0)])
    if len(boxes) == 0:
        return boxes
    keep = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32),
                               torch.tensor(scores, dtype=torch.float32), 0.5)
    return boxes[keep.numpy()]


# ---------------------------------------------------------------- body masks
def body_prompt_box(head, W, H):
    x1, y1, x2, y2 = head
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    return [max(0, cx - 1.3 * w), max(0, y1),
            min(W - 1, cx + 1.3 * w), min(H - 1, y2 + 2.5 * h)]


def get_body_masks(sam, frame, heads):
    H, W = frame.shape[:2]
    prompts = [body_prompt_box(h, W, H) for h in heads]
    res = sam(frame, bboxes=prompts, device="cpu", verbose=False)[0]
    raw = res.masks.data.cpu().numpy().astype(bool)

    masks = []
    for i, m in enumerate(raw):
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        bx1, by1, bx2, by2 = [int(v) for v in prompts[i]]
        limit = np.zeros((H, W), bool)
        limit[by1:by2, bx1:bx2] = True
        m = m & limit
        for j, other in enumerate(heads):       # cut out neighbours' heads
            if j != i:
                ox1, oy1, ox2, oy2 = [int(v) for v in other]
                m[oy1:oy2, ox1:ox2] = False
        masks.append(m)
    return masks


def make_crop(frame_rgb, mask):
    ys, xs = np.where(mask)
    if len(ys) < 100:
        return None
    img = frame_rgb.copy()
    img[~mask] = 128
    return Image.fromarray(img[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


# ---------------------------------------------------------------- drawing
def draw(frame, heads, masks, results):
    overlay = frame.copy()
    for m, (label, _) in zip(masks, results):
        overlay[m] = COLORS[label]
    out = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    for h, (label, conf) in zip(heads, results):
        x1, y1, x2, y2 = [int(v) for v in h]
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[label], 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[label], 1)
    n = {k: sum(1 for r in results if r[0] == k) for k in COLORS}
    lines = [f"Students (heads): {len(heads)}", f"Long sleeve: {n['long']}",
             f"Short sleeve: {n['short']}", f"Unsure: {n['unsure']}"]
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, t, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------- main
def process(frame, head_model, sam, sleeve, args):
    heads = detect_heads(head_model, frame, args.conf, args.imgsz, args.tiles)
    print(f"heads found: {len(heads)}")
    if len(heads) == 0:
        return draw(frame, [], [], [])
    masks = get_body_masks(sam, frame, heads)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crops = [make_crop(rgb, m) for m in masks]

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)
        stamp = int(time.time())
        for i, c in enumerate(crops):
            if c is not None:
                c.save(os.path.join(args.save_crops, f"{stamp}_{i}.png"))

    results = sleeve.predict(crops)
    return draw(frame, heads, masks, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 = webcam, or photo/video path")
    ap.add_argument("--head-weights", default="medium.pt")
    ap.add_argument("--clf", default="sleeve_clf.joblib", help="trained sleeve classifier")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tiles", type=int, default=1, help="1 = off, 2 = 2x2 tiles, 3 = 3x3")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--save-crops", default=None, help="folder to save crops for labelling")
    args = ap.parse_args()

    print("Loading head detector...")
    head_model = YOLO(args.head_weights)
    print("Loading MobileSAM (downloads on first run)...")
    sam = SAM("mobile_sam.pt")
    sleeve = SleeveClassifier(args.clf)

    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        out = process(cv2.imread(args.source), head_model, sam, sleeve, args)
        cv2.imshow("Classroom counter (any key to close)", out)
        cv2.waitKey(0)
        return

    if args.source.isdigit():
        src = WebcamLatest(int(args.source))
        time.sleep(1.0)
    else:
        src = VideoFile(args.source, args.interval)

    print("Running. Click the video window and press 'q' to quit.")
    while True:
        t0 = time.time()
        frame = src.read()
        if frame is None:
            print("No more frames.")
            break
        out = process(frame, head_model, sam, sleeve, args)
        print(f"cycle took {time.time() - t0:.1f}s")
        cv2.imshow("Classroom counter (press q to quit)", out)
        wait_ms = max(1, int((args.interval - (time.time() - t0)) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()