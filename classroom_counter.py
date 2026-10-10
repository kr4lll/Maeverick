"""
Classroom counter: head detection -> torso box -> sleeve type (no SAM, no training).
CPU only. Runs the whole pipeline about once per second.

  python classroom_counter.py                       # webcam
  python classroom_counter.py --source test.jpg     # photo
  python classroom_counter.py --source clip.mp4     # video
Press 'q' in the video window to quit.
"""
import argparse
import threading
import time

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

from sleeve_utils import SleeveParser

COLORS = {"long": (0, 200, 0), "short": (0, 140, 255), "unsure": (170, 170, 170)}
SHOW = {"long": "long", "short": "short", "unsure": "short?"}   # unsure counts as short
MIN_CROP = 24      # torso crops smaller than this (pixels) are too tiny to read


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


# ---------------------------------------------------------------- torso box logic
def ideal_torso(head):
    """Expected upper-body box under a head: shoulders down to ~2.5 head-heights."""
    x1, y1, x2, y2 = head
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2
    return cx - 1.3 * w, y2, cx + 1.3 * w, y2 + 2.5 * h


def prepare_torso(heads, i, rgb, W, H):
    """Returns (crop or None, visible_fraction).
    - Area outside the image counts as hidden.
    - Heads lower in the image (nearer to a front camera) block this person's torso;
      the area under them is greyed out and counted as hidden."""
    head = heads[i]
    ix1, iy1, ix2, iy2 = ideal_torso(head)
    ideal_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    bx1, by1 = int(max(0, ix1)), int(max(0, iy1))
    bx2, by2 = int(min(W, ix2)), int(min(H, iy2))
    if bx2 - bx1 < MIN_CROP or by2 - by1 < MIN_CROP:
        return None, 0.0

    blocked = np.zeros((by2 - by1, bx2 - bx1), bool)
    hh = head[3] - head[1]
    cy = (head[1] + head[3]) / 2
    for j, o in enumerate(heads):
        if j == i:
            continue
        ocy = (o[1] + o[3]) / 2
        if ocy > cy + 0.6 * hh:                      # clearly nearer to the camera
            ow = o[2] - o[0]
            ocx = (o[0] + o[2]) / 2
            rx1 = int(max(bx1, ocx - 1.3 * ow)); rx2 = int(min(bx2, ocx + 1.3 * ow))
            ry1 = int(max(by1, o[1]));            ry2 = by2
            if rx2 > rx1 and ry2 > ry1:
                blocked[ry1 - by1:ry2 - by1, rx1 - bx1:rx2 - bx1] = True

    visible = float((~blocked).sum()) / ideal_area
    crop = rgb[by1:by2, bx1:bx2].copy()
    crop[blocked] = 128
    return crop, visible


def decide(stats, visible, args):
    """Returns (label, score, note). 'unsure' is shown/counted as short."""
    if stats is None:
        return "unsure", 0.0, "too small"
    if visible < args.min_visible:
        return "unsure", visible, f"hidden {visible:.0%}"
    clothes = (stats["upper"] + stats["arm"]) / stats["content"]
    if clothes < args.min_clothes:
        return "unsure", clothes, "no shirt seen"
    arm_ratio = stats["arm"] / max(1, stats["arm"] + stats["upper"])
    return ("short" if arm_ratio >= args.arm_ratio else "long"), arm_ratio, ""


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- drawing
def draw(frame, entries):
    out = frame.copy()
    H, W = frame.shape[:2]
    for e in entries:
        col = COLORS[e["label"]]
        tx1, ty1, tx2, ty2 = [int(v) for v in ideal_torso(e["box"])]
        cv2.rectangle(out, (max(0, tx1), max(0, ty1)), (min(W - 1, tx2), min(H - 1, ty2)), col, 1)
        x1, y1, x2, y2 = [int(v) for v in e["box"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        text = SHOW[e["label"]] + (f" {e['score']:.2f}" if e["label"] != "unsure" else "")
        if e["note"]:
            text += f" ({e['note']})"
        cv2.putText(out, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    n_long = sum(e["label"] == "long" for e in entries)
    n_short = sum(e["label"] == "short" for e in entries)
    n_assumed = sum(e["label"] == "unsure" for e in entries)
    lines = [f"Students (heads): {len(entries)}",
             f"Long sleeve: {n_long}",
             f"Short sleeve: {n_short + n_assumed} (assumed: {n_assumed})"]
    for i, t in enumerate(lines):
        cv2.putText(out, t, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, t, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------- main loop
def process(frame, head_model, parser, cache, args):
    H, W = frame.shape[:2]
    heads = detect_heads(head_model, frame, args.conf, args.imgsz, args.tiles)
    print(f"heads found: {len(heads)}")
    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Reuse cached results for heads that haven't moved (classroom = mostly static)
    entries, todo, used = [None] * len(heads), [], set()
    for i, h in enumerate(heads):
        best, bj = 0.0, -1
        for j, c in enumerate(cache):
            if j in used:
                continue
            v = iou(h, c["box"])
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= 0.5 and now - cache[bj]["time"] < args.recheck_seconds:
            used.add(bj)
            e = dict(cache[bj])
            e["box"] = h
            entries[i] = e
        else:
            todo.append(i)

    # Run the sleeve parser only on new / changed / expired heads
    if todo:
        jobs = []
        for i in todo:
            crop, visible = prepare_torso(heads, i, rgb, W, H)
            if crop is not None and visible >= args.min_visible:
                jobs.append((i, crop, visible))
            else:
                label, score, note = decide(None if crop is None else {"upper": 0, "arm": 0, "content": 1},
                                            visible, args) if crop is not None else ("unsure", 0.0, "too small")
                entries[i] = {"box": heads[i], "label": label, "score": score, "note": note, "time": now}
        if jobs:
            stats = parser.parse([c for _, c, _ in jobs])
            for (i, _, visible), st in zip(jobs, stats):
                label, score, note = decide(st, visible, args)
                entries[i] = {"box": heads[i], "label": label, "score": score, "note": note, "time": now}
        print(f"re-analysed {len(todo)} head(s), reused {len(heads) - len(todo)}")

    return draw(frame, entries), entries


def shrink(frame, max_width):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    s = max_width / w
    return cv2.resize(frame, (max_width, int(h * s)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 = webcam, or photo/video path")
    ap.add_argument("--head-weights", default="medium.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="head detection threshold")
    ap.add_argument("--imgsz", type=int, default=960, help="head detector input size")
    ap.add_argument("--tiles", type=int, default=2, help="1 = off, 2 = 2x2, 3 = 3x3")
    ap.add_argument("--min-visible", type=float, default=0.20,
                    help="min fraction of torso box not blocked by people in front")
    ap.add_argument("--min-clothes", type=float, default=0.10,
                    help="min fraction of the crop that must be shirt/arm pixels")
    ap.add_argument("--arm-ratio", type=float, default=0.12,
                    help="bare-arm share at/above which a person counts as short sleeve")
    ap.add_argument("--parse-size", type=int, default=384, help="SegFormer input size")
    ap.add_argument("--recheck-seconds", type=float, default=30.0)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    print("Loading head detector...")
    head_model = YOLO(args.head_weights)
    parser = SleeveParser(args.parse_size)

    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = shrink(cv2.imread(args.source), args.max_width)
        out, _ = process(frame, head_model, parser, [], args)
        cv2.imshow("Classroom counter (any key to close)", out)
        cv2.waitKey(0)
        return

    if args.source.isdigit():
        src = WebcamLatest(int(args.source))
        time.sleep(1.0)
    else:
        src = VideoFile(args.source, args.interval)

    cache = []
    print("Running. Click the video window and press 'q' to quit.")
    while True:
        t0 = time.time()
        frame = src.read()
        if frame is None:
            print("No more frames.")
            break
        out, cache = process(shrink(frame, args.max_width), head_model, parser, cache, args)
        print(f"cycle took {time.time() - t0:.1f}s")
        cv2.imshow("Classroom counter (press q to quit)", out)
        wait_ms = max(1, int((args.interval - (time.time() - t0)) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()