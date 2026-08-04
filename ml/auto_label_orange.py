#!/usr/bin/env python3
import colorsys
import os
import subprocess
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO = Path("/Users/bytedance/Downloads/飞书20260804-120018.mp4")
FRAME_DIR = ROOT / "frames"
LABEL_DIR = ROOT / "labels"
PREVIEW_DIR = ROOT / "preview"
WIDTH = 540
HEIGHT = 960
FPS = 2


def is_can_orange(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    return 5.0 <= hue <= 32.0 and s >= 0.42 and v >= 0.42 and r > g * 1.18 and r > b * 1.55


def components(mask, width, height):
    seen = bytearray(width * height)
    result = []
    for y in range(height):
        row = y * width
        for x in range(width):
            idx = row + x
            if seen[idx] or not mask[idx]:
                continue
            q = deque([(x, y)])
            seen[idx] = 1
            min_x = max_x = x
            min_y = max_y = y
            count = 0
            while q:
                cx, cy = q.popleft()
                count += 1
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    nidx = ny * width + nx
                    if not seen[nidx] and mask[nidx]:
                        seen[nidx] = 1
                        q.append((nx, ny))
            result.append((min_x, min_y, max_x + 1, max_y + 1, count))
    return result


def pick_can_box(frame):
    mask = bytearray(WIDTH * HEIGHT)
    for i in range(WIDTH * HEIGHT):
        base = i * 3
        if is_can_orange(frame[base], frame[base + 1], frame[base + 2]):
            mask[i] = 1

    best = None
    best_score = 0.0
    for x1, y1, x2, y2, count in components(mask, WIDTH, HEIGHT):
        w = x2 - x1
        h = y2 - y1
        if w < 18 or h < 45:
            continue
        area_ratio = count / float(WIDTH * HEIGHT)
        if area_ratio < 0.0015 or area_ratio > 0.08:
            continue
        aspect = h / float(max(w, 1))
        if aspect < 1.15 or aspect > 5.8:
            continue
        score = count * min(aspect, 3.0)
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    if best is None:
        return None

    x1, y1, x2, y2 = best
    w = x2 - x1
    h = y2 - y1
    # Orange label is the can body; expand to include top rim and bottom.
    x1 = max(0, int(x1 - w * 0.18))
    x2 = min(WIDTH, int(x2 + w * 0.18))
    y1 = max(0, int(y1 - h * 0.32))
    y2 = min(HEIGHT, int(y2 + h * 0.18))
    return x1, y1, x2, y2


def write_label(path, box):
    if box is None:
        path.write_text("")
        return
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / WIDTH
    cy = ((y1 + y2) / 2.0) / HEIGHT
    bw = (x2 - x1) / WIDTH
    bh = (y2 - y1) / HEIGHT
    path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def draw_box(frame, box):
    if box is None:
        return frame
    frame = bytearray(frame)
    x1, y1, x2, y2 = box
    for x in range(x1, x2):
        for y in (y1, y2 - 1):
            set_pixel(frame, x, y, 0, 255, 0)
    for y in range(y1, y2):
        for x in (x1, x2 - 1):
            set_pixel(frame, x, y, 0, 255, 0)
    return bytes(frame)


def set_pixel(frame, x, y, r, g, b):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        idx = (y * WIDTH + x) * 3
        frame[idx] = r
        frame[idx + 1] = g
        frame[idx + 2] = b


def write_ppm(path, frame):
    path.write_bytes(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode() + frame)


def main():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(VIDEO),
        "-vf",
        f"fps={FPS}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    frame_size = WIDTH * HEIGHT * 3
    total = len(raw) // frame_size
    labeled = 0
    for index in range(total):
        frame = raw[index * frame_size : (index + 1) * frame_size]
        box = pick_can_box(frame)
        name = f"frame_{index + 1:04d}"
        write_label(LABEL_DIR / f"{name}.txt", box)
        if box is not None:
            labeled += 1
        write_ppm(PREVIEW_DIR / f"{name}.ppm", draw_box(frame, box))
    print(f"frames={total} labeled={labeled}")


if __name__ == "__main__":
    main()
