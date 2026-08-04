"""
motion.py - Analyzes motion intensity across the video using OpenCV frame
differencing, then classifies short windows as:
  - "slomo"  : high-motion / high-impact moment -> good for slow-mo
  - "fastmo" : low-content / low-motion moment -> good for a speed-up
  - "normal" : leave at normal speed

This is a real per-frame analysis, not a random/template guess: we sample
motion energy across the clip, then use percentile thresholds so it adapts
to that specific video's own range of motion rather than a fixed number.
"""
import cv2
import numpy as np


def analyze_motion(video_path, window_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    window_frames = max(1, int(window_seconds * fps))

    prev_gray = None
    motion_scores = []  # one score per frame (mean abs diff)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))  # downsample for speed
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
        else:
            motion_scores.append(0.0)
        prev_gray = gray

    cap.release()

    if not motion_scores:
        return [], duration

    # Aggregate into windows
    windows = []
    for i in range(0, len(motion_scores), window_frames):
        chunk = motion_scores[i:i + window_frames]
        if not chunk:
            continue
        avg = float(np.mean(chunk))
        start_t = i / fps
        end_t = min((i + len(chunk)) / fps, duration)
        windows.append({"start": start_t, "end": end_t, "score": avg})

    return windows, duration


def classify_windows(windows, slomo_percentile=80, fastmo_percentile=25):
    """
    Classifies each window as slomo / fastmo / normal using percentile
    thresholds computed from THIS video's own motion distribution, so it
    adapts per clip instead of using a hardcoded magic number.
    """
    if not windows:
        return []

    scores = [w["score"] for w in windows]
    hi_thresh = float(np.percentile(scores, slomo_percentile))
    lo_thresh = float(np.percentile(scores, fastmo_percentile))

    classified = []
    for w in windows:
        if w["score"] >= hi_thresh and hi_thresh > lo_thresh:
            effect = "slomo"
        elif w["score"] <= lo_thresh:
            effect = "fastmo"
        else:
            effect = "normal"
        classified.append({**w, "effect": effect})

    return classified


def merge_adjacent(classified, min_gap=0.3):
    """Merge consecutive windows with the same effect into single segments."""
    if not classified:
        return []
    merged = [dict(classified[0])]
    for w in classified[1:]:
        last = merged[-1]
        if w["effect"] == last["effect"] and w["start"] - last["end"] <= min_gap:
            last["end"] = w["end"]
        else:
            merged.append(dict(w))
    return merged
