"""
reframe.py - Converts landscape (or any) footage into vertical 9:16 social
format WITHOUT just center-cropping and cutting the subject's head off.

How it works (this is the part that makes it worth paying for):
1. Samples frames across the video and runs real face detection on each.
2. Builds a track of where the subject actually is over time.
3. Smooths that track so the crop glides instead of jittering frame-to-frame.
4. Emits an ffmpeg crop expression that follows the subject.

If no face is found (product shots, scenery, pets), it falls back to
motion-weighted centering - it crops toward wherever the action is, rather
than blindly taking the middle.
"""
import subprocess

import cv2
import numpy as np


def _detect_face_track(video_path, sample_fps=4):
    """Returns a list of (timestamp, center_x_fraction) where a face was
    found. center_x_fraction is 0.0 (far left) to 1.0 (far right)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    step = max(1, int(fps / sample_fps))

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    track = []
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
                if len(faces) > 0:
                    # Largest face = the subject
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    center_x = (x + w / 2) / small.shape[1]
                    track.append((idx / fps, float(center_x)))
        idx += 1
    cap.release()
    return track, width


def _motion_center_fallback(video_path, sample_fps=3):
    """When no face is present, find the horizontal band with the most
    motion and center on that instead of blindly using the middle."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / sample_fps))

    prev = None
    column_energy = None
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    diff = cv2.absdiff(gray, prev)
                    col = diff.mean(axis=0)  # motion per column
                    column_energy = col if column_energy is None else column_energy + col
                prev = gray
        idx += 1
    cap.release()

    if column_energy is None or column_energy.sum() == 0:
        return 0.5
    # Weighted centroid of motion across the width
    xs = np.arange(len(column_energy))
    centroid = float((xs * column_energy).sum() / column_energy.sum())
    return centroid / len(column_energy)


def _smooth_track(track, duration, smoothing_window=5):
    """Smooths the raw per-sample centers so the crop moves like a slow
    camera pan instead of snapping around."""
    if not track:
        return []
    centers = np.array([c for _, c in track])
    if len(centers) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        centers = np.convolve(centers, kernel, mode="same")
        # convolve edges get dragged toward 0; repair with nearest valid
        half = smoothing_window // 2
        if len(centers) > 2 * half:
            centers[:half] = centers[half]
            centers[-half:] = centers[-half - 1]
    return [(track[i][0], float(centers[i])) for i in range(len(track))]


def build_crop_filter(video_path, target_ratio=(9, 16), max_keyframes=40):
    """
    Returns an ffmpeg filter string that crops to vertical while following
    the subject, or None if the source is already vertical/narrower.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()

    target_w_ratio, target_h_ratio = target_ratio
    desired_width = int(height * target_w_ratio / target_h_ratio)

    if desired_width >= width:
        # Source is already as narrow as (or narrower than) the target.
        return None, {"reframed": False, "reason": "source already vertical"}

    # Make even (h264 requires even dimensions)
    if desired_width % 2 != 0:
        desired_width -= 1

    track, _ = _detect_face_track(video_path)
    used_faces = len(track) >= 3

    if used_faces:
        smoothed = _smooth_track(track, duration)
        # Downsample to a manageable number of keyframes for the expression
        if len(smoothed) > max_keyframes:
            stride = len(smoothed) // max_keyframes
            smoothed = smoothed[::stride]

        max_x = width - desired_width
        # Build a piecewise-linear x position over time
        pieces = []
        for i in range(len(smoothed) - 1):
            t0, c0 = smoothed[i]
            t1, c1 = smoothed[i + 1]
            x0 = max(0, min(max_x, c0 * width - desired_width / 2))
            x1 = max(0, min(max_x, c1 * width - desired_width / 2))
            if t1 <= t0:
                continue
            slope = (x1 - x0) / (t1 - t0)
            pieces.append((t0, t1, x0, slope))

        if pieces:
            expr = f"{pieces[-1][2]:.1f}"  # default = last position
            for t0, t1, x0, slope in reversed(pieces):
                expr = f"if(lt(t,{t1:.3f}),{x0:.1f}+({slope:.2f})*(t-{t0:.3f}),{expr})"
            crop_filter = f"crop={desired_width}:{height}:x='{expr}':y=0"
            return crop_filter, {
                "reframed": True,
                "method": "face-tracked",
                "face_samples": len(track),
                "output_size": f"{desired_width}x{height}",
            }

    # Fallback: static crop centered on where the motion actually is
    center_fraction = _motion_center_fallback(video_path)
    x = int(max(0, min(width - desired_width, center_fraction * width - desired_width / 2)))
    crop_filter = f"crop={desired_width}:{height}:{x}:0"
    return crop_filter, {
        "reframed": True,
        "method": "motion-centered",
        "output_size": f"{desired_width}x{height}",
    }
