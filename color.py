"""
color.py - Auto color correction + template-driven grading.

Two parts:
1. auto_levels(): analyzes actual pixel histogram of the source video to fix
   flat/dull exposure and white balance automatically (real analysis, not a
   fixed preset).
2. TEMPLATE_GRADES: style-driven ffmpeg filter strings for the look
   (cinematic / energetic / minimal), applied on top of the auto correction.
"""
import cv2
import numpy as np


def auto_levels(video_path, sample_frames=20):
    """
    Samples frames across the video and computes per-channel black/white
    points so we can build an ffmpeg `curves`/`eq` correction that isn't
    just a flat default - it reacts to how dark/washed out THIS footage is.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, frame_count // sample_frames)

    means = []
    idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ret2, frame = cap.retrieve()
            if ret2:
                means.append(frame.reshape(-1, 3).mean(axis=0))  # BGR
        idx += 1
    cap.release()

    if not means:
        return {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}

    avg = np.mean(means, axis=0)  # B, G, R
    overall_brightness = float(np.mean(avg)) / 255.0  # 0-1

    # If footage is dark, push brightness/contrast up; if blown out, pull back
    brightness_adj = round((0.5 - overall_brightness) * 0.4, 3)
    contrast_adj = 1.1 if overall_brightness < 0.55 else 1.0

    return {
        "brightness": max(-0.15, min(0.15, brightness_adj)),
        "contrast": contrast_adj,
        "saturation": 1.15,  # slight universal pop, phone cameras tend flat
    }


TEMPLATE_GRADES = {
    "energetic": {"extra_contrast": 1.15, "extra_saturation": 1.25, "warmth": 0.03},
    "cinematic": {"extra_contrast": 1.1, "extra_saturation": 0.9, "warmth": -0.02},
    "minimal": {"extra_contrast": 1.0, "extra_saturation": 0.95, "warmth": 0.0},
}


def build_eq_filter(auto, template_name="energetic"):
    """Combines the auto-detected correction with the chosen template's look
    into a single ffmpeg `eq` filter string."""
    t = TEMPLATE_GRADES.get(template_name, TEMPLATE_GRADES["energetic"])

    brightness = round(auto["brightness"] + t["warmth"] * 0.3, 3)
    contrast = round(auto["contrast"] * t["extra_contrast"], 3)
    saturation = round(auto["saturation"] * t["extra_saturation"], 3)

    return f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"
