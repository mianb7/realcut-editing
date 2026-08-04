"""
cinematic.py - Real film-emulation colour grading, plus image cleanup.

This is NOT "apply a filter". The pipeline is:
  1. Analyse the actual footage - how dark, how flat, how noisy, what the
     colour cast is, whether it's indoor/outdoor.
  2. Decide the correction needed for THAT footage.
  3. Apply a filmic look on top: S-curve contrast, split toning
     (cool shadows / warm highlights - the "teal & orange" look every
     colourist uses), highlight rolloff so bright areas don't clip harshly,
     controlled saturation, grain, and a subtle vignette.

Why these specific choices: phone video is recorded flat and bright with
crushed blacks and clipped highlights. Film looks different because of three
things - lifted blacks with a colour tint, smooth highlight rolloff, and
restrained saturation with hue separation between shadows and highlights.
Reproducing those three is what actually reads as "cinematic" to the eye.
"""
import subprocess

import cv2
import numpy as np


def analyse_footage(video_path, sample_frames=24):
    """Real measurement of the source, not guesswork. Returns the stats the
    grading decisions are based on."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, frame_count // sample_frames)

    brightness_vals, contrast_vals, sat_vals = [], [], []
    b_means, g_means, r_means = [], [], []
    noise_vals = []

    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok2, frame = cap.retrieve()
            if ok2:
                small = cv2.resize(frame, (240, 135))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

                brightness_vals.append(float(gray.mean()) / 255.0)
                contrast_vals.append(float(gray.std()) / 128.0)
                sat_vals.append(float(hsv[:, :, 1].mean()) / 255.0)

                b, g, r = small[:, :, 0].mean(), small[:, :, 1].mean(), small[:, :, 2].mean()
                b_means.append(float(b)); g_means.append(float(g)); r_means.append(float(r))

                # Noise estimate: high-frequency energy in flat areas
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                noise_vals.append(float(np.mean(np.abs(gray.astype(float) - blur.astype(float)))))
        idx += 1
    cap.release()

    if not brightness_vals:
        return None

    avg_b, avg_g, avg_r = np.mean(b_means), np.mean(g_means), np.mean(r_means)
    total = avg_b + avg_g + avg_r
    # Warm/cool cast: >0 means the footage leans warm (orange/tungsten),
    # <0 means it leans cool (blue/daylight-shade)
    colour_cast = float((avg_r - avg_b) / total) if total else 0.0

    return {
        "brightness": float(np.mean(brightness_vals)),
        "contrast": float(np.mean(contrast_vals)),
        "saturation": float(np.mean(sat_vals)),
        "colour_cast": colour_cast,
        "noise": float(np.mean(noise_vals)),
    }


# Film-look presets. Each defines the *character* of the grade; the actual
# numbers get adapted per-video by build_cinematic_chain() using the analysis.
LOOKS = {
    "cinematic": {
        "shadow_tint": (-0.030, -0.010, 0.045),   # R,G,B lift -> cool shadows
        "highlight_tint": (0.040, 0.012, -0.030),  # warm highlights
        "black_lift": 0.030,      # film never has pure black
        "highlight_rolloff": 0.92,
        "target_saturation": 1.02,
        "grain": 3.5,
        "vignette": True,
        "curve_strength": 0.80,
    },
    "warm_film": {
        "shadow_tint": (0.010, -0.005, 0.020),
        "highlight_tint": (0.055, 0.020, -0.040),
        "black_lift": 0.035,
        "highlight_rolloff": 0.90,
        "target_saturation": 1.10,
        "grain": 4.0,
        "vignette": True,
        "curve_strength": 0.70,
    },
    "clean_commercial": {
        "shadow_tint": (-0.010, 0.000, 0.015),
        "highlight_tint": (0.015, 0.008, -0.008),
        "black_lift": 0.015,
        "highlight_rolloff": 0.96,
        "target_saturation": 1.12,
        "grain": 1.5,
        "vignette": False,
        "curve_strength": 0.55,
    },
    "moody": {
        "shadow_tint": (-0.040, -0.015, 0.060),
        "highlight_tint": (0.030, 0.005, -0.020),
        "black_lift": 0.045,
        "highlight_rolloff": 0.86,
        "target_saturation": 0.88,
        "grain": 5.0,
        "vignette": True,
        "curve_strength": 0.95,
    },
}


def _build_curve(black_lift, rolloff, strength):
    """
    Builds the filmic S-curve as an ffmpeg `curves` control-point string.

    The three things that make it filmic:
      - blacks lifted off zero (film base never reaches pure black)
      - midtone contrast increased (the S bend)
      - highlights compressed toward a shoulder instead of clipping flat
    """
    s = strength
    pts = [
        (0.00, round(black_lift, 4)),
        (0.15, round(0.15 - 0.030 * s + black_lift * 0.6, 4)),
        (0.35, round(0.35 - 0.035 * s + black_lift * 0.3, 4)),
        (0.50, round(0.50, 4)),
        (0.70, round(0.70 + 0.030 * s, 4)),
        (0.85, round(min(0.97, 0.85 + 0.025 * s), 4)),
        (1.00, round(rolloff, 4)),
    ]
    return " ".join(f"{x}/{y}" for x, y in pts)


def build_cinematic_chain(video_path, look="cinematic", denoise=True,
                          sharpen=True, letterbox=False):
    """
    Returns (filter_string, analysis, decisions) - a full cinematic grade
    adapted to this specific footage.
    """
    stats = analyse_footage(video_path)
    preset = LOOKS.get(look, LOOKS["cinematic"])
    stages = []
    decisions = {}

    if stats is None:
        return None, None, {"error": "could not analyse footage"}

    # ---- 1. CLEAN: denoise before grading, so grading doesn't amplify noise
    if denoise and stats["noise"] > 1.5:
        # Strength scales with how noisy the footage actually is
        strength = min(4.0, max(1.0, stats["noise"] / 2.0))
        stages.append(f"hqdn3d={strength:.1f}:{strength*0.75:.1f}:{strength*1.5:.1f}:{strength*1.5:.1f}")
        decisions["denoise"] = round(strength, 2)

    # ---- 2. EXPOSURE + WHITE BALANCE correction, based on measurement
    # Push dark footage up, pull blown-out footage down.
    brightness_adj = round((0.46 - stats["brightness"]) * 0.45, 4)
    brightness_adj = max(-0.14, min(0.16, brightness_adj))
    # Flat footage (low contrast) needs more contrast restored
    contrast_adj = round(1.0 + max(0.0, (0.22 - stats["contrast"])) * 1.1, 3)
    contrast_adj = min(1.30, contrast_adj)
    decisions["exposure_correction"] = brightness_adj
    decisions["contrast_restore"] = contrast_adj

    # Neutralise a strong colour cast before applying the intended look,
    # otherwise the look fights the cast instead of sitting on top of it.
    cast = stats["colour_cast"]
    cast_correction = -cast * 0.5
    decisions["colour_cast_detected"] = round(cast, 4)

    if abs(brightness_adj) > 0.005 or contrast_adj > 1.005:
        stages.append(f"eq=brightness={brightness_adj}:contrast={contrast_adj}")

    # ---- 3. FILMIC CURVE: the core of the cinematic look
    curve = _build_curve(preset["black_lift"], preset["highlight_rolloff"],
                         preset["curve_strength"])
    stages.append(f"curves=all='{curve}'")
    decisions["filmic_curve"] = curve

    # ---- 4. SPLIT TONING: cool shadows, warm highlights (teal & orange)
    sr, sg, sb = preset["shadow_tint"]
    hr, hg, hb = preset["highlight_tint"]
    sr += cast_correction * 0.5
    hr += cast_correction * 0.5
    sb -= cast_correction * 0.5
    hb -= cast_correction * 0.5
    stages.append(
        f"colorbalance=rs={sr:.3f}:gs={sg:.3f}:bs={sb:.3f}:"
        f"rh={hr:.3f}:gh={hg:.3f}:bh={hb:.3f}"
    )
    decisions["split_toning"] = {"shadows": [round(sr,3), round(sg,3), round(sb,3)],
                                  "highlights": [round(hr,3), round(hg,3), round(hb,3)]}

    # ---- 5. SATURATION: shaped, not cranked. Over-saturation is the #1
    # giveaway of amateur grading, so we target a level rather than add.
    sat_target = preset["target_saturation"]
    if stats["saturation"] < 0.25:      # washed out source needs more help
        sat_target += 0.10
    elif stats["saturation"] > 0.50:    # already vivid, pull back
        sat_target -= 0.08
    stages.append(f"eq=saturation={sat_target:.3f}")
    decisions["saturation_target"] = round(sat_target, 3)

    # ---- 6. SHARPEN: restore micro-detail lost to denoising/compression
    if sharpen:
        stages.append("unsharp=5:5:0.7:5:5:0.0")
        decisions["sharpen"] = True

    # ---- 7. GRAIN: subtle film grain unifies the image and hides
    # compression blocking. Real film has it; digital video's total absence
    # of it is part of why phone video reads as "cheap".
    if preset["grain"] > 0:
        stages.append(f"noise=alls={preset['grain']:.0f}:allf=t+u")
        decisions["grain"] = preset["grain"]

    # ---- 8. VIGNETTE: pulls the eye to the centre of frame
    if preset["vignette"]:
        stages.append("vignette=PI/5:mode=forward")
        decisions["vignette"] = True

    # ---- 9. LETTERBOX: 2.39:1 cinema bars, optional
    if letterbox:
        stages.append("pad=iw:ih:0:0:black")  # placeholder, applied downstream
        decisions["letterbox"] = True

    return ",".join(stages), stats, decisions
