"""
gpu_enhance.py - Automated GPU enhancement (upscaling + face restoration).

This replaces the manual Colab step. Colab cannot be automated - Google
blocks it, it needs a live browser session, and it disconnects when idle.
For unattended processing you need a hosted GPU API, which costs a few pence
per video.

Currently supports Replicate, because the models we want (Real-ESRGAN,
GFPGAN) are already hosted there as callable endpoints - no deploying or
managing a GPU box ourselves.

SETUP:
  1. Sign up at replicate.com
  2. Copy your API token from replicate.com/account/api-tokens
  3. In Railway: Variables tab -> add REPLICATE_API_TOKEN = your token

If no token is set, this module cleanly does nothing and the pipeline runs
without GPU enhancement - so the site never breaks just because the token is
missing or credit runs out.
"""
import os
import time

import requests

REPLICATE_API = "https://api.replicate.com/v1/predictions"

# Model versions on Replicate. These are pinned hashes - Replicate requires
# an exact version, and pinning means a model update upstream can't silently
# change our output.
MODELS = {
    "face_restore": "9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",  # GFPGAN
    "upscale": "42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b",      # Real-ESRGAN
}


def is_available():
    """True if a GPU API token is configured."""
    return bool(os.environ.get("REPLICATE_API_TOKEN"))


def _poll_prediction(prediction_url, headers, timeout=600, progress_callback=None):
    """Waits for a prediction to finish. Returns the output URL or raises."""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(prediction_url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")

        if status == "succeeded":
            return data.get("output")
        if status in ("failed", "canceled"):
            raise RuntimeError(f"GPU job {status}: {data.get('error')}")

        if progress_callback:
            elapsed = time.time() - start
            # Rough progress signal - the API doesn't report percentage
            progress_callback(min(90, int(elapsed / timeout * 100)))
        time.sleep(3)

    raise TimeoutError("GPU enhancement timed out")


def enhance_image(image_url_or_path, mode="face_restore", scale=2,
                  progress_callback=None):
    """
    Runs one image through GPU enhancement. Returns the URL of the result.

    Note this works on IMAGES. Video enhancement means extracting frames,
    enhancing each, and rebuilding - which at a few pence per frame becomes
    expensive fast. See enhance_video_sampled() for the practical approach.
    """
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }

    if mode == "face_restore":
        payload = {
            "version": MODELS["face_restore"],
            "input": {"img": image_url_or_path, "version": "v1.4", "scale": scale},
        }
    else:
        payload = {
            "version": MODELS["upscale"],
            "input": {"image": image_url_or_path, "scale": scale, "face_enhance": False},
        }

    r = requests.post(REPLICATE_API, json=payload, headers=headers, timeout=30)
    r.raise_for_status()
    prediction = r.json()

    return _poll_prediction(prediction["urls"]["get"], headers,
                            progress_callback=progress_callback)


def estimate_cost(frame_count):
    """
    Rough cost estimate so you can price a job before running it.
    Replicate bills by GPU-second; these models run roughly 1-3 seconds a
    frame on their hardware. Treat this as an order-of-magnitude guide, not
    a quote - check your actual Replicate billing.
    """
    seconds_per_frame = 2.0
    cost_per_second = 0.0005  # approximate, varies by GPU tier
    usd = frame_count * seconds_per_frame * cost_per_second
    return {
        "frames": frame_count,
        "estimated_usd": round(usd, 3),
        "note": "Per-frame video enhancement gets expensive fast. For a "
                "30s/30fps clip that's ~900 frames. Consider enhancing only "
                "the thumbnail/key frames, or accept the cost per order.",
    }
