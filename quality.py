"""
quality.py - Resolution handling and encoding quality.

HONEST FRAMING - read this before assuming what it does:

There are two very different things people mean by "make it 4K":

  (a) AI upscaling that INVENTS detail that was never captured
      (Real-ESRGAN, Topaz). This genuinely reconstructs plausible detail
      but needs a GPU. On our CPU-only server it would take minutes per
      second of footage. Not viable here.

  (b) High-quality mathematical upscaling + detail enhancement + high
      bitrate encoding. This does NOT invent detail. What it DOES do is
      real and worth having:
        - Lanczos scaling is genuinely sharper than the default bilinear
        - Most "low quality" in phone video is COMPRESSION damage, not
          resolution - phones encode at low bitrates and re-encoding at
          high bitrate stops further degradation
        - Detail enhancement recovers apparent sharpness lost to that
          compression
        - Outputting at a platform's native resolution stops the platform
          re-encoding and degrading it again

This module does (b), honestly. A 1080p source upscaled here becomes a
genuine 4K FILE that looks cleaner and sharper - but it is not the same as
true 4K capture, and this code does not pretend otherwise.
"""

RESOLUTION_PRESETS = {
    "source": None,
    "hd": (1280, 720),
    "fullhd": (1920, 1080),
    "4k": (3840, 2160),
}

# Bitrate targets (kbps) that preserve quality rather than just inflating
# file size. These are chosen around what the platforms actually accept
# before re-compressing.
BITRATE_TARGETS = {
    "hd": "8M",
    "fullhd": "16M",
    "4k": "45M",
}


def get_output_dimensions(src_width, src_height, target="fullhd", vertical=False):
    """
    Works out the output size, preserving aspect ratio and keeping both
    dimensions even (h264 requirement).
    """
    if target == "source" or target not in RESOLUTION_PRESETS:
        return src_width, src_height

    preset_w, preset_h = RESOLUTION_PRESETS[target]

    if vertical or src_height > src_width:
        # Vertical: the preset's smaller number becomes the width
        out_h = preset_h if preset_h > preset_w else preset_w
        out_w = int(out_h * src_width / src_height)
    else:
        out_w = preset_w if preset_w > preset_h else preset_h
        out_h = int(out_w * src_height / src_width)

    out_w -= out_w % 2
    out_h -= out_h % 2
    return out_w, out_h


def build_scale_filter(src_width, src_height, out_width, out_height,
                       enhance_detail=True):
    """
    Lanczos scaling plus detail enhancement. Lanczos is used specifically
    because it preserves edge definition far better than the default when
    scaling up; the unsharp pass afterwards restores micro-contrast that
    both compression and scaling soften.
    """
    stages = []

    if (out_width, out_height) != (src_width, src_height):
        stages.append(f"scale={out_width}:{out_height}:flags=lanczos")

        if enhance_detail:
            upscaling = out_width > src_width
            if upscaling:
                # Stronger recovery when upscaling, since scaling up softens
                stages.append("unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=1.0")
            else:
                stages.append("unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount=0.5")

    return ",".join(stages) if stages else None


def build_encode_args(target="fullhd", high_quality=True):
    """
    Encoding settings. CRF 18 is visually near-lossless; combined with a
    generous maxrate this stops the output being the weak link in the chain.
    'slow' preset costs more processing time but produces noticeably better
    quality at the same bitrate than 'fast'.
    """
    bitrate = BITRATE_TARGETS.get(target, "16M")

    if high_quality:
        return [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-maxrate", bitrate,
            "-bufsize", bitrate,
            "-pix_fmt", "yuv420p",      # maximum player/platform compatibility
            "-profile:v", "high",
            "-movflags", "+faststart",  # starts playing before fully downloaded
            "-c:a", "aac",
            "-b:a", "192k",
        ]
    return [
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
    ]
