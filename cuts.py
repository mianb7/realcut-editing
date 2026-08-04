"""
cuts.py - Detects dead air / silence in raw footage using ffmpeg's silencedetect
filter, and produces a list of (start, end) segments worth KEEPING (i.e. the
silent/dead parts are cut out).
"""
import re
import subprocess


def detect_silence(video_path, noise_db=-30, min_silence_dur=0.6):
    """
    Runs ffmpeg silencedetect on the audio track of video_path.
    Returns a list of (silence_start, silence_end) tuples in seconds.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_dur}",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log = result.stderr

    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", log)]

    # silence_end lines always come with silence_start already emitted first,
    # but if the clip ends mid-silence there may be one extra start with no end.
    silences = list(zip(starts, ends))
    return silences


def get_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def build_keep_segments(video_path, noise_db=-30, min_silence_dur=0.6, padding=0.15):
    """
    Returns a list of (start, end) segments to KEEP, with silent dead-air
    removed. `padding` keeps a small buffer around cuts so words aren't
    clipped mid-syllable.
    """
    duration = get_duration(video_path)
    silences = detect_silence(video_path, noise_db, min_silence_dur)

    keep = []
    cursor = 0.0
    for s_start, s_end in silences:
        cut_start = max(cursor, s_start + padding)
        cut_end = max(cut_start, s_end - padding)
        if cut_start > cursor:
            keep.append((cursor, cut_start))
        cursor = max(cursor, cut_end)

    if cursor < duration:
        keep.append((cursor, duration))

    # Drop segments that are too tiny to matter (ffmpeg trim artifacts)
    keep = [(a, b) for a, b in keep if b - a > 0.05]
    return keep
