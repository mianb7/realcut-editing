"""
captions.py - Transcribes the (post-cut) video's speech to timed captions
using faster-whisper, and writes an .srt file ffmpeg can burn in.

NOTE: this needs to download a small speech-to-text model on first run
(one-time, then cached). That requires normal internet access on whatever
server this runs on in production - it does NOT work inside this sandboxed
build/test environment (network is locked to package registries only), so
this module can't be exercised end-to-end here. The code itself is real and
will run once deployed to a normal server (Netlify function, VPS, etc).
"""
import os


def transcribe_to_srt(audio_or_video_path, srt_out_path, model_size="small"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        )

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio_or_video_path, beam_size=5)

    def fmt_ts(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")

    return srt_out_path


def has_whisper_available():
    try:
        import faster_whisper  # noqa
        return True
    except ImportError:
        return False
