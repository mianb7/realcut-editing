"""
audio_clean.py - Makes phone-recorded audio sound like it was recorded
properly. This is the single most noticeable quality jump for most footage:
viewers forgive shaky video far more readily than muddy, quiet, echoey audio.

Chain (in order, and the order matters):
1. highpass  - removes low rumble (traffic, aircon, handling noise, wind)
2. afftdn    - FFT denoiser, strips steady background hiss/hum
3. lowpass   - trims harsh hiss above the range of speech
4. compand   - gentle compression, lifts quiet speech, tames loud peaks
5. loudnorm  - EBU R128 normalization to broadcast/social loudness standard
               (-16 LUFS is the target most social platforms expect, so the
               clip doesn't play noticeably quieter than everything else in
               someone's feed)
"""

# Strength presets so a template can pick how aggressive to be. Heavier
# denoising can make speech sound slightly processed, so "light" exists for
# footage that already has decent audio.
NOISE_PROFILES = {
    "off": None,
    "light": {"highpass": 80, "denoise_db": 6, "lowpass": 12000},
    "standard": {"highpass": 100, "denoise_db": 12, "lowpass": 10000},
    "heavy": {"highpass": 120, "denoise_db": 20, "lowpass": 8000},
}


def build_audio_filter(profile="standard", normalize=True, target_lufs=-16):
    """Returns an ffmpeg audio filter chain string, or None if fully off."""
    stages = []

    settings = NOISE_PROFILES.get(profile)
    if settings:
        stages.append(f"highpass=f={settings['highpass']}")
        stages.append(f"afftdn=nr={settings['denoise_db']}:nf=-25")
        stages.append(f"lowpass=f={settings['lowpass']}")
        # Gentle compression: lifts quiet speech without pumping
        stages.append(
            "compand=attacks=0.02:decays=0.3:"
            "points=-70/-70|-40/-20|-20/-10|0/-4:soft-knee=6"
        )

    if normalize:
        stages.append(f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11")

    if not stages:
        return None
    return ",".join(stages)
