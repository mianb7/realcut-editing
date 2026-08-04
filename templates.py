"""
templates.py - Style presets the customer picks from. Each one is a bundle
of settings that drives every other module (color grade, caption font/style,
how aggressively we apply slow-mo/fast-mo, silence sensitivity).
"""

TEMPLATES = {
    "energetic": {
        "label": "Energetic",
        "description": "Fast cuts, punchy captions, bold slow-mo on peak moments.",
        "color_grade": "energetic",
        "film_look": "clean_commercial",
        "slomo_percentile": 85,   # only the top 15% most intense moments get slow-mo
        "fastmo_percentile": 30,  # bottom 30% gets sped up
        "slomo_speed": 0.5,       # 2x slower
        "fastmo_speed": 1.6,      # 1.6x faster
        "silence_sensitivity": -30,  # dB threshold for cutting dead air
        "transition_duration": 0,  # disabled - see note below
        "transition_style": "fade",
        "ken_burns": False,  # disabled - see note below
        "ken_burns_intensity": 1.06,  # max zoom factor over the segment
        "caption_style": {
            "font": "DejaVu Sans Bold",
            "size": 13,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "position": "bottom",
        },
    },
    "cinematic": {
        "label": "Cinematic",
        "description": "Slower pacing, subtle grade, minimal fast-mo, dramatic slow-mo.",
        "color_grade": "cinematic",
        "film_look": "cinematic",
        "slomo_percentile": 80,
        "fastmo_percentile": 15,
        "slomo_speed": 0.4,
        "fastmo_speed": 1.3,
        "silence_sensitivity": -32,
        "transition_duration": 0,  # disabled - see note in energetic template
        "transition_style": "fade",
        "ken_burns": False,
        "ken_burns_intensity": 1.1,
        "caption_style": {
            "font": "DejaVu Serif",
            "size": 12,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "position": "bottom",
        },
    },
    "minimal": {
        "label": "Clean / Minimal",
        "description": "Tight cuts, no speed ramping, clean simple captions.",
        "color_grade": "minimal",
        "film_look": "clean_commercial",
        "slomo_percentile": 100,  # effectively disables slomo
        "fastmo_percentile": 0,   # effectively disables fastmo
        "slomo_speed": 1.0,
        "fastmo_speed": 1.0,
        "silence_sensitivity": -28,
        "transition_duration": 0,     # hard cuts on purpose - fits the "minimal" feel
        "transition_style": "none",
        "ken_burns": False,
        "ken_burns_intensity": 1.0,
        "caption_style": {
            "font": "DejaVu Sans",
            "size": 11,
            "color": "&H00FFFFFF",
            "outline_color": "&H00000000",
            "position": "bottom",
        },
    },
}


def get_template(name):
    return TEMPLATES.get(name, TEMPLATES["energetic"])
