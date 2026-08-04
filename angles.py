"""
angles.py - Simulated multi-camera editing from single-camera footage.

IMPORTANT - what this is and isn't:
It does NOT invent camera angles that were never filmed. That's impossible;
the pixels don't exist. What it DOES is what a professional editor does with
a single locked-off camera: cut between a WIDE, a MEDIUM push-in and a
CLOSE-UP, all pulled from different crop regions of the same frame, tracked
to the subject. To a viewer this reads as a multi-camera edit, because
varying shot size is the actual visual language of multi-cam - not the
literal number of cameras.

For this to look good rather than soft, the source needs headroom: a 1080p
source cropped to 55% and scaled back to 1080p loses real detail. So
close-ups are only used when the source resolution can afford them, and the
crop factors are chosen per source resolution.
"""

SHOT_SIZES = {
    "wide": 1.00,     # full frame, no crop
    "medium": 0.78,   # modest push-in
    "close": 0.60,    # close-up on the subject
}


def _affordable_shots(src_width, src_height, out_width, out_height):
    """
    Only allow a shot size if cropping to it still leaves at least the
    output resolution - otherwise we'd be upscaling a crop and it looks soft.
    """
    allowed = ["wide"]
    for name, factor in SHOT_SIZES.items():
        if name == "wide":
            continue
        cropped_w = src_width * factor
        cropped_h = src_height * factor
        # Allow a little upscale headroom (10%) before rejecting the shot
        if cropped_w >= out_width * 0.9 and cropped_h >= out_height * 0.9:
            allowed.append(name)
    return allowed


def assign_shot_sizes(segments, allowed_shots, face_track=None):
    """
    Assigns a shot size to each segment using editing logic, not randomness:
      - never repeat the same shot size back-to-back (that's the whole point
        of cutting - if the framing doesn't change, the cut looks like a
        glitch rather than an edit)
      - slow-mo / emphasis moments get the closest available shot
      - fast-mo / filler moments get the widest, so the eye can keep up
      - long normal segments alternate to keep the edit from going static
    Returns a list of shot-size names, one per segment.
    """
    assigned = []
    previous = None

    for i, seg in enumerate(segments):
        effect = seg[3] if len(seg) > 3 else "normal"
        duration = (seg[1] - seg[0]) / seg[2] if len(seg) > 2 else 1.0

        if effect == "slomo" and "close" in allowed_shots:
            choice = "close"
        elif effect == "fastmo":
            choice = "wide"
        elif duration > 2.5 and "medium" in allowed_shots:
            # Long shots get a push-in so they don't feel static
            choice = "medium"
        else:
            # Alternate through what's available for visual variety
            options = [s for s in allowed_shots if s != previous]
            if not options:
                options = allowed_shots
            choice = options[i % len(options)]

        # Enforce the no-repeat rule
        if choice == previous:
            alternatives = [s for s in allowed_shots if s != previous]
            if alternatives:
                choice = alternatives[0]

        assigned.append(choice)
        previous = choice

    return assigned


def build_shot_crop(shot_size, src_width, src_height, face_center_x=0.5,
                    face_center_y=0.42):
    """
    Returns an ffmpeg crop filter for one shot size, framed on the subject.

    face_center_y defaults to 0.42 rather than 0.5 because faces sit above
    centre in a well-composed frame (headroom rule) - centring on 0.5 puts
    the face too low and crops the top of the head on close-ups.
    """
    factor = SHOT_SIZES.get(shot_size, 1.0)
    if factor >= 1.0:
        return None  # wide = no crop needed

    crop_w = int(src_width * factor)
    crop_h = int(src_height * factor)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    # Position the crop on the subject, clamped inside the frame
    x = int(face_center_x * src_width - crop_w / 2)
    y = int(face_center_y * src_height - crop_h / 2)
    x = max(0, min(src_width - crop_w, x))
    y = max(0, min(src_height - crop_h, y))

    return f"crop={crop_w}:{crop_h}:{x}:{y}"
