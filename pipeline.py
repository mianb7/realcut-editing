"""
pipeline.py - The real engine. Takes raw footage + a template name, and
produces a finished, edited video:
  1. Detects dead air and removes it (cuts.py)
  2. Analyzes motion to decide slow-mo / fast-mo segments (motion.py)
  3. Analyzes color and applies template grade (color.py)
  4. Builds one ffmpeg filter_complex that trims, speed-ramps, concatenates,
     color-grades, and (optionally) burns captions - in a single render pass.
"""
import os
import subprocess
import tempfile

from cuts import build_keep_segments
from motion import analyze_motion, classify_windows, merge_adjacent
from color import auto_levels, build_eq_filter
from templates import get_template
from reframe import build_crop_filter
from audio_clean import build_audio_filter
from cinematic import build_cinematic_chain
from angles import _affordable_shots, assign_shot_sizes, build_shot_crop
from quality import get_output_dimensions, build_scale_filter, build_encode_args


def _get_orientation_filter(video_path):
    """
    ffmpeg's automatic camera-orientation correction (from the phone's
    rotation/mirror metadata) does NOT reliably apply inside a custom
    filter_complex graph the way it does with simple -vf usage. Without
    this, footage can come out inconsistently flipped/rotated. We read the
    real orientation from the file's side data ourselves and apply the
    correct correction explicitly to every segment, so it's uniform
    throughout the whole output.
    Returns an ffmpeg filter string (e.g. "transpose=1") or None.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream_side_data=rotation",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    rotation_str = result.stdout.strip()
    try:
        rotation = int(float(rotation_str)) if rotation_str else 0
    except ValueError:
        rotation = 0

    # Normalize to 0/90/180/270
    rotation = rotation % 360
    if rotation == 90 or rotation == -270:
        return "transpose=2"  # rotate 90 counter-clockwise
    elif rotation == -90 or rotation == 270:
        return "transpose=1"  # rotate 90 clockwise
    elif rotation == 180 or rotation == -180:
        return "hflip,vflip"
    return None


def _intersect(a_start, a_end, b_start, b_end):
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return (s, e) if e > s else None


def _split_segment_by_motion(seg_start, seg_end, motion_segments):
    """Splits one 'keep' segment into sub-segments tagged with an effect,
    based on overlap with the classified motion windows."""
    pieces = []
    cursor = seg_start
    for m in sorted(motion_segments, key=lambda x: x["start"]):
        overlap = _intersect(seg_start, seg_end, m["start"], m["end"])
        if not overlap:
            continue
        o_start, o_end = overlap
        if o_start > cursor:
            pieces.append((cursor, o_start, "normal"))
        pieces.append((o_start, o_end, m["effect"]))
        cursor = o_end
    if cursor < seg_end:
        pieces.append((cursor, seg_end, "normal"))
    # merge tiny slivers into neighbors (avoid ffmpeg choking on <0.05s clips)
    cleaned = [p for p in pieces if p[1] - p[0] > 0.08]
    return cleaned if cleaned else [(seg_start, seg_end, "normal")]


def _atempo_chain(factor):
    """atempo filter only supports 0.5-2.0 per instance; chain multiple
    instances to reach factors outside that range."""
    filters = []
    f = factor
    if f < 0.5:
        while f < 0.5:
            filters.append("atempo=0.5")
            f /= 0.5
        filters.append(f"atempo={round(f, 3)}")
    elif f > 2.0:
        while f > 2.0:
            filters.append("atempo=2.0")
            f /= 2.0
        filters.append(f"atempo={round(f, 3)}")
    else:
        filters.append(f"atempo={round(f, 3)}")
    return ",".join(filters)


def build_final_segments(video_path, template):
    keep = build_keep_segments(
        video_path, noise_db=template["silence_sensitivity"]
    )
    windows, _duration = analyze_motion(video_path)
    classified = classify_windows(
        windows,
        slomo_percentile=template["slomo_percentile"],
        fastmo_percentile=template["fastmo_percentile"],
    )
    merged_motion = merge_adjacent(classified)

    final_segments = []
    for seg_start, seg_end in keep:
        pieces = _split_segment_by_motion(seg_start, seg_end, merged_motion)
        for p_start, p_end, effect in pieces:
            if effect == "slomo":
                speed = template["slomo_speed"]
            elif effect == "fastmo":
                speed = template["fastmo_speed"]
            else:
                speed = 1.0
            final_segments.append((p_start, p_end, speed, effect))
    return final_segments


def _get_video_specs(video_path):
    """Source width/height/fps, needed to build the Ken Burns zoom filter
    at matching dimensions."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    parts = result.stdout.strip().split(",")
    width, height, fps_frac = int(parts[0]), int(parts[1]), parts[2]
    num, den = fps_frac.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    return width, height, round(fps, 2)


def render(video_path, template_name, output_path, srt_path=None, max_segments=60,
           vertical=True, audio_profile="standard", resolution="fullhd",
           multi_angle=True):
    template = get_template(template_name)
    segments = build_final_segments(video_path, template)

    # Safety cap: extremely choppy footage could produce hundreds of tiny
    # segments, which would make the ffmpeg filter graph unmanageable.
    if len(segments) > max_segments:
        segments = segments[:max_segments]

    # Full cinematic grade (analysis-driven) replaces the basic eq grade
    cine_chain, cine_stats, cine_decisions = build_cinematic_chain(
        video_path, look=template.get("film_look", "cinematic")
    )
    if cine_chain:
        eq_filter = cine_chain
    else:
        auto = auto_levels(video_path)
        eq_filter = build_eq_filter(auto, template["color_grade"])
        cine_stats, cine_decisions = None, {}
    orientation_filter = _get_orientation_filter(video_path)
    src_width, src_height, src_fps = _get_video_specs(video_path)

    use_transitions = template.get("transition_duration", 0) > 0
    use_ken_burns = template.get("ken_burns", False)
    zoom_max = template.get("ken_burns_intensity", 1.08)

    # Multi-angle: assign wide/medium/close shot sizes across the segments
    shot_assignments = None
    subject_x = 0.5
    if multi_angle:
        out_w_probe, out_h_probe = get_output_dimensions(
            src_width, src_height, resolution, vertical)
        allowed = _affordable_shots(src_width, src_height, out_w_probe, out_h_probe)
        if len(allowed) > 1:
            shot_assignments = assign_shot_sizes(segments, allowed)
            try:
                from reframe import _detect_face_track
                track, _ = _detect_face_track(video_path)
                if track:
                    subject_x = sum(c for _, c in track) / len(track)
            except Exception:
                subject_x = 0.5

    filter_parts = []
    seg_v_labels = []
    seg_a_labels = []
    seg_durations = []  # output duration of each segment, post speed-change

    for i, (start, end, speed, effect) in enumerate(segments):
        v_label = f"v{i}"
        a_label = f"a{i}"
        orient_stage = f",{orientation_filter}" if orientation_filter else ""
        out_duration = (end - start) / speed

        # Ken Burns: only on normal-speed segments (slomo/fastmo already
        # have visual movement from the speed change itself) and only
        # when the segment is long enough for a zoom to read as intentional
        # rather than a jitter.
        zoom_stage = ""
        if use_ken_burns and effect == "normal" and out_duration >= 0.6:
            total_frames = max(1, int(out_duration * src_fps))
            zoom_increment = (zoom_max - 1.0) / total_frames
            zoom_stage = (
                f",zoompan=z='min(zoom+{zoom_increment:.6f},{zoom_max})':"
                f"d=1:s={src_width}x{src_height}:fps={src_fps}"
            )

        shot_stage = ""
        if shot_assignments:
            shot_crop = build_shot_crop(shot_assignments[i], src_width, src_height,
                                        subject_x, 0.42)
            if shot_crop:
                # Crop to the shot size, then scale back to a common size so
                # every segment matches before they're joined together.
                shot_stage = f",{shot_crop},scale={src_width}:{src_height}:flags=lanczos"

        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS{orient_stage}{shot_stage},"
            f"setpts={1/speed:.4f}*PTS{zoom_stage},fps={src_fps}[{v_label}]"
        )
        atempo = _atempo_chain(speed)
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
            f"{atempo}[{a_label}]"
        )
        seg_v_labels.append(v_label)
        seg_a_labels.append(a_label)
        seg_durations.append(out_duration)

    n = len(segments)

    if use_transitions and n > 1:
        configured_trans = template["transition_duration"]
        cur_v = seg_v_labels[0]
        cur_a = seg_a_labels[0]
        cur_duration = seg_durations[0]

        for i in range(1, n):
            next_v = seg_v_labels[i]
            next_a = seg_a_labels[i]
            next_duration = seg_durations[i]
            # Clamp so the transition never eats more than 40% of either
            # neighboring clip - keeps short segments from breaking xfade.
            trans_dur = min(configured_trans, cur_duration * 0.4, next_duration * 0.4)
            trans_dur = max(trans_dur, 0.05)

            offset = max(0.0, cur_duration - trans_dur)
            out_v = f"vx{i}"
            out_a = f"ax{i}"
            filter_parts.append(
                f"[{cur_v}][{next_v}]xfade=transition={template['transition_style']}:"
                f"duration={trans_dur:.3f}:offset={offset:.3f}[{out_v}]"
            )
            filter_parts.append(
                f"[{cur_a}][{next_a}]acrossfade=d={trans_dur:.3f}[{out_a}]"
            )
            cur_v, cur_a = out_v, out_a
            cur_duration = cur_duration + next_duration - trans_dur

        filter_parts.append(f"[{cur_v}]{eq_filter}[vgraded]")
        final_audio_label = cur_a
    else:
        concat_str = "".join(f"[{seg_v_labels[i]}][{seg_a_labels[i]}]" for i in range(n))
        filter_parts.append(f"{concat_str}concat=n={n}:v=1:a=1[vconcat][aout]")
        filter_parts.append(f"[vconcat]{eq_filter}[vgraded]")
        final_audio_label = "aout"

    final_video_label = "vgraded"

    # Vertical reframe: crop to 9:16 following the subject. Applied after
    # grading so the crop math uses the source dimensions consistently.
    reframe_meta = {"reframed": False}
    if vertical:
        crop_filter, reframe_meta = build_crop_filter(video_path)
        if crop_filter:
            filter_parts.append(f"[{final_video_label}]{crop_filter}[vcrop]")
            final_video_label = "vcrop"

    if srt_path and os.path.exists(srt_path):
        style = template["caption_style"]
        force_style = (
            f"FontName={style['font']},FontSize={style['size']},"
            f"PrimaryColour={style['color']},OutlineColour={style['outline_color']}"
        )
        srt_escaped = srt_path.replace(":", "\\:")
        filter_parts.append(
            f"[{final_video_label}]subtitles='{srt_escaped}':force_style='{force_style}'[vcap]"
        )
        final_video_label = "vcap"

    # Audio cleanup: denoise + normalize to social loudness standard
    audio_filter = build_audio_filter(audio_profile)
    if audio_filter:
        filter_parts.append(f"[{final_audio_label}]{audio_filter}[aclean]")
        final_audio_label = "aclean"

    # Final resolution + detail enhancement stage
    if reframe_meta.get("reframed") and reframe_meta.get("output_size"):
        cw, ch = reframe_meta["output_size"].split("x")
        cur_w, cur_h = int(cw), int(ch)
    else:
        cur_w, cur_h = src_width, src_height

    out_w, out_h = get_output_dimensions(cur_w, cur_h, resolution, vertical)
    scale_filter = build_scale_filter(cur_w, cur_h, out_w, out_h)
    if scale_filter:
        filter_parts.append(f"[{final_video_label}]{scale_filter}[vfinal]")
        final_video_label = "vfinal"

    filter_complex = ";".join(filter_parts)

    encode_args = build_encode_args(resolution, high_quality=True)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", f"[{final_video_label}]", "-map", f"[{final_audio_label}]",
        *encode_args,
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "success": result.returncode == 0,
        "segments_used": len(segments),
        "transitions_used": use_transitions and n > 1,
        "ken_burns_used": use_ken_burns,
        "reframe": reframe_meta,
        "shot_variation": shot_assignments,
        "output_resolution": f"{out_w}x{out_h}",
        "footage_analysis": cine_stats,
        "grading_decisions": cine_decisions,
        "audio_cleaned": bool(audio_filter),
        "effects_applied": {
            "slomo": sum(1 for s in segments if s[3] == "slomo"),
            "fastmo": sum(1 for s in segments if s[3] == "fastmo"),
            "normal": sum(1 for s in segments if s[3] == "normal"),
        },
        "stderr_tail": result.stderr[-2000:] if result.returncode != 0 else None,
        "output_path": output_path if result.returncode == 0 else None,
    }
