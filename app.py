"""
app.py - The web API, now fully asynchronous.

Flow:
  POST /api/edit          -> queues the job, returns job_id immediately
  GET  /api/status/<id>   -> progress + state (browser polls this)
  GET  /api/download/<id> -> the finished file

Nothing waits on a held-open HTTP connection, which is what makes
unattended/automated processing possible. The old version processed inside
the request and would time out on any render over ~30s.
"""
import os
import subprocess
import uuid

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import jobs
from pipeline import render
from templates import TEMPLATES

try:
    import gpu_enhance
    GPU_MODULE = True
except ImportError:
    GPU_MODULE = False

UPLOAD_DIR = "/tmp/autoeditor_uploads"
OUTPUT_DIR = "/tmp/autoeditor_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

MAX_UPLOAD_MB = 500
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

jobs.start_cleanup_loop()


def _validate_output(path):
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return False, "Output file missing or too small"
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False, f"ffprobe could not read output: {result.stderr[-200:]}"
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return False, "ffprobe returned unreadable duration"
    if duration <= 0:
        return False, "Output has zero duration"
    return True, duration


def _process(input_path, output_path, template_name, options, progress_callback=None):
    if progress_callback:
        progress_callback(5, "analysing footage")

    result = render(
        input_path, output_path=output_path, template_name=template_name,
        vertical=options.get("vertical", True),
        audio_profile=options.get("audio_profile", "standard"),
        resolution=options.get("resolution", "fullhd"),
        multi_angle=options.get("multi_angle", True),
    )

    if progress_callback:
        progress_callback(90, "verifying output")

    if os.path.exists(input_path):
        try:
            os.remove(input_path)
        except OSError:
            pass

    if not result["success"]:
        raise RuntimeError(f"Render failed: {str(result.get('stderr_tail'))[-400:]}")

    is_valid, info = _validate_output(output_path)
    if not is_valid:
        raise RuntimeError(f"Render produced an invalid file: {info}")

    return {
        "output_path": output_path,
        "duration_seconds": round(info, 2),
        "size_bytes": os.path.getsize(output_path),
        "effects_applied": result["effects_applied"],
        "reframe": result.get("reframe", {}),
        "audio_cleaned": result.get("audio_cleaned", False),
        "shot_variation": result.get("shot_variation"),
        "output_resolution": result.get("output_resolution"),
        "grading_decisions": result.get("grading_decisions", {}),
    }


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "gpu_enhancement_available": GPU_MODULE and gpu_enhance.is_available(),
        "templates": list(TEMPLATES.keys()),
    })


@app.route("/api/templates", methods=["GET"])
def list_templates():
    return jsonify([
        {"id": key, "label": t["label"], "description": t["description"]}
        for key, t in TEMPLATES.items()
    ])


@app.route("/api/edit", methods=["POST"])
def edit_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files["video"]
    template_name = request.form.get("template", "energetic")
    if template_name not in TEMPLATES:
        return jsonify({"error": f"Unknown template: {template_name}"}), 400

    options = {
        "vertical": request.form.get("vertical", "true").lower() != "false",
        "audio_profile": request.form.get("audio_profile", "standard"),
        "resolution": request.form.get("resolution", "fullhd"),
        "multi_angle": request.form.get("multi_angle", "true").lower() != "false",
    }

    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_raw.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_edited.mp4")
    file.save(input_path)

    job_id = jobs.create_job({"template": template_name, "options": options})
    jobs.run_in_background(job_id, _process, input_path, output_path,
                           template_name, options)

    return jsonify({
        "job_id": job_id,
        "status_url": f"/api/status/{job_id}",
        "state": "queued",
    }), 202


@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = jobs.get_job(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404

    response = {
        "job_id": job_id,
        "state": job["state"],
        "progress": job["progress"],
        "stage": job["stage"],
    }

    if job["state"] == jobs.DONE and job["result"]:
        r = job["result"]
        response.update({
            "download_url": f"/api/download/{job_id}",
            "duration_seconds": r.get("duration_seconds"),
            "size_bytes": r.get("size_bytes"),
            "effects_applied": r.get("effects_applied"),
            "reframe": r.get("reframe"),
            "audio_cleaned": r.get("audio_cleaned"),
            "shot_variation": r.get("shot_variation"),
            "output_resolution": r.get("output_resolution"),
        })
    elif job["state"] == jobs.FAILED:
        response["error"] = job.get("error")

    return jsonify(response)


@app.route("/api/download/<job_id>", methods=["GET"])
def download(job_id):
    job = jobs.get_job(job_id)
    if not job or job["state"] != jobs.DONE or not job.get("result"):
        return jsonify({"error": "Not ready or not found"}), 404

    path = job["result"].get("output_path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "File no longer available"}), 404

    return send_file(path, mimetype="application/octet-stream",
                     as_attachment=True, download_name="edited_video.mp4")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
