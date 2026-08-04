"""
jobs.py - Background job queue.

WHY THIS EXISTS (it's not optional for automation):
The old flow processed the video inside the HTTP request, so the browser sat
waiting. Renders now take 1-3+ minutes, and browsers, proxies and Railway
itself all cut connections long before that. The upload would simply fail.

New flow:
  POST /api/edit   -> queues the work, returns a job_id IMMEDIATELY
  GET  /api/status -> browser polls this to watch progress
  GET  /api/download -> fetches the file once the job reports done

This is what makes unattended automation possible: nothing depends on a
browser staying connected.
"""
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict

# Job states
QUEUED = "queued"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"

# Bounded store so a long-running server can't grow memory forever.
MAX_JOBS_KEPT = 200
# How long a finished job's file stays on disk before cleanup (seconds)
JOB_TTL = 3600

_jobs = OrderedDict()
_lock = threading.Lock()


def create_job(metadata=None):
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "state": QUEUED,
            "progress": 0,
            "stage": "queued",
            "created_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
            "metadata": metadata or {},
        }
        # Evict the oldest entries past the cap
        while len(_jobs) > MAX_JOBS_KEPT:
            old_id, old = _jobs.popitem(last=False)
            _cleanup_files(old)
    return job_id


def _cleanup_files(job):
    """Remove a job's output file from disk."""
    result = job.get("result") or {}
    path = result.get("output_path")
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def update_job(job_id, **fields):
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(fields)


def set_progress(job_id, progress, stage=None):
    """Called from inside the worker so the browser can show real progress
    instead of an indeterminate spinner."""
    fields = {"progress": max(0, min(100, int(progress)))}
    if stage:
        fields["stage"] = stage
    update_job(job_id, **fields)


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def run_in_background(job_id, fn, *args, **kwargs):
    """
    Runs fn in a worker thread. fn receives a `progress_callback` kwarg it can
    call with (percent, stage_name).

    A thread (rather than a separate process/queue service) is deliberate:
    it keeps deployment to a single Railway service with no extra
    infrastructure. The tradeoff is that concurrent heavy renders compete for
    the same CPU - fine at low volume, and the point to move to a real
    worker queue is when concurrent orders become normal.
    """
    def _worker():
        update_job(job_id, state=PROCESSING, stage="starting", progress=1)
        try:
            def progress_callback(pct, stage=None):
                set_progress(job_id, pct, stage)

            result = fn(*args, progress_callback=progress_callback, **kwargs)
            update_job(job_id, state=DONE, progress=100, stage="done",
                       result=result, finished_at=time.time())
        except Exception as e:
            update_job(job_id, state=FAILED, error=str(e),
                       stage="failed", finished_at=time.time(),
                       result={"traceback": traceback.format_exc()[-1500:]})

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def cleanup_expired():
    """Deletes output files for jobs older than JOB_TTL. Called periodically
    so finished renders don't fill the disk."""
    now = time.time()
    with _lock:
        for job in list(_jobs.values()):
            finished = job.get("finished_at")
            if finished and (now - finished) > JOB_TTL:
                _cleanup_files(job)
                job["result"] = None
                job["state"] = "expired"


def start_cleanup_loop(interval=600):
    def _loop():
        while True:
            time.sleep(interval)
            try:
                cleanup_expired()
            except Exception:
                pass
    threading.Thread(target=_loop, daemon=True).start()
