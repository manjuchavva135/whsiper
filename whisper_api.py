"""
YSRCP Talks — GPU Whisper Server (RunPod)
==========================================
Simple raw transcription — NO chunking here.
Chunking is done on GCP server side (30s chunks).
This server receives small 30s chunks and transcribes each one fast.

Result: Each request completes in ~5 seconds (GPU, 30s audio).
No timeouts. No 502 errors.
"""

import logging
import os
import tempfile
import time
import unicodedata
import re

from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("whisper_api")

WHISPER_API_KEY    = os.getenv("WHISPER_API_KEY",     "ysrcp_gpu_2026")
# RTX 2000 Ada Generation = 16GB VRAM
# large-v3 = ~10GB → fits with 6GB headroom for audio processing
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL",       "large-v3")
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE",      "cuda")
WHISPER_COMPUTE    = os.getenv("WHISPER_COMPUTE_TYPE","float16")

import queue as _queue
import threading as _threading

app = Flask(__name__)

# ── Thread-safe model pool ─────────────────────────────────────────────────────
# faster-whisper is NOT thread-safe — one instance per concurrent request.
# POOL_SIZE=2 on RTX 3090 (large-v3 ~10GB each, 24GB total fits 2).
# Requests that exceed the pool size wait (up to 120s) for a free slot.
POOL_SIZE   = int(os.getenv("WHISPER_POOL_SIZE", "1"))   # 16GB VRAM — 1x large-v3 fits (10GB used, 6GB free)
_model_pool = _queue.Queue()
_pool_lock  = _threading.Lock()
_pool_ready = False


def _build_pool():
    global _pool_ready
    with _pool_lock:
        if _pool_ready:
            return
        from faster_whisper import WhisperModel
        log.info("Building model pool: %d x %s on %s (%s)...",
                 POOL_SIZE, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE)
        for i in range(POOL_SIZE):
            m = WhisperModel(
                WHISPER_MODEL_SIZE,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE,
            )
            _model_pool.put(m)
            log.info("  Model instance %d/%d ready", i + 1, POOL_SIZE)
        _pool_ready = True
        log.info("Pool ready — %d parallel slots available", POOL_SIZE)


def get_model():
    if not _pool_ready:
        _build_pool()
    return _model_pool.get(block=True, timeout=120)


def return_model(m):
    _model_pool.put(m)


def authorized(req) -> bool:
    # Accept both X-API-Key and Authorization: Bearer (YSRCP backend sends Bearer)
    key = (req.headers.get("X-API-Key") or
           req.headers.get("Authorization", "").replace("Bearer ", "") or
           req.args.get("api_key", ""))
    return key == WHISPER_API_KEY


def normalize_telugu(text: str) -> str:
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200d", "").replace("\u200c", "")
    text = re.sub("\u0c4d{2,}", "\u0c4d", text)
    text = re.sub(r"\u0c4d(\s)", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "model":   WHISPER_MODEL_SIZE,
        "device":  WHISPER_DEVICE,
        "compute": WHISPER_COMPUTE,
        "mode":    "raw_chunk",
        "font":    "Gowthami",
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not authorized(request):
        return jsonify({"error": "Unauthorized"}), 401
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    f          = request.files["audio"]
    language   = (request.form.get("language") or "te").split("-")[0].lower()
    # VAD filter disabled by default — it aggressively strips Telugu speech
    # which it mistakes for silence (VAD model is trained primarily on English)
    vad_filter = (request.form.get("vad_filter") or "false").lower() == "true"
    beam_size  = max(1, min(10, int(request.form.get("beam_size") or 5)))
    lang_map   = {"te-in":"te","hi-in":"hi","en-in":"en","kn-in":"kn","ta-in":"ta"}
    language   = lang_map.get(language, language)

    suffix = os.path.splitext(f.filename or "audio.mp3")[1] or ".mp3"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        f.save(tmp)
        size_kb = os.path.getsize(tmp) / 1024
        t0 = time.time()
        log.info("Received: %s %.1fKB lang=%s", f.filename, size_kb, language)

        model = get_model()
        try:
            segments, info = model.transcribe(
                tmp,
                language   = language,
                task       = "transcribe",
                vad_filter = vad_filter,
                beam_size  = beam_size,
                condition_on_previous_text = False,
            )
            texts = [(seg.text or "").strip() for seg in segments]
        finally:
            return_model(model)   # always return model to pool even on error

        text  = normalize_telugu(" ".join(t for t in texts if t))
        elapsed = round(time.time() - t0, 2)
        words   = len(text.split())
        log.info("Done: %d words | %.2fs", words, elapsed)

        return jsonify({
            "transcript":         text,
            "transcript_raw":     text,
            "word_count":         words,
            "processing_seconds": elapsed,
            "language_detected":  getattr(info, "language", language),
            "model":              WHISPER_MODEL_SIZE,
            "font":               "Gowthami",
            "encoding":           "UTF-8 NFC",
        })

    except Exception as exc:
        log.error("Error: %r", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        try: os.unlink(tmp)
        except: pass


if __name__ == "__main__":
    log.info("YSRCP Whisper | Model:%s | Device:%s | Mode:raw_chunk | Font:Gowthami",
             WHISPER_MODEL_SIZE, WHISPER_DEVICE)
    get_model()
    _build_pool()   # pre-load all model instances before first request
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
