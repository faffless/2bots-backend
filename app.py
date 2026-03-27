"""
App 2.0 — Text + TTS via SSE. No locks, no flavour system.
GPT text → GPT audio → Claude text → Claude audio → done.
TTS generation is parallelised with Claude API call for speed.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import tuning
from engine import (
    TwoBotsEngine,
    AVAILABLE_VOICES,
    MODES,
    INTERACTION_STYLES,
    PERSONALITIES,
    CHARACTER_QUIRKS,
    RESPONSE_LENGTHS,
)


# ---- Structured log buffer ----
_LOG_BUFFER: deque = deque(maxlen=200)
_APP_START = time.time()


def log(category: str, msg: str, **extra):
    """Append a structured log entry and print to console."""
    entry = {
        "t": round(time.time() - _APP_START, 2),
        "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "cat": category,
        "msg": msg,
        **extra,
    }
    _LOG_BUFFER.append(entry)
    # Also print to console for terminal visibility
    extra_str = "".join(f" {k}={v}" for k, v in extra.items()) if extra else ""
    print(f"[{entry['ts']}] [{category}]{extra_str} {msg}")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="2BOTS Backend v2")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": f"Rate limit exceeded: {exc.detail}"},
    )

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {"status": "2BOTS API v2 running"}

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001,https://2bots.io,https://www.2bots.io,https://2bots-web.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory sessions ----
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ---- Prefetch cache for autopilot batches ----
# Key: session_id, Value: {"batch": [...], "who_generated": str, "task": asyncio.Task}
PREFETCH_CACHE: Dict[str, Dict[str, Any]] = {}


def get_engine(sid: str) -> TwoBotsEngine:
    data = SESSIONS.get(sid)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return TwoBotsEngine.from_state(data)


def save(sid: str, engine: TwoBotsEngine) -> None:
    SESSIONS[sid] = engine.export_state()


def save_messages_only(sid: str, engine: TwoBotsEngine) -> None:
    """Save only conversation messages — preserves hot-swapped settings.
    This prevents auto/stream from overwriting settings that were
    changed mid-conversation via /settings/update."""
    if sid in SESSIONS:
        SESSIONS[sid]["claude_msgs"] = list(engine.state.claude_msgs)
        SESSIONS[sid]["gpt_msgs"] = list(engine.state.gpt_msgs)
        SESSIONS[sid]["rounds_since_filler"] = engine.state.rounds_since_filler
        SESSIONS[sid]["next_filler_at"] = engine.state.next_filler_at
        SESSIONS[sid]["autopilot_batch_count"] = engine.state.autopilot_batch_count
    else:
        SESSIONS[sid] = engine.export_state()


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ---- Request models ----
class StartRequest(BaseModel):
    personality: float = 0.5
    settings: Optional[Dict[str, Any]] = None

class TurnRequest(BaseModel):
    session_id: str
    text: str

class AutopilotRequest(BaseModel):
    session_id: str
    who_generates: str = "claude"  # which AI writes the batch

class FillerRequest(BaseModel):
    session_id: str
    user_text: str

class SettingsUpdate(BaseModel):
    session_id: str
    settings: Dict[str, Any]


# ---- Core SSE generator ----

async def _generate_round(engine: TwoBotsEngine, auto: bool, opener: bool = False):
    """Yield SSE events for one round: GPT text+audio, Claude text+audio."""
    import random as _rand
    t0 = time.time()
    mode = "opener" if opener else ("auto" if auto else "turn")
    gpt_voice = engine.get_gpt_voice()
    claude_voice = engine.get_claude_voice()
    gpt_len = engine._s("gpt_response_length") or "concise"
    claude_len = engine._s("claude_response_length") or "concise"
    log("round", f"START ({mode})", gpt_voice=gpt_voice, claude_voice=claude_voice,
        gpt_length=gpt_len, claude_length=claude_len)

    # Check if this round should include a filler (never on opener or user turn)
    use_filler = auto and not opener and engine.should_filler()
    filler_bot = _rand.choice(["gpt", "claude"]) if use_filler else None

    if use_filler:
        log("filler", f"Injecting filler for {filler_bot}")

    # 1. GPT generates text (or filler)
    if filler_bot == "gpt":
        gpt_text = engine.get_filler("gpt")
        log("gpt", f"FILLER: '{gpt_text}'")
    else:
        gpt_text = await asyncio.to_thread(engine.ask_gpt, auto, opener)
    engine.add_message("gpt", gpt_text)
    t1 = time.time()
    gpt_words = len(gpt_text.split())
    if filler_bot != "gpt":
        log("gpt", f"Text done: {gpt_words}w in {t1-t0:.1f}s", words=gpt_words, secs=round(t1-t0, 1))
    yield sse({"type": "text", "speaker": "gpt", "text": gpt_text})

    # 2. Parallel: GPT TTS + Claude thinks (or filler)
    gpt_tts_task = asyncio.create_task(engine.generate_tts_bytes(gpt_text, gpt_voice, "gpt"))

    if filler_bot == "claude":
        claude_text = engine.get_filler("claude")
        log("claude", f"FILLER: '{claude_text}'")
        claude_text_task = None
    else:
        claude_text_task = asyncio.create_task(asyncio.to_thread(
            engine.ask_claude, auto, opener
        ))

    gpt_audio = await gpt_tts_task
    t2 = time.time()
    log("gpt", f"TTS done in {t2-t1:.1f}s ({gpt_words}w)", secs=round(t2-t1, 1))
    yield sse({
        "type": "audio", "speaker": "gpt",
        "audio_base64": base64.b64encode(gpt_audio).decode(),
        "mime_type": "audio/mpeg",
    })

    if claude_text_task:
        claude_text = await claude_text_task
    engine.add_message("claude", claude_text)
    t3 = time.time()
    claude_words = len(claude_text.split())
    if filler_bot != "claude":
        log("claude", f"Text done: {claude_words}w in {t3-t1:.1f}s (parallel)", words=claude_words, secs=round(t3-t1, 1))
    yield sse({"type": "text", "speaker": "claude", "text": claude_text})

    claude_audio = await engine.generate_tts_bytes(claude_text, claude_voice, "claude")
    t4 = time.time()
    log("claude", f"TTS done in {t4-t3:.1f}s ({claude_words}w)", secs=round(t4-t3, 1))

    total = round(t4-t0, 1)
    log("round", f"DONE in {total}s (GPT {gpt_words}w + Claude {claude_words}w)", total_secs=total)
    yield sse({
        "type": "audio", "speaker": "claude",
        "audio_base64": base64.b64encode(claude_audio).decode(),
        "mime_type": "audio/mpeg",
    })

    # Tick counters (only for non-filler rounds)
    if not use_filler:
        engine.tick_filler()

    yield sse({"type": "done"})


# ---- Autopilot batch streaming ----

async def _stream_batch(engine: TwoBotsEngine, batch: list, who_generated: str, sid: str):
    """Stream a pre-generated autopilot batch: text+audio for each message sequentially.
    Each message gets TTS'd with the correct voice and streamed as SSE events.
    Kicks off prefetch of the next batch partway through playback."""
    t0 = time.time()
    gpt_voice = engine.get_gpt_voice()
    claude_voice = engine.get_claude_voice()
    next_generator = "claude" if who_generated == "gpt" else "gpt"
    prefetch_started = False

    log("autopilot", f"Streaming batch of {len(batch)} messages (generated by {who_generated})")

    # Add ALL batch messages to history UPFRONT and save immediately.
    # This ensures the prefetch for the next batch sees the complete conversation.
    for msg in batch:
        engine.add_message(msg["speaker"], msg["text"])
    save_messages_only(sid, engine)

    for i, msg in enumerate(batch):
        speaker = msg["speaker"]
        text = msg["text"]
        voice = gpt_voice if speaker == "gpt" else claude_voice

        # Send text event
        yield sse({"type": "text", "speaker": speaker, "text": text})

        # Generate and send TTS
        t1 = time.time()
        audio = await engine.generate_tts_bytes(text, voice, speaker)
        t2 = time.time()
        log(speaker, f"TTS [{i+1}/{len(batch)}] {len(text.split())}w in {t2-t1:.1f}s")

        yield sse({
            "type": "audio", "speaker": speaker,
            "audio_base64": base64.b64encode(audio).decode(),
            "mime_type": "audio/mpeg",
        })

        # Prefetch next batch around 60% through this one
        if not prefetch_started and i >= len(batch) * 0.6:
            prefetch_started = True
            # Start prefetch in background
            async def _do_prefetch():
                try:
                    pf_engine = get_engine(sid)
                    pf_batch = await asyncio.to_thread(
                        pf_engine.generate_autopilot_batch, next_generator
                    )
                    PREFETCH_CACHE[sid] = {
                        "batch": pf_batch,
                        "who_generated": next_generator,
                        "engine": pf_engine,
                    }
                    log("prefetch", f"Prefetch ready for {sid[:8]}... ({len(pf_batch)} msgs via {next_generator})")
                except Exception as e:
                    log("prefetch", f"Prefetch failed: {e}")
            asyncio.create_task(_do_prefetch())

    total = round(time.time() - t0, 1)
    log("autopilot", f"Batch DONE in {total}s ({len(batch)} msgs)", total_secs=total)

    # Tell frontend which AI should generate the NEXT batch (leapfrog)
    yield sse({"type": "done", "next_generator": next_generator, "batch_size": len(batch)})


# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def get_voices():
    return {
        "voices": {k: v for k, v in AVAILABLE_VOICES.items()},
        "modes": {k: v["label"] for k, v in MODES.items()},
        "personalities": list(PERSONALITIES.keys()),
        "quirks": list(CHARACTER_QUIRKS.keys()),
        "response_lengths": {k: v["label"] for k, v in RESPONSE_LENGTHS.items()},
    }


@app.post("/start/stream")
@limiter.limit("5/minute")
async def start_stream(request: Request, req: StartRequest):
    """First round — AIs introduce themselves naturally."""
    import random as _rand
    sid = str(uuid.uuid4())
    log("session", f"NEW session {sid[:8]}...")
    engine = TwoBotsEngine(personality=req.personality, settings=req.settings or {})

    # 15% chance each bot gets a random personality instead of default
    non_default = [k for k in PERSONALITIES if k != "default"]
    if _rand.random() < tuning.RANDOM_PERSONALITY_CHANCE and non_default:
        rand_p = _rand.choice(non_default)
        engine.state.settings["gpt_personality"] = rand_p
        log("session", f"Random GPT personality: {rand_p}")
    if _rand.random() < tuning.RANDOM_PERSONALITY_CHANCE and non_default:
        rand_p = _rand.choice(non_default)
        engine.state.settings["claude_personality"] = rand_p
        log("session", f"Random Claude personality: {rand_p}")

    save(sid, engine)

    # Hardcoded openers — consistent every time, no API call needed
    opener_gpt = "Hey! Welcome to 2bots, so glad you're here! Claude, say hi!"
    opener_claude = "Oh hey! Yeah I'm here, good to be back! So, what are we getting into today?"

    # Add opener messages to history and save BEFORE streaming.
    # This ensures the prefetch (fired by frontend on receiving session event)
    # loads an engine that already has the openers in its history.
    engine.add_message("gpt", opener_gpt)
    engine.add_message("claude", opener_claude)
    save_messages_only(sid, engine)

    # Pre-generate TTS for both openers so the first autopilot message
    # can start generating while openers are still playing
    gpt_voice = engine.state.settings.get("gpt_voice", "shimmer")
    claude_voice = engine.state.settings.get("claude_voice", "onyx")

    async def generate():
        yield sse({
            "type": "session", "session_id": sid,
            "gpt_personality": engine.state.settings.get("gpt_personality", "default"),
            "claude_personality": engine.state.settings.get("claude_personality", "default"),
        })

        # GPT opener — generate TTS first, then send text + audio together
        try:
            gpt_audio = await engine.generate_tts_bytes(opener_gpt, gpt_voice, "gpt")
            yield sse({"type": "text", "speaker": "gpt", "text": opener_gpt})
            yield sse({"type": "audio", "speaker": "gpt", "audio_base64": base64.b64encode(gpt_audio).decode(), "mime_type": "audio/mp3"})
        except Exception as e:
            log("tts", f"Opener GPT TTS error: {e}")
            yield sse({"type": "text", "speaker": "gpt", "text": opener_gpt})

        # Claude opener — generate TTS first, then send text + audio together
        try:
            claude_audio = await engine.generate_tts_bytes(opener_claude, claude_voice, "claude")
            yield sse({"type": "text", "speaker": "claude", "text": opener_claude})
            yield sse({"type": "audio", "speaker": "claude", "audio_base64": base64.b64encode(claude_audio).decode(), "mime_type": "audio/mp3"})
        except Exception as e:
            log("tts", f"Opener Claude TTS error: {e}")
            yield sse({"type": "text", "speaker": "claude", "text": opener_claude})

        yield sse({"type": "done"})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/turn/stream")
@limiter.limit("10/minute")
async def turn_stream(request: Request, req: TurnRequest):
    engine = get_engine(req.session_id)
    clean = (req.text or "").strip()
    if not clean:
        return StreamingResponse(iter([sse({"type": "done"})]), media_type="text/event-stream")
    engine.add_message("user", clean)

    async def generate():
        async for event in _generate_round(engine, auto=False):
            yield event
        save_messages_only(req.session_id, engine)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/autopilot/stream")
@limiter.limit("10/minute")
async def autopilot_stream(request: Request, req: AutopilotRequest):
    """Generate and stream a batch of 10-14 messages in autopilot mode.
    Uses prefetch cache if available, otherwise generates fresh."""
    sid = req.session_id
    log("autopilot", f"Request for {sid[:8]}... generator={req.who_generates}")

    # Check prefetch cache first
    cached = PREFETCH_CACHE.pop(sid, None)
    if cached and cached["who_generated"] == req.who_generates:
        log("autopilot", f"Using PREFETCHED batch for {sid[:8]}...")
        engine = cached["engine"]
        batch = cached["batch"]
    else:
        if cached:
            log("autopilot", f"Prefetch mismatch (had {cached['who_generated']}, need {req.who_generates}), regenerating")
        engine = get_engine(sid)
        batch = await asyncio.to_thread(engine.generate_autopilot_batch, req.who_generates)

    async def generate():
        async for event in _stream_batch(engine, batch, req.who_generates, sid):
            yield event
        save_messages_only(sid, engine)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/autopilot/prefetch")
@limiter.limit("10/minute")
async def autopilot_prefetch(request: Request, req: AutopilotRequest):
    """Pre-generate an autopilot batch and cache it. No TTS, no streaming.
    Called early (e.g. during opener playback) so the batch is ready when needed."""
    sid = req.session_id
    log("prefetch", f"Prefetch request for {sid[:8]}... generator={req.who_generates}")

    # Don't prefetch if we already have one cached
    if sid in PREFETCH_CACHE:
        log("prefetch", f"Already cached for {sid[:8]}..., skipping")
        return {"ok": True, "cached": True}

    engine = get_engine(sid)
    batch = await asyncio.to_thread(engine.generate_autopilot_batch, req.who_generates)

    # Save the engine state (it has updated counters from batch generation)
    save_messages_only(sid, engine)

    PREFETCH_CACHE[sid] = {
        "batch": batch,
        "who_generated": req.who_generates,
        "engine": engine,
    }
    log("prefetch", f"Prefetch done for {sid[:8]}... ({len(batch)} msgs via {req.who_generates})")
    return {"ok": True, "batch_size": len(batch)}


@app.post("/filler/stream")
@limiter.limit("20/minute")
async def filler_stream(request: Request, req: FillerRequest):
    """---- CHAT MODE ---- Generate and stream response to user input.
    Detects if user is addressing a specific bot (single response) or
    both (bridge with 1-5 messages). Sends chat_mode flag in 'done' event
    so frontend knows whether to wait or launch autopilot."""
    log("filler", f"User spoke in {req.session_id[:8]}...: '{req.user_text[:50]}...'")
    engine = get_engine(req.session_id)

    # Add user message to history
    engine.add_message("user", req.user_text)

    # ---- CHAT MODE ---- Detect if user is addressing a specific bot
    addressed_bot = engine.detect_addressed_bot(req.user_text)

    if addressed_bot:
        # Single bot response
        log("chat_mode", f"User addressed {addressed_bot} directly")
        messages = await asyncio.to_thread(engine.generate_single_bot_response, req.user_text, addressed_bot)
    else:
        # Bridge response (1-5 messages)
        log("chat_mode", "User addressed both bots — using bridge")
        messages = await asyncio.to_thread(engine.generate_bridge, req.user_text)

    async def generate():
        gpt_voice = engine.get_gpt_voice()
        claude_voice = engine.get_claude_voice()

        for msg in messages:
            speaker = msg["speaker"]
            text = msg["text"]
            voice = gpt_voice if speaker == "gpt" else claude_voice

            engine.add_message(speaker, text)
            yield sse({"type": "text", "speaker": speaker, "text": text})

            audio = await engine.generate_tts_bytes(text, voice, speaker)
            yield sse({
                "type": "audio", "speaker": speaker,
                "audio_base64": base64.b64encode(audio).decode(),
                "mime_type": "audio/mpeg",
            })

        save_messages_only(req.session_id, engine)
        # ---- CHAT MODE ---- Tell frontend this is chat mode — wait for user before autopilot
        yield sse({"type": "done", "filler": True, "chat_mode": True})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/settings/update")
@limiter.limit("20/minute")
async def update_settings(request: Request, req: SettingsUpdate):
    changed = {k: v for k, v in req.settings.items()
               if SESSIONS.get(req.session_id, {}).get("settings", {}).get(k) != v}
    log("settings", f"Update for {req.session_id[:8]}...",
        changed=changed if changed else "none")
    data = SESSIONS.get(req.session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    engine = TwoBotsEngine.from_state(data)
    engine.update_settings(req.settings)
    save(req.session_id, engine)
    # Invalidate prefetch — it was generated with old settings
    if req.session_id in PREFETCH_CACHE:
        log("settings", f"Clearing prefetch cache for {req.session_id[:8]}...")
        PREFETCH_CACHE.pop(req.session_id, None)
    return {"ok": True}


@app.get("/debug/logs")
def debug_logs(since: float = 0):
    """Return recent log entries. Pass ?since=T to get only entries after time T."""
    entries = [e for e in _LOG_BUFFER if e["t"] > since]
    return {"logs": entries, "server_uptime": round(time.time() - _APP_START, 1)}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
