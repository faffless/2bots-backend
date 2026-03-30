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
import random
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
    PINGPONG_MODES,
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
        # ---- PING-PONG MODE ----
        SESSIONS[sid]["pingpong_msg_count"] = engine.state.pingpong_msg_count
        SESSIONS[sid]["pingpong_conclusions"] = list(engine.state.pingpong_conclusions)
        SESSIONS[sid]["pingpong_complete"] = engine.state.pingpong_complete
        SESSIONS[sid]["pingpong_reviews"] = list(engine.state.pingpong_reviews)
        SESSIONS[sid]["debate_score_gpt"] = engine.state.debate_score_gpt
        SESSIONS[sid]["debate_score_claude"] = engine.state.debate_score_claude
        SESSIONS[sid]["milestone_target"] = engine.state.milestone_target
        SESSIONS[sid]["exchanges_per_milestone"] = engine.state.exchanges_per_milestone
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

# ---- PING-PONG MODE ----
class ResearchRequest(BaseModel):
    session_id: str
    who: str  # "gpt" or "claude" — which bot responds this turn
# ---- END PING-PONG MODE ----


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
    """Stream a pre-generated autopilot batch: pre-generates ALL TTS first,
    then streams text+audio pairs with zero gap between messages.
    Kicks off prefetch of the next batch immediately after TTS is done."""
    t0 = time.time()
    gpt_voice = engine.get_gpt_voice()
    claude_voice = engine.get_claude_voice()
    next_generator = "claude" if who_generated == "gpt" else "gpt"

    log("autopilot", f"Buffering batch of {len(batch)} messages (generated by {who_generated})")

    # Add ALL batch messages to history UPFRONT and save immediately.
    for msg in batch:
        engine.add_message(msg["speaker"], msg["text"])
    save_messages_only(sid, engine)

    # Pre-generate ALL TTS for the batch (parallel)
    async def _gen_tts(msg):
        voice = gpt_voice if msg["speaker"] == "gpt" else claude_voice
        return await engine.generate_tts_bytes(msg["text"], voice, msg["speaker"])

    batch_audio = await asyncio.gather(*[_gen_tts(m) for m in batch])

    t_tts = time.time()
    log("autopilot", f"All TTS generated in {t_tts-t0:.1f}s for {len(batch)} msgs")

    # Kick off prefetch immediately — audio playback will take a while
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

    # Stream all text+audio pairs (TTS already ready — zero gap)
    for msg, audio in zip(batch, batch_audio):
        yield sse({"type": "text", "speaker": msg["speaker"], "text": msg["text"]})
        yield sse({
            "type": "audio", "speaker": msg["speaker"],
            "audio_base64": base64.b64encode(audio).decode(),
            "mime_type": "audio/mpeg",
        })

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
    """First round — buffers ALL initial content + TTS before streaming.
    Ping-pong: first 2 research responses + TTS.
    Scripted: opener + full autopilot batch + all TTS.
    Loading screen covers the entire generation phase."""
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

    current_mode = engine.state.settings.get("mode", "conversation")
    is_pingpong = current_mode in PINGPONG_MODES

    # Pick which bot kicks things off
    opener_who = _rand.choice(["gpt", "claude"])
    other_who = "claude" if opener_who == "gpt" else "gpt"

    gpt_voice = engine.state.settings.get("gpt_voice", "shimmer")
    claude_voice = engine.state.settings.get("claude_voice", "onyx")
    opener_voice = gpt_voice if opener_who == "gpt" else claude_voice
    other_voice = gpt_voice if other_who == "gpt" else claude_voice

    async def generate():
        yield sse({
            "type": "session", "session_id": sid,
            "gpt_personality": engine.state.settings.get("gpt_personality", "default"),
            "claude_personality": engine.state.settings.get("claude_personality", "default"),
        })

        if is_pingpong:
            # ---- PING-PONG: Buffer first 2 messages + TTS ----
            t0 = time.time()
            log("start", f"Ping-pong buffered start: generating {opener_who}'s opener...")

            # 1. Generate first response (opener with plan)
            opener_text = await asyncio.to_thread(engine.generate_research_response, opener_who)
            engine.add_message(opener_who, opener_text)
            engine.state.pingpong_msg_count += 1

            # 2. Parallel: opener TTS + second response text
            log("start", "Parallel: opener TTS + second response...")
            opener_tts_task = engine.generate_tts_bytes(opener_text, opener_voice, opener_who)
            second_text_task = asyncio.to_thread(engine.generate_research_response, other_who)
            opener_audio, second_text = await asyncio.gather(opener_tts_task, second_text_task)

            engine.add_message(other_who, second_text)
            engine.state.pingpong_msg_count += 1

            # 3. Generate second TTS
            second_audio = await engine.generate_tts_bytes(second_text, other_voice, other_who)
            save_messages_only(sid, engine)

            t1 = time.time()
            log("start", f"Ping-pong buffer ready in {t1-t0:.1f}s — streaming 2 messages")

            # 4. Calculate countdown info
            epm = engine.state.exchanges_per_milestone
            mode = engine._s("mode") or "conversation"

            # 5. Stream all buffered content
            text_event_1 = {"type": "text", "speaker": opener_who, "text": opener_text}
            if mode != "conversation":
                msgs_until = epm - 1
                if msgs_until > 0:
                    text_event_1["msgs_until_review"] = msgs_until
            yield sse(text_event_1)
            yield sse({
                "type": "audio", "speaker": opener_who,
                "audio_base64": base64.b64encode(opener_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            text_event_2 = {"type": "text", "speaker": other_who, "text": second_text}
            if mode != "conversation":
                msgs_until = epm - 2
                if msgs_until > 0:
                    text_event_2["msgs_until_review"] = msgs_until
            yield sse(text_event_2)
            yield sse({
                "type": "audio", "speaker": other_who,
                "audio_base64": base64.b64encode(second_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            # Prefetch next response in background
            next_who = opener_who  # alternates back
            async def _prefetch():
                try:
                    pf_engine = get_engine(sid)
                    pf_text = await asyncio.to_thread(pf_engine.generate_research_response, next_who)
                    RESEARCH_PREFETCH[sid] = {"who": next_who, "text": pf_text}
                    log("research", f"Prefetched {next_who}'s response for {sid[:8]}...")
                except Exception as e:
                    log("research", f"Start prefetch failed: {e}")
            asyncio.create_task(_prefetch())

            yield sse({"type": "done", "next_who": next_who})

        else:
            # ---- SCRIPTED: Buffer opener + full batch + all TTS ----
            t0 = time.time()
            log("start", f"Scripted buffered start: generating opener + batch...")

            # 1. Generate opener text
            opener_text = await asyncio.to_thread(engine.generate_scripted_opener, opener_who)
            engine.add_message(opener_who, opener_text)

            # 2. Parallel: opener TTS + batch text generation
            batch_generator = "claude"
            log("start", "Parallel: opener TTS + batch generation...")
            opener_tts_task = engine.generate_tts_bytes(opener_text, opener_voice, opener_who)
            batch_task = asyncio.to_thread(engine.generate_autopilot_batch, batch_generator)
            opener_audio, batch = await asyncio.gather(opener_tts_task, batch_task)

            # 3. Add batch to history
            for msg in batch:
                engine.add_message(msg["speaker"], msg["text"])

            # 4. Generate ALL TTS for batch (parallel)
            log("start", f"Generating TTS for {len(batch)} batch messages (parallel)...")
            async def _gen_tts(msg):
                v = gpt_voice if msg["speaker"] == "gpt" else claude_voice
                return await engine.generate_tts_bytes(msg["text"], v, msg["speaker"])

            batch_audio = await asyncio.gather(*[_gen_tts(m) for m in batch])

            save_messages_only(sid, engine)

            t1 = time.time()
            log("start", f"Scripted buffer ready in {t1-t0:.1f}s — streaming opener + {len(batch)} msgs")

            # 5. Stream everything: opener then batch
            yield sse({"type": "text", "speaker": opener_who, "text": opener_text})
            yield sse({
                "type": "audio", "speaker": opener_who,
                "audio_base64": base64.b64encode(opener_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            for msg, audio in zip(batch, batch_audio):
                yield sse({"type": "text", "speaker": msg["speaker"], "text": msg["text"]})
                yield sse({
                    "type": "audio", "speaker": msg["speaker"],
                    "audio_base64": base64.b64encode(audio).decode(),
                    "mime_type": "audio/mpeg",
                })

            # Prefetch next batch in background
            next_generator = "gpt" if batch_generator == "claude" else "claude"
            async def _prefetch_batch():
                try:
                    pf_engine = get_engine(sid)
                    pf_batch = await asyncio.to_thread(pf_engine.generate_autopilot_batch, next_generator)
                    PREFETCH_CACHE[sid] = {
                        "batch": pf_batch,
                        "who_generated": next_generator,
                        "engine": pf_engine,
                    }
                    log("prefetch", f"Start prefetch ready for {sid[:8]}... ({len(pf_batch)} msgs via {next_generator})")
                except Exception as e:
                    log("prefetch", f"Start prefetch failed: {e}")
            asyncio.create_task(_prefetch_batch())

            yield sse({"type": "done", "next_generator": next_generator, "batch_size": len(batch)})

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

    # ---- CHAT MODE ---- Invalidate prefetch cache — user spoke, so any cached batch is stale
    if req.session_id in PREFETCH_CACHE:
        log("chat_mode", f"Clearing stale prefetch cache for {req.session_id[:8]}... (user spoke)")
        del PREFETCH_CACHE[req.session_id]

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

        # Pre-generate all TTS before streaming (zero gap between messages)
        msg_audio = []
        for msg in messages:
            voice = gpt_voice if msg["speaker"] == "gpt" else claude_voice
            audio = await engine.generate_tts_bytes(msg["text"], voice, msg["speaker"])
            msg_audio.append(audio)

        for msg, audio in zip(messages, msg_audio):
            engine.add_message(msg["speaker"], msg["text"])
            yield sse({"type": "text", "speaker": msg["speaker"], "text": msg["text"]})
            yield sse({
                "type": "audio", "speaker": msg["speaker"],
                "audio_base64": base64.b64encode(audio).decode(),
                "mime_type": "audio/mpeg",
            })

        save_messages_only(req.session_id, engine)
        # ---- CHAT MODE ---- Tell frontend this is chat mode — wait for user before autopilot
        yield sse({"type": "done", "filler": True, "chat_mode": True})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---- PING-PONG MODE ----
# Prefetch cache for research mode: stores the next bot's pre-generated response
# Key: session_id, Value: {"who": str, "text": str, "task": asyncio.Task | None}
RESEARCH_PREFETCH: Dict[str, Dict[str, Any]] = {}


@app.post("/research/stream")
@limiter.limit("20/minute")
async def research_stream(request: Request, req: ResearchRequest):
    """---- PING-PONG MODE (research, debate, advice) ----
    Generate a single response from one bot (real API call, not scripted).
    Streams text + TTS as SSE events, same format as autopilot.
    Also prefetches the OTHER bot's response while TTS plays."""
    sid = req.session_id
    who = req.who
    log("research", f"Ping-pong turn: {who} for {sid[:8]}...")

    # Check research prefetch cache first
    cached = RESEARCH_PREFETCH.pop(sid, None)
    if cached and cached["who"] == who and "text" in cached:
        log("research", f"Using PREFETCHED response for {who} in {sid[:8]}...")
        text = cached["text"]
        engine = get_engine(sid)
    else:
        if cached:
            log("research", f"Prefetch mismatch (had {cached.get('who')}, need {who}), regenerating")
        engine = get_engine(sid)
        text = await asyncio.to_thread(engine.generate_research_response, who)

    # Read the current mode for mode-aware events
    mode = engine._s("mode") or "conversation"

    # Add the response to conversation history
    engine.add_message(who, text)
    engine.state.pingpong_msg_count += 1
    msg_count = engine.state.pingpong_msg_count
    save_messages_only(sid, engine)

    other = "claude" if who == "gpt" else "gpt"

    # Check if review is due — using AI-chosen exchanges_per_milestone
    epm = engine.state.exchanges_per_milestone  # 8-12, set by opener AI
    cycle_position = msg_count % epm  # 0 means milestone reached
    current_mode = engine._s("mode") or "conversation"
    needs_review = (cycle_position == 0 and msg_count > 0 and not engine.state.pingpong_complete and current_mode != "conversation")
    milestone_word = {"debate": "motion", "advice": "recommendation", "help_me_decide": "decision"}.get(current_mode, "finding")
    log("research", f"MSG COUNT: {msg_count}, cycle_position: {cycle_position}/{epm}, needs_review: {needs_review}, target: {engine.state.milestone_target} {milestone_word}s")

    # Determine who reviews: alternates each cycle
    cycle_number = msg_count // epm
    if cycle_number % 2 == 1:
        reviewer = "gpt"
        responder = "claude"
    else:
        reviewer = "claude"
        responder = "gpt"

    initial_conclusions = len(engine.state.pingpong_conclusions)
    initial_complete = engine.state.pingpong_complete

    async def generate():
        voice = engine.get_gpt_voice() if who == "gpt" else engine.get_claude_voice()

        # Generate TTS BEFORE streaming text (eliminates text-to-audio gap)
        t0 = time.time()
        audio = await engine.generate_tts_bytes(text, voice, who)
        t1 = time.time()
        log(who, f"Ping-pong TTS in {t1-t0:.1f}s")

        # Send text event with conclusion info
        text_event = {"type": "text", "speaker": who, "text": text}
        if initial_conclusions > 0:
            text_event["conclusions"] = initial_conclusions
        if initial_complete:
            text_event["pingpong_complete"] = True

        # Send countdown: messages until next milestone (not for conversation mode)
        if current_mode != "conversation":
            msgs_until_review = epm - (msg_count % epm) if (msg_count % epm) != 0 else 0
            if msgs_until_review > 0 and not initial_complete:
                text_event["msgs_until_review"] = msgs_until_review

        yield sse(text_event)
        yield sse({
            "type": "audio", "speaker": who,
            "audio_base64": base64.b64encode(audio).decode(),
            "mime_type": "audio/mpeg",
        })

        # Forced review cycle
        if needs_review:
            log("research", f"Forced review at msg #{msg_count} — {reviewer} reviews, {responder} responds")

            # Tell frontend the threshold has been reached
            yield sse({"type": "research_status", "event": "threshold_reached", "mode": mode})

            # Step 1: Reviewer proposes a finding/judgement
            review_text = await asyncio.to_thread(engine.generate_research_review, reviewer)
            engine.add_message(reviewer, review_text)
            engine.state.pingpong_msg_count += 1
            save_messages_only(sid, engine)

            # Generate review TTS before streaming text
            review_voice = engine.get_gpt_voice() if reviewer == "gpt" else engine.get_claude_voice()
            r_audio = await engine.generate_tts_bytes(review_text, review_voice, reviewer)
            yield sse({"type": "text", "speaker": reviewer, "text": review_text})
            yield sse({
                "type": "audio", "speaker": reviewer,
                "audio_base64": base64.b64encode(r_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            # Step 2: Responder agrees/disagrees
            respond_result = await asyncio.to_thread(engine.generate_research_respond, responder, review_text)
            respond_text, agreed = respond_result
            engine.add_message(responder, respond_text)
            engine.state.pingpong_msg_count += 1
            save_messages_only(sid, engine)

            # Generate respond TTS before streaming text
            respond_voice = engine.get_gpt_voice() if responder == "gpt" else engine.get_claude_voice()
            s_audio = await engine.generate_tts_bytes(respond_text, respond_voice, responder)
            updated_conclusions = len(engine.state.pingpong_conclusions)
            resp_event = {"type": "text", "speaker": responder, "text": respond_text, "conclusions": updated_conclusions}
            if engine.state.pingpong_complete:
                resp_event["pingpong_complete"] = True
            yield sse(resp_event)
            yield sse({
                "type": "audio", "speaker": responder,
                "audio_base64": base64.b64encode(s_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            # Tell frontend whether milestone was reached
            conclusion_num = updated_conclusions
            milestone_total = engine.state.milestone_target
            if agreed:
                # Only send the latest conclusion text, not the full list
                latest_conclusion = engine.state.pingpong_conclusions[-1] if engine.state.pingpong_conclusions else review_text.strip()
                status_event = {
                    "type": "research_status",
                    "event": "conclusion_reached",
                    "conclusion_num": conclusion_num,
                    "milestone_total": milestone_total,
                    "latest_conclusion": latest_conclusion,
                    "mode": mode,
                }
                # For debate, also send scores and winner
                if mode == "debate":
                    status_event["debate_score_gpt"] = engine.state.debate_score_gpt
                    status_event["debate_score_claude"] = engine.state.debate_score_claude
                yield sse(status_event)
            else:
                yield sse({"type": "research_status", "event": "conclusion_rejected", "mode": mode})

        # Start prefetch for the OTHER bot (skip if complete)
        if not engine.state.pingpong_complete:
            async def _prefetch_other():
                try:
                    pf_engine = get_engine(sid)
                    pf_text = await asyncio.to_thread(pf_engine.generate_research_response, other)
                    RESEARCH_PREFETCH[sid] = {"who": other, "text": pf_text}
                    log("research", f"Prefetched {other}'s response for {sid[:8]}...")
                except Exception as e:
                    log("research", f"Prefetch failed: {e}")
            asyncio.create_task(_prefetch_other())

        # Tell frontend who goes next
        done_event = {"type": "done", "next_who": other}
        if engine.state.pingpong_complete:
            done_event["pingpong_complete"] = True
            done_event["milestone_target"] = engine.state.milestone_target
            done_event["conclusions"] = list(engine.state.pingpong_conclusions)
            done_event["mode"] = mode
            if mode == "debate":
                done_event["debate_score_gpt"] = engine.state.debate_score_gpt
                done_event["debate_score_claude"] = engine.state.debate_score_claude
        yield sse(done_event)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/research/discard-prefetch")
@limiter.limit("20/minute")
async def research_discard_prefetch(request: Request, req: AutopilotRequest):
    """Discard any prefetched ping-pong response (called when user interrupts)."""
    sid = req.session_id
    if sid in RESEARCH_PREFETCH:
        log("research", f"Discarding prefetched response for {sid[:8]}... (user interrupted)")
        del RESEARCH_PREFETCH[sid]
    return {"ok": True}
# ---- END PING-PONG MODE ----


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
