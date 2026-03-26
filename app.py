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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

app = FastAPI(title="2BOTS Backend v2")

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
        SESSIONS[sid]["gpt_motivation"] = engine.state.gpt_motivation
        SESSIONS[sid]["claude_motivation"] = engine.state.claude_motivation
        SESSIONS[sid]["gpt_motivation_rounds_left"] = engine.state.gpt_motivation_rounds_left
        SESSIONS[sid]["claude_motivation_rounds_left"] = engine.state.claude_motivation_rounds_left
        SESSIONS[sid]["temperature"] = engine.state.temperature
        SESSIONS[sid]["next_bot_boosted"] = engine.state.next_bot_boosted
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

class AutoRequest(BaseModel):
    session_id: str

class SettingsUpdate(BaseModel):
    session_id: str
    settings: Dict[str, Any]


# ---- Core SSE generator ----

async def _generate_round(engine: TwoBotsEngine, auto: bool, opener: bool = False):
    """Yield SSE events for one round: GPT text+audio, Claude text+audio.
    Includes Experiment 1 features: cook rounds, triggers, double turns, 4th wall."""
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

    # EXPERIMENT 1: determine round modifiers
    is_cook = not use_filler and not opener and engine.exp1_is_cook_round()
    is_double = not use_filler and not opener and not is_cook and engine.exp1_is_double_turn()
    gpt_fourth_wall = not use_filler and not opener and engine.exp1_is_fourth_wall()
    claude_fourth_wall = not use_filler and not opener and engine.exp1_is_fourth_wall()
    gpt_boosted = engine.state.next_bot_boosted  # from previous round's trigger
    engine.state.next_bot_boosted = False  # reset

    if use_filler:
        log("filler", f"Injecting filler for {filler_bot}")
    if is_cook:
        log("exp1", "LET THEM COOK round")
    if is_double:
        log("exp1", "DOUBLE TURN round")

    # 1. GPT generates text (or filler)
    if filler_bot == "gpt":
        gpt_text = engine.get_filler("gpt")
        if engine.exp1_enabled("EXP1_FOURTH_WALL") and _rand.random() < 0.15:
            gpt_text = engine.exp1_get_fourth_wall_filler()
        log("gpt", f"FILLER: '{gpt_text}'")
    else:
        gpt_text = await asyncio.to_thread(
            engine.ask_gpt, auto, opener,
            cook=is_cook, fourth_wall=gpt_fourth_wall, boosted=gpt_boosted
        )
    engine.add_message("gpt", gpt_text)
    t1 = time.time()
    gpt_words = len(gpt_text.split())
    if filler_bot != "gpt":
        log("gpt", f"Text done: {gpt_words}w in {t1-t0:.1f}s", words=gpt_words, secs=round(t1-t0, 1))
    yield sse({"type": "text", "speaker": "gpt", "text": gpt_text})

    # EXPERIMENT 1: trigger detection on GPT's text
    if not use_filler:
        triggers = engine.exp1_check_triggers(gpt_text)
        claude_boosted = triggers["boost_next"]
        engine.exp1_update_temperature(gpt_text)
    else:
        claude_boosted = False

    # 2. Parallel: GPT TTS + Claude thinks (or filler)
    gpt_tts_task = asyncio.create_task(engine.generate_tts_bytes(gpt_text, gpt_voice))

    if filler_bot == "claude":
        claude_text = engine.get_filler("claude")
        if engine.exp1_enabled("EXP1_FOURTH_WALL") and _rand.random() < 0.15:
            claude_text = engine.exp1_get_fourth_wall_filler()
        log("claude", f"FILLER: '{claude_text}'")
        claude_text_task = None
    else:
        claude_text_task = asyncio.create_task(asyncio.to_thread(
            engine.ask_claude, auto, opener,
            cook=is_cook, fourth_wall=claude_fourth_wall, boosted=claude_boosted
        ))

    gpt_audio = await gpt_tts_task
    t2 = time.time()
    log("gpt", f"TTS done in {t2-t1:.1f}s ({gpt_words}w)", secs=round(t2-t1, 1))
    yield sse({
        "type": "audio", "speaker": "gpt",
        "audio_base64": base64.b64encode(gpt_audio).decode(),
        "mime_type": "audio/mpeg",
    })

    # EXPERIMENT 1: Double turn — GPT speaks again before Claude
    if is_double and filler_bot != "gpt":
        double_text = await asyncio.to_thread(
            engine.ask_gpt, True, False, double_turn=True
        )
        engine.add_message("gpt", double_text)
        log("exp1", f"GPT double turn: '{double_text}'")
        yield sse({"type": "text", "speaker": "gpt", "text": double_text})
        double_audio = await engine.generate_tts_bytes(double_text, gpt_voice)
        yield sse({
            "type": "audio", "speaker": "gpt",
            "audio_base64": base64.b64encode(double_audio).decode(),
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

    # EXPERIMENT 1: trigger detection on Claude's text → boost next round's GPT
    if not use_filler:
        triggers = engine.exp1_check_triggers(claude_text)
        engine.state.next_bot_boosted = triggers["boost_next"]
        engine.exp1_update_temperature(claude_text)

    claude_audio = await engine.generate_tts_bytes(claude_text, claude_voice)
    t4 = time.time()
    log("claude", f"TTS done in {t4-t3:.1f}s ({claude_words}w)", secs=round(t4-t3, 1))

    total = round(t4-t0, 1)
    log("round", f"DONE in {total}s (GPT {gpt_words}w + Claude {claude_words}w) temp={engine.state.temperature:.1f}", total_secs=total)
    yield sse({
        "type": "audio", "speaker": "claude",
        "audio_base64": base64.b64encode(claude_audio).decode(),
        "mime_type": "audio/mpeg",
    })

    # Tick counters (only for non-filler rounds)
    if not use_filler:
        engine.tick_filler()
    engine.tick_motivations()

    # Send motivation state to frontend
    yield sse({
        "type": "motivations",
        "gpt": engine.state.gpt_motivation or "",
        "claude": engine.state.claude_motivation or "",
    })

    yield sse({"type": "done"})


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
async def start_stream(req: StartRequest):
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

    async def generate():
        yield sse({
            "type": "session", "session_id": sid,
            "gpt_personality": engine.state.settings.get("gpt_personality", "default"),
            "claude_personality": engine.state.settings.get("claude_personality", "default"),
        })
        async for event in _generate_round(engine, auto=True, opener=True):
            yield event
        save_messages_only(sid, engine)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/turn/stream")
async def turn_stream(req: TurnRequest):
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


@app.post("/auto/stream")
async def auto_stream(req: AutoRequest):
    log("auto", f"Request for {req.session_id[:8]}...")
    engine = get_engine(req.session_id)

    async def generate():
        async for event in _generate_round(engine, auto=True):
            yield event
        save_messages_only(req.session_id, engine)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/settings/update")
async def update_settings(req: SettingsUpdate):
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
