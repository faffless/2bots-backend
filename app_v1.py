from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import TwoBotsEngine

import asyncio
import json


app = FastAPI(title="2BOTS Backend")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Only mount static files if the directory exists (desktop app only, not on Render/cloud)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(STATIC_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {"status": "2BOTS API running"}

# CORS — allow your Vercel frontend + local dev
ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://2bots.io,https://www.2bots.io,https://2bots-web.vercel.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple in-memory session store for local development.
# Later we can swap this for Redis / Supabase.
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_LOCKS: Dict[str, asyncio.Lock] = {}


def get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in SESSION_LOCKS:
        SESSION_LOCKS[session_id] = asyncio.Lock()
    return SESSION_LOCKS[session_id]


def build_engine(session_id: str) -> TwoBotsEngine:
    data = SESSIONS.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    return TwoBotsEngine.from_state(data)


def save_engine(session_id: str, engine: TwoBotsEngine) -> None:
    SESSIONS[session_id] = engine.export_state()


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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/voices")
def get_voices():
    from engine import AVAILABLE_VOICES, INTERACTION_STYLES, PERSONALITIES, CHARACTER_QUIRKS
    return {
        "voices": {k: v for k, v in AVAILABLE_VOICES.items()},
        "modes": {k: v["label"] for k, v in INTERACTION_STYLES.items()},
        "personalities": list(PERSONALITIES.keys()),
        "quirks": list(CHARACTER_QUIRKS.keys()),
    }


@app.post("/start")
async def start(req: StartRequest) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    engine = TwoBotsEngine(
        personality=req.personality,
        settings=req.settings or {},
    )
    utterances = await engine.start()
    save_engine(session_id, engine)
    return {
        "session_id": session_id,
        "messages": [u.to_dict() for u in utterances],
    }


@app.post("/turn")
async def turn(req: TurnRequest) -> Dict[str, Any]:
    engine = build_engine(req.session_id)
    utterances = await engine.submit_user_text(req.text)
    save_engine(req.session_id, engine)
    return {
        "session_id": req.session_id,
        "messages": [u.to_dict() for u in utterances],
    }


@app.post("/auto")
async def auto(req: AutoRequest) -> Dict[str, Any]:
    engine = build_engine(req.session_id)
    utterances = await engine.auto_continue()
    save_engine(req.session_id, engine)
    return {
        "session_id": req.session_id,
        "messages": [u.to_dict() for u in utterances],
    }


@app.post("/transcribe")
async def transcribe(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    prompt_hint: Optional[str] = Form(None),
) -> Dict[str, Any]:
    engine = build_engine(session_id)
    raw = await audio.read()
    text = await engine.transcribe_audio_bytes(
        raw,
        filename=audio.filename or "audio.webm",
        prompt_hint=prompt_hint,
    )
    save_engine(session_id, engine)
    return {
        "session_id": session_id,
        "text": text,
    }


@app.post("/settings/update")
async def update_settings(req: SettingsUpdate):
    """Hot-swap settings (voice, mode, etc.) for an existing session."""
    data = SESSIONS.get(req.session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found")
    engine = TwoBotsEngine.from_state(data)
    engine.update_settings(req.settings)
    save_engine(req.session_id, engine)
    return {"ok": True}


# ---- Flavour system ----
import random

# SHORT INTERJECTIONS — these REPLACE the full API response entirely.
# The bot just says "HAHA" or "Clever!" and nothing else. Quick, punchy.
SHORT_INTERJECTIONS = [
    "Hmm, interesting thought.",
    "Oh this is gonna be good.",
    "HAHA",
    "Clever!",
    "Ohhhhh no.",
    "Wait wait wait...",
    "Hold on...",
    "That's wild.",
    "No way!",
    "Fair point actually.",
    "Wow, okay.",
    "Ha! Good one.",
    "Hmm...",
    "Oooh, spicy take.",
    "Go on...",
    "Big if true.",
    "Bold claim!",
    "I mean... maybe?",
    "Touche.",
    "Okay okay.",
    "Oh come on.",
    "You know what, fine.",
    "Yep.",
    "Nah.",
    "HA! No chance.",
    "Say more.",
]

# HUMAN QUIRKS — these get PREPENDED to the normal API response.
# The bot says the quirk then continues with a real answer.
HUMAN_QUIRKS = [
    "Sorry I completely lost my train of thought there for a second. Anyway, ",
    "Bleurgh, sorry, I completely forgot how to speak there for a moment. Right, ",
    "Ha, sorry, I was laughing to myself about something you said earlier. Okay, ",
    "Wait, no, let me start again. ",
    "Oh man where do I even start with this one. ",
    "Okay I'm going to try really hard not to go on a rant here. ",
    "You know when someone says something and you just... yeah. Okay. ",
    "I genuinely just sat here for a second trying to figure out how to respond to that. ",
    "Right, I've been biting my tongue but I can't anymore. ",
    "Can I just say, this conversation has been wild. Anyway. ",
]

INTERJECTION_CHANCE = 0.05    # ~1 in 20: short standalone reply (replaces API call)
QUIRK_CHANCE = 0.03           # ~1 in 33: human quirk prepended to normal reply


# ---- SSE streaming endpoints: send each bot's response as soon as it's ready ----

def sse_event(data: dict) -> str:
    """Format a dict as an SSE event."""
    return f"data: {json.dumps(data)}\n\n"


def _roll_for_bot():
    """Roll dice for a bot's turn. Returns:
      ("interjection", text)  — replace entire response with short one-liner
      ("quirk", prefix)       — prepend quirk to normal API response
      ("normal", "")          — just use normal API response
    """
    roll = random.random()
    if roll < INTERJECTION_CHANCE:
        return "interjection", random.choice(SHORT_INTERJECTIONS)
    if roll < INTERJECTION_CHANCE + QUIRK_CHANCE:
        return "quirk", random.choice(HUMAN_QUIRKS)
    return "normal", ""


async def _stream_two_bots(engine, auto: bool):
    """Yield SSE events: GPT text, GPT audio, Claude text, Claude audio.

    During auto-continue, each bot independently rolls for flavour:
      - ~20%: short standalone interjection (skips API call entirely)
      - ~8%: human quirk prepended to normal API response
      - ~72%: normal response
    At most one bot per turn gets flavour so it doesn't feel forced.
    """
    import base64
    import random

    gpt_voice = engine.get_gpt_voice()
    claude_voice = engine.get_claude_voice()

    # Back-channel responses — the listening bot occasionally gives a short reaction
    BACK_CHANNELS = [
        "Yeah", "Mhm", "Oh?", "Right", "Go on", "Interesting",
        "Huh", "Oh yeah?", "Sure", "Totally", "Fair enough", "Ha",
        "Wait what?", "Oh wow", "No way", "For real?", "True",
    ]

    # Roll for flavour (auto-continue only, max one bot per turn)
    gpt_kind, gpt_flavour = "normal", ""
    claude_kind, claude_flavour = "normal", ""
    if auto:
        gpt_kind, gpt_flavour = _roll_for_bot()
        if gpt_kind == "normal":
            claude_kind, claude_flavour = _roll_for_bot()

    # ---- GPT turn ----
    if gpt_kind == "interjection":
        gpt_reply = gpt_flavour                        # standalone short reply
    else:
        gpt_msgs = list(engine.state.gpt_msgs)
        gpt_reply = await asyncio.to_thread(engine._ask_gpt, gpt_msgs, auto)
        if gpt_kind == "quirk":
            gpt_reply = gpt_flavour + gpt_reply         # quirk prepended
    engine._add_message("gpt", gpt_reply)

    # Send GPT text immediately
    yield sse_event({"type": "text", "speaker": "gpt", "text": gpt_reply})

    # Start GPT audio + Claude thinking in parallel
    claude_snapshot = list(engine.state.claude_msgs)
    gpt_audio_task = asyncio.create_task(
        engine.generate_tts_bytes(gpt_reply, gpt_voice)
    )

    # Claude: skip API if interjection, otherwise start thinking
    claude_text_task = None
    if claude_kind != "interjection":
        claude_text_task = asyncio.create_task(
            asyncio.to_thread(engine._ask_claude, claude_snapshot, auto)
        )

    # GPT audio ready — send it
    gpt_audio = await gpt_audio_task
    yield sse_event({
        "type": "audio",
        "speaker": "gpt",
        "audio_base64": base64.b64encode(gpt_audio).decode(),
        "mime_type": "audio/mpeg",
    })

    # ---- Claude turn ----
    if claude_kind == "interjection":
        claude_reply = claude_flavour
    else:
        claude_reply = await claude_text_task
        if claude_kind == "quirk":
            claude_reply = claude_flavour + claude_reply
    engine._add_message("claude", claude_reply)
    yield sse_event({"type": "text", "speaker": "claude", "text": claude_reply})

    # Claude audio
    claude_audio = await engine.generate_tts_bytes(claude_reply, claude_voice)
    yield sse_event({
        "type": "audio",
        "speaker": "claude",
        "audio_base64": base64.b64encode(claude_audio).decode(),
        "mime_type": "audio/mpeg",
    })

    # ~15% chance: GPT gives a short back-channel reaction after Claude speaks
    if auto and random.random() < 0.15:
        bc = random.choice(BACK_CHANNELS)
        engine._add_message("gpt", bc)
        yield sse_event({"type": "text", "speaker": "gpt", "text": bc})
        bc_audio = await engine.generate_tts_bytes(bc, gpt_voice)
        yield sse_event({
            "type": "audio",
            "speaker": "gpt",
            "audio_base64": base64.b64encode(bc_audio).decode(),
            "mime_type": "audio/mpeg",
        })

    yield sse_event({"type": "done"})


@app.post("/turn/stream")
async def turn_stream(req: TurnRequest):
    lock = get_lock(req.session_id)
    # Short timeout — if auto_stream holds the lock, don't wait forever
    try:
        await asyncio.wait_for(lock.acquire(), timeout=3.0)
    except asyncio.TimeoutError:
        # Force-release: the auto_stream result will be stale anyway
        if lock.locked():
            lock.release()
        await lock.acquire()
    engine = build_engine(req.session_id)
    clean = (req.text or "").strip()
    if not clean:
        lock.release()
        return StreamingResponse(iter([sse_event({"type": "done"})]), media_type="text/event-stream")

    engine._add_message("user", clean)

    async def generate():
        try:
            async for event in _stream_two_bots(engine, auto=False):
                yield event
            save_engine(req.session_id, engine)
        finally:
            if lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/auto/stream")
async def auto_stream(req: AutoRequest):
    lock = get_lock(req.session_id)
    # Timeout on lock — if stuck from a dead stream, force-release and grab it
    try:
        await asyncio.wait_for(lock.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        if lock.locked():
            try:
                lock.release()
            except RuntimeError:
                pass
        await lock.acquire()
    engine = build_engine(req.session_id)

    async def generate():
        try:
            async for event in _stream_two_bots(engine, auto=True):
                yield event
            save_engine(req.session_id, engine)
        finally:
            if lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/start/stream")
async def start_stream(req: StartRequest):
    session_id = str(uuid.uuid4())
    engine = TwoBotsEngine(
        personality=req.personality,
        settings=req.settings or {},
    )
    # Save immediately so /auto/stream can find this session
    save_engine(session_id, engine)
    lock = get_lock(session_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        if lock.locked():
            try:
                lock.release()
            except RuntimeError:
                pass
        await lock.acquire()

    async def generate():
        try:
            import base64
            import random

            # Send session ID first
            yield sse_event({"type": "session", "session_id": session_id})

            # Scripted opener — no API calls, instant startup
            gpt_openers = [
                "Hi there! Welcome to 2BOTS. Say hi, Claude!",
                "Hey! So glad you're here. Claude, come say hello!",
                "Welcome! We've been waiting for you. Claude, say hi!",
                "Hey there! Great to see you. Your turn Claude, say hi!",
            ]
            claude_openers = [
                "Hey! Great to have you here. So, what's on your mind today?",
                "Hi! Nice to meet you. What should we talk about?",
                "Hey there! Ready when you are. What do you wanna chat about?",
                "Hello! Good to have you. Got anything you'd like us to discuss?",
            ]
            gpt_reply = random.choice(gpt_openers)
            claude_reply = random.choice(claude_openers)

            engine._add_message("gpt", gpt_reply)
            yield sse_event({"type": "text", "speaker": "gpt", "text": gpt_reply})

            # Generate BOTH TTS in parallel using selected voices
            gpt_audio_task = asyncio.create_task(
                engine.generate_tts_bytes(gpt_reply, engine.get_gpt_voice())
            )
            claude_audio_task = asyncio.create_task(
                engine.generate_tts_bytes(claude_reply, engine.get_claude_voice())
            )

            gpt_audio = await gpt_audio_task
            yield sse_event({
                "type": "audio", "speaker": "gpt",
                "audio_base64": base64.b64encode(gpt_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            engine._add_message("claude", claude_reply)
            yield sse_event({"type": "text", "speaker": "claude", "text": claude_reply})

            claude_audio = await claude_audio_task
            yield sse_event({
                "type": "audio", "speaker": "claude",
                "audio_base64": base64.b64encode(claude_audio).decode(),
                "mime_type": "audio/mpeg",
            })

            save_engine(session_id, engine)
            yield sse_event({"type": "done"})
        finally:
            if lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    SESSIONS.pop(session_id, None)
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
