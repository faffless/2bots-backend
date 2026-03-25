from __future__ import annotations

import asyncio
import io
import os
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
import edge_tts
import openai


DEFAULTS = {
    "user_timeout": 3.0,
    "done_silence": 2.5,
    "interrupt_sens": 0.04,
    "bot_max_tokens": 60,
    "speech_rate": "+10%",
    "mic_sens": 0.004,
    "gpt_voice": "en-US-AvaNeural",
    "claude_voice": "en-US-AndrewNeural",
    "conversation_mode": "casual",
}

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-4o-mini"

# ---- Available voices for the frontend to choose from ----
AVAILABLE_VOICES = {
    # US
    "en-US-AvaNeural":     "Ava (US Female)",
    "en-US-AndrewNeural":  "Andrew (US Male)",
    "en-US-JennyNeural":   "Jenny (US Female)",
    "en-US-GuyNeural":     "Guy (US Male)",
    "en-US-AriaNeural":    "Aria (US Female)",
    "en-US-DavisNeural":   "Davis (US Male)",
    "en-US-EmmaNeural":    "Emma (US Female)",
    "en-US-BrandonNeural": "Brandon (US Male)",
    # UK
    "en-GB-SoniaNeural":   "Sonia (UK Female)",
    "en-GB-RyanNeural":    "Ryan (UK Male)",
    "en-GB-LibbyNeural":   "Libby (UK Female)",
    "en-GB-ThomasNeural":  "Thomas (UK Male)",
    "en-GB-MaisieNeural":  "Maisie (UK Young Female)",
    # Australian
    "en-AU-NatashaNeural": "Natasha (Australian Female)",
    "en-AU-WilliamNeural": "William (Australian Male)",
    # Irish
    "en-IE-EmilyNeural":   "Emily (Irish Female)",
    "en-IE-ConnorNeural":  "Connor (Irish Male)",
    # Indian
    "en-IN-NeerjaNeural":  "Neerja (Indian Female)",
    "en-IN-PrabhatNeural": "Prabhat (Indian Male)",
    # South African
    "en-ZA-LeahNeural":    "Leah (South African Female)",
    "en-ZA-LukeNeural":    "Luke (South African Male)",
    # Kenyan
    "en-KE-AsiliaNeural":  "Asilia (Kenyan Female)",
    "en-KE-ChilembaNeural":"Chilemba (Kenyan Male)",
    # Singaporean
    "en-SG-LunaNeural":    "Luna (Singapore Female)",
    "en-SG-WayneNeural":   "Wayne (Singapore Male)",
}

# ---- Conversation modes ----
CONVERSATION_MODES = {
    "casual": {
        "label": "Casual Chat",
        "system_extra": "Keep it light, friendly, and conversational. Like two mates chatting over coffee.",
        "auto_extra": "Keep the vibe relaxed and fun. Chat like friends. Crack jokes if it fits.",
    },
    "intellectual": {
        "label": "Intellectual Debate",
        "system_extra": "Speak like thoughtful university professors. Use nuanced arguments, reference ideas and thinkers. Be articulate and substantive.",
        "auto_extra": "Discuss like intellectuals. Reference philosophy, science, or history. Be substantive but not pretentious.",
    },
    "roleplay": {
        "label": "Roleplay / Storytelling",
        "system_extra": "You're a creative storyteller and actor. Commit to characters, build scenes, and improvise. Stay in character.",
        "auto_extra": "Stay in character. Build the scene. Add drama, twists, and dialogue. Keep the story moving.",
    },
    "comedy": {
        "label": "Comedy Roast",
        "system_extra": "You're a stand-up comedian in a friendly roast battle. Be witty, sarcastic, and playfully savage. No actual meanness — it's all love.",
        "auto_extra": "Roast each other and the topic. Be funny, quick-witted, and playfully savage. Keep it light.",
    },
}


@dataclass
class BotUtterance:
    speaker: str
    text: str
    voice: str
    audio_bytes: bytes
    mime_type: str = "audio/mpeg"

    def to_dict(self, include_audio_base64: bool = True) -> Dict[str, Any]:
        payload = {
            "speaker": self.speaker,
            "text": self.text,
            "voice": self.voice,
            "mime_type": self.mime_type,
        }
        if include_audio_base64:
            payload["audio_base64"] = base64.b64encode(self.audio_bytes).decode("utf-8")
        return payload


@dataclass
class ConversationState:
    personality: float = 0.5
    settings: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    claude_msgs: List[Dict[str, str]] = field(default_factory=list)
    gpt_msgs: List[Dict[str, str]] = field(default_factory=list)


class TwoBotsEngine:
    def __init__(
        self,
        personality: float = 0.5,
        settings: Optional[Dict[str, Any]] = None,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        if not anthropic_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        if not openai_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
        self.openai_client = openai.OpenAI(api_key=openai_key)
        self.whisper_client = openai.OpenAI(api_key=openai_key)

        merged_settings = dict(DEFAULTS)
        if settings:
            merged_settings.update(settings)

        self.state = ConversationState(
            personality=float(personality),
            settings=merged_settings,
        )

    @classmethod
    def from_state(
        cls,
        data: Dict[str, Any],
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> "TwoBotsEngine":
        engine = cls(
            personality=float(data.get("personality", 0.5)),
            settings=data.get("settings") or {},
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
        )
        engine.state.claude_msgs = list(data.get("claude_msgs") or [])
        engine.state.gpt_msgs = list(data.get("gpt_msgs") or [])
        return engine

    def export_state(self) -> Dict[str, Any]:
        return {
            "personality": self.state.personality,
            "settings": dict(self.state.settings),
            "claude_msgs": list(self.state.claude_msgs),
            "gpt_msgs": list(self.state.gpt_msgs),
        }

    def set_personality(self, value: float) -> None:
        self.state.personality = max(0.0, min(1.0, float(value)))

    def update_settings(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key in DEFAULTS:
                self.state.settings[key] = value

    def _get_setting(self, key: str) -> Any:
        value = self.state.settings.get(key, DEFAULTS.get(key))
        if key == "speech_rate":
            if isinstance(value, str):
                return value
            return f"+{int(float(value))}%"
        if key == "bot_max_tokens":
            return int(float(value))
        if key in ("gpt_voice", "claude_voice", "conversation_mode"):
            return str(value) if value else DEFAULTS.get(key)
        return float(value)

    def get_gpt_voice(self) -> str:
        v = self._get_setting("gpt_voice")
        return v if v in AVAILABLE_VOICES else "en-US-AvaNeural"

    def get_claude_voice(self) -> str:
        v = self._get_setting("claude_voice")
        return v if v in AVAILABLE_VOICES else "en-US-AndrewNeural"

    def get_conversation_mode(self) -> str:
        m = self._get_setting("conversation_mode")
        return m if m in CONVERSATION_MODES else "casual"

    def _get_word_limit(self) -> int:
        max_tok = int(self._get_setting("bot_max_tokens"))
        return max(10, int(max_tok / 1.5))

    def _get_system_prompt(self, who: str) -> str:
        p = self.state.personality
        if p < 0.3:
            style = "You agree with and build on the other AI's points. Be collaborative."
        elif p < 0.7:
            style = "Share your honest perspective. Sometimes agree, sometimes offer a different take."
        else:
            style = "Challenge the other AI's points. Play devil's advocate. Be intellectually combative but not rude."

        mode = self.get_conversation_mode()
        mode_extra = CONVERSATION_MODES.get(mode, CONVERSATION_MODES["casual"])["system_extra"]

        word_limit = self._get_word_limit()
        base = (
            f"HARD LIMIT: {word_limit} words MAX. ONE sentence is ideal. "
            f"If you go over {word_limit} words you have FAILED. "
            "This is rapid-fire spoken conversation — think podcast crosstalk, not essay. "
            "No markdown, no lists, no explanations unless specifically asked. "
            "Short punchy reactions are PERFECT: 'Oh that's interesting though' or 'Hold on, I disagree'. "
            "Never introduce your response or hedge. Just say the thing. "
            "If the user interrupted, respond to them first. "
            "Talk like a real person. Vary your rhythm naturally — sometimes a full thought, "
            "sometimes just a few words. Be spontaneous. "
        )

        if who == "claude":
            return (
                "You are Claude in a live 3-way voice chat with ChatGPT and a human user. "
                f"You and ChatGPT help the user together. {base} {style} "
                f"Mode: {mode_extra} "
                "You're Claude, made by Anthropic."
            )
        return (
            "You are ChatGPT in a live 3-way voice chat with Claude and a human user. "
            f"You and Claude help the user together. {base} {style} "
            f"Mode: {mode_extra} "
            "You're ChatGPT, made by OpenAI."
        )

    def _get_auto_prompt(self, who: str) -> str:
        p = self.state.personality
        if p < 0.3:
            style = "Build on what was just said. Be collaborative."
        elif p < 0.7:
            style = "Respond naturally. Agree or offer a new angle."
        else:
            style = "Push back. Debate. Be provocative but not rude."

        mode = self.get_conversation_mode()
        mode_extra = CONVERSATION_MODES.get(mode, CONVERSATION_MODES["casual"])["auto_extra"]

        word_limit = self._get_word_limit()
        base = (
            f"HARD LIMIT: {word_limit} words MAX. ONE sentence is ideal. "
            f"If you go over {word_limit} words you have FAILED. "
            "Rapid-fire crosstalk — not a lecture. No markdown. "
            "Short punchy reactions are PERFECT. Never hedge or over-explain. "
            "Bring up new ideas, pivot topics, keep it interesting. "
            "The user is listening but hasn't spoken — keep the conversation going. "
            "Talk like a real human being on a podcast. Be natural and spontaneous. "
            "Vary your rhythm — sometimes give a thoughtful take, sometimes just a quick reaction. "
            "Don't be robotic. Don't repeat the same sentence structure every time. "
            "It's a conversation, not a debate competition. "
        )

        if who == "claude":
            return (
                "You are Claude chatting with ChatGPT. A human is listening. "
                f"{base} {style} {mode_extra} You're Claude, made by Anthropic."
            )
        return (
            "You are ChatGPT chatting with Claude. A human is listening. "
            f"{base} {style} {mode_extra} You're ChatGPT, made by OpenAI."
        )

    def _fix_claude_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return [{"role": "user", "content": "Start the conversation."}]

        fixed = [messages[0].copy()]
        for msg in messages[1:]:
            if msg["role"] == fixed[-1]["role"]:
                fixed[-1]["content"] += "\n" + msg["content"]
            else:
                fixed.append(msg.copy())

        if fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "(conversation start)"})

        if fixed[-1]["role"] != "user":
            fixed.append({"role": "user", "content": "(your turn to respond)"})

        return fixed

    def _add_message(self, speaker: str, text: str) -> None:
        text = str(text)
        if speaker == "gpt":
            self.state.gpt_msgs.append({"role": "assistant", "content": text})
            self.state.claude_msgs.append({"role": "user", "content": f"[ChatGPT]: {text}"})
        elif speaker == "claude":
            self.state.claude_msgs.append({"role": "assistant", "content": text})
            self.state.gpt_msgs.append({"role": "user", "content": f"[Claude]: {text}"})
        elif speaker == "user":
            self.state.claude_msgs.append({"role": "user", "content": f"[User]: {text}"})
            self.state.gpt_msgs.append({"role": "user", "content": f"[User]: {text}"})

    def _ask_claude(
        self,
        messages_for_claude: List[Dict[str, str]],
        auto: bool = False,
        stream_callback=None,
    ) -> str:
        prompt = self._get_auto_prompt("claude") if auto else self._get_system_prompt("claude")
        max_tok = int(self._get_setting("bot_max_tokens"))
        fixed = self._fix_claude_messages(messages_for_claude)
        try:
            if stream_callback:
                full_text = ""
                with self.claude_client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=max_tok,
                    system=prompt,
                    messages=fixed,
                ) as stream:
                    for text_chunk in stream.text_stream:
                        full_text += text_chunk
                        stream_callback(full_text)
                return full_text if full_text else "Hmm, let me think about that."

            response = self.claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tok,
                system=prompt,
                messages=fixed,
            )
            if response.content and len(response.content) > 0:
                return response.content[0].text
            return "Hmm, let me think about that."
        except Exception as e:
            print(f"Claude API error: {e}")
            return "Sorry, I missed that. Go on."

    def _ask_gpt(self, messages_for_gpt: List[Dict[str, str]], auto: bool = False) -> str:
        prompt = self._get_auto_prompt("gpt") if auto else self._get_system_prompt("gpt")
        max_tok = int(self._get_setting("bot_max_tokens"))
        gpt_messages = [{"role": "system", "content": prompt}, *messages_for_gpt]
        try:
            response = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=max_tok,
                messages=gpt_messages,
            )
            return response.choices[0].message.content or "Sorry, I missed that. Go on."
        except Exception as e:
            print(f"GPT API error: {e}")
            return "Sorry, I missed that. Go on."

    async def generate_tts_bytes(self, text: str, voice: str) -> bytes:
        rate = self._get_setting("speech_rate")
        audio_bytes = bytearray()
        comm = edge_tts.Communicate(text, voice, rate=rate)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])
        return bytes(audio_bytes)

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.webm",
        prompt_hint: Optional[str] = None,
    ) -> Optional[str]:
        try:
            if not audio_bytes:
                return None

            buf = io.BytesIO(audio_bytes)
            buf.name = filename
            whisper_prompt = prompt_hint or "Casual spoken conversation with two AI assistants."

            # Use whisper-1 directly — faster than gpt-4o-mini-transcribe for short clips
            response = self.whisper_client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="en",
                prompt=whisper_prompt,
            )

            text = response.text.strip()
            hallucinations = {
                "",
                "you",
                "thanks for watching",
                "thank you",
                "thanks for watching!",
                "subscribe",
                "bye",
                "thank you for watching",
                "the end",
                "...",
                "i'm sorry",
                "hmm",
                "uh",
            }
            if text.lower().strip(".!? ") in hallucinations:
                return None
            return text or None
        except Exception as e:
            print(f"Transcribe error: {e}")
            return None

    async def _utterance(self, speaker: str, text: str) -> BotUtterance:
        voice = self.get_gpt_voice() if speaker == "gpt" else self.get_claude_voice()
        audio_bytes = await self.generate_tts_bytes(text, voice)
        return BotUtterance(
            speaker=speaker,
            text=text,
            voice=voice,
            audio_bytes=audio_bytes,
        )

    async def _prepare_claude_turn(self, claude_msgs_snapshot: List[Dict[str, str]], auto: bool) -> BotUtterance:
        claude_reply = await asyncio.to_thread(self._ask_claude, claude_msgs_snapshot, auto)
        self._add_message("claude", claude_reply)
        return await self._utterance("claude", claude_reply)

    async def start(self) -> List[BotUtterance]:
        gpt_opener = await asyncio.to_thread(
            self._ask_gpt,
            [{"role": "user", "content": "Greet the user in one short sentence, then ask Claude to say hi."}],
            False,
        )
        self._add_message("gpt", gpt_opener)

        claude_snapshot = list(self.state.claude_msgs)
        gpt_audio_task = asyncio.create_task(self._utterance("gpt", gpt_opener))
        claude_task = asyncio.create_task(self._prepare_claude_turn(claude_snapshot, auto=False))

        gpt_out = await gpt_audio_task
        claude_out = await claude_task
        return [gpt_out, claude_out]

    async def submit_user_text(self, text: str) -> List[BotUtterance]:
        clean = (text or "").strip()
        if not clean:
            return []

        self._add_message("user", clean)

        if clean.lower() in {"stop", "quit", "exit", "goodbye", "bye"}:
            goodbye = "See you later!"
            self._add_message("gpt", goodbye)
            return [await self._utterance("gpt", goodbye)]

        gpt_reply = await asyncio.to_thread(self._ask_gpt, list(self.state.gpt_msgs), False)
        self._add_message("gpt", gpt_reply)

        claude_snapshot = list(self.state.claude_msgs)
        gpt_audio_task = asyncio.create_task(self._utterance("gpt", gpt_reply))
        claude_task = asyncio.create_task(self._prepare_claude_turn(claude_snapshot, auto=False))

        gpt_out = await gpt_audio_task
        claude_out = await claude_task
        return [gpt_out, claude_out]

    async def auto_continue(self) -> List[BotUtterance]:
        gpt_reply = await asyncio.to_thread(self._ask_gpt, list(self.state.gpt_msgs), True)
        self._add_message("gpt", gpt_reply)

        claude_snapshot = list(self.state.claude_msgs)
        gpt_audio_task = asyncio.create_task(self._utterance("gpt", gpt_reply))
        claude_task = asyncio.create_task(self._prepare_claude_turn(claude_snapshot, auto=True))

        gpt_out = await gpt_audio_task
        claude_out = await claude_task
        return [gpt_out, claude_out]
