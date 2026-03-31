"""
Engine 2.0 — Clean text + TTS. No locks, no queue management.
ChatGPT on the left, Claude on the right.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
import openai
# Pre-import OpenAI submodules to prevent deadlock when parallel threads
# trigger lazy imports simultaneously (Python import lock issue)
import openai.resources.audio
import openai.resources.chat

import tuning
from prompts import (
    TTS_MODEL, TTS_BASE_INSTRUCTION,
    MODES, INTERACTION_STYLES, PINGPONG_MODES, FORMAT_ROLES,
    PERSONALITIES, CHARACTER_QUIRKS,
    AGREEABLENESS_BATCH, AGREEABLENESS_LEGACY,
    SCRIPTED_BATCH_SYSTEM, SCRIPTED_BATCH_PROMPT,
    SCRIPTED_CONTEXT_FIRST_BATCH,
    SCRIPTED_CONTEXT_CONTINUATION, SCRIPTED_RANDOM_TOPIC_IMMERSIVE,
    PACING_NUDGES,
    SCRIPTED_OPENER_SYSTEM, SCRIPTED_OPENER_PROMPT, SCRIPTED_OPENER_DESCRIPTIONS,
    BRIDGE_SYSTEM, BRIDGE_PROMPT,
    SINGLE_BOT_SYSTEM, SINGLE_BOT_PROMPT,
    PINGPONG_CONVERSATION_PROMPT, PINGPONG_CONVERSATION_SYSTEM,
    PINGPONG_OPENER_CONVERSATION,
    PINGPONG_OPENER_DEBATE, PINGPONG_OPENER_ADVICE,
    PINGPONG_OPENER_HELP_ME_DECIDE, PINGPONG_OPENER_RESEARCH,
    PINGPONG_ONGOING_DEBATE, PINGPONG_ONGOING_ADVICE,
    PINGPONG_ONGOING_HELP_ME_DECIDE, PINGPONG_ONGOING_RESEARCH,
    PINGPONG_SYSTEM,
    REVIEW_INSTRUCTION, REVIEW_SYSTEM, REVIEW_PROMPT, REVIEW_MODE_VERB,
    RESPOND_PROMPT, RESPOND_SYSTEM,
    LEGACY_ROLE_GPT, LEGACY_ROLE_CLAUDE, LEGACY_BASE_RULES,
    LEGACY_TURN_GOAL_OPENER_GPT, LEGACY_TURN_GOAL_OPENER_CLAUDE,
    LEGACY_TURN_GOAL_AUTO, LEGACY_TURN_GOAL_USER_SPOKE,
    LEGACY_QUIRK_REMINDER, LEGACY_CUSTOM_STRENGTH,
    CONCLUSIONS_HEADER, MILESTONE_WORD,
    WORD_LIMIT_TIERS, WORD_LIMIT_DEFAULT,
)


# ---- Filler & hesitation helpers ----

def pick_filler(agreeableness: float, active_quirks: list = None) -> str:
    """Pick a random filler weighted by agreeableness + active quirks."""
    # Start with global pool based on agreeableness
    if agreeableness < 0.3:
        pool_names = tuning.FILLER_WEIGHTS["agreeable"]
    elif agreeableness > 0.7:
        pool_names = tuning.FILLER_WEIGHTS["disagreeable"]
    else:
        pool_names = tuning.FILLER_WEIGHTS["balanced"]
    pool = []
    for name in pool_names:
        pool.extend(tuning.FILLERS.get(name, []))

    # Mix in trait-specific fillers if quirks are active (50/50 chance of trait vs global)
    if active_quirks:
        trait_pool = []
        for quirk in active_quirks:
            trait_pool.extend(tuning.TRAIT_FILLERS.get(quirk, []))
        if trait_pool and random.random() < 0.5:
            return random.choice(trait_pool)

    return random.choice(pool) if pool else "Hmm."


def apply_word_limit_variance(base_words: int) -> int:
    """Randomize a word limit using WORD_LIMIT_VARIANCE from tuning."""
    roll = random.random()
    cumulative = 0.0
    offset = 0
    for prob, off in tuning.WORD_LIMIT_VARIANCE:
        cumulative += prob
        if roll <= cumulative:
            offset = off
            break
    return max(tuning.WORD_LIMIT_FLOOR, base_words + offset)


def inject_hesitation(text: str) -> str:
    """Maybe inject a human hesitation (um, uh, well...) into the response text."""
    if random.random() > tuning.HESITATION_CHANCE:
        return text  # no hesitation this time

    # Decide position: start or mid
    roll = random.random()
    cumulative = 0.0
    position = "start"
    for prob, pos in tuning.HESITATION_POSITION_WEIGHTS:
        cumulative += prob
        if roll <= cumulative:
            position = pos
            break

    if position == "start":
        hesitation = random.choice(tuning.HESITATIONS_START)
        # Don't double up if response already starts with a hesitation word
        first_word = text.split()[0].rstrip(",.!?") if text else ""
        if first_word.lower() in ("um", "uh", "well", "like", "so", "hmm", "right", "look", "see", "ok"):
            return text
        return f"{hesitation} {text[0].lower()}{text[1:]}" if text else text
    else:
        # Insert after first sentence (look for . ! or ?)
        for i, ch in enumerate(text):
            if ch in ".!?" and i < len(text) - 2:
                mid_hesitation = random.choice(tuning.HESITATIONS_MID)
                rest = text[i+1:].lstrip()
                return text[:i+1] + mid_hesitation + " " + rest
        # No sentence break found, put at start instead
        hesitation = random.choice(tuning.HESITATIONS_START)
        return f"{hesitation} {text[0].lower()}{text[1:]}" if text else text


CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-4o-mini"

AVAILABLE_VOICES = {
    "alloy":   "Alloy (Neutral)",
    "ash":     "Ash (Warm Male)",
    "ballad":  "Ballad (Soft)",
    "coral":   "Coral (Warm Female)",
    "echo":    "Echo (Smooth Male)",
    "fable":   "Fable (British)",
    "onyx":    "Onyx (Deep Male)",
    "nova":    "Nova (Friendly Female)",
    "sage":    "Sage (Calm)",
    "shimmer": "Shimmer (Bright Female)",
}

RESPONSE_LENGTHS = {
    "avg_10": {
        "max_tokens": tuning.MAX_TOKENS["avg_10"],
        "label": "~10",
        "base_words": tuning.WORD_LIMITS["avg_10"],
    },
    "avg_20": {
        "max_tokens": tuning.MAX_TOKENS["avg_20"],
        "label": "~20",
        "base_words": tuning.WORD_LIMITS["avg_20"],
    },
    "avg_30": {
        "max_tokens": tuning.MAX_TOKENS["avg_30"],
        "label": "~30",
        "base_words": tuning.WORD_LIMITS["avg_30"],
    },
    "avg_40": {
        "max_tokens": tuning.MAX_TOKENS["avg_40"],
        "label": "~40",
        "base_words": tuning.WORD_LIMITS["avg_40"],
    },
    "avg_50": {
        "max_tokens": tuning.MAX_TOKENS["avg_50"],
        "label": "~50",
        "base_words": tuning.WORD_LIMITS["avg_50"],
    },
}

DEFAULTS = {
    "gpt_max_tokens": 150,
    "claude_max_tokens": 150,
    "gpt_response_length": "avg_20",
    "claude_response_length": "avg_20",
    "gpt_voice": "shimmer",
    "claude_voice": "onyx",
    "mode": "conversation",
    "gpt_personality": "default",
    "claude_personality": "default",
    "gpt_quirks": [],
    "claude_quirks": [],
    "gpt_custom": "",
    "claude_custom": "",
    "gpt_personality_strength": 1,
    "claude_personality_strength": 1,
    "gpt_quirk_strength": 1,
    "claude_quirk_strength": 1,
    "gpt_tts_speed": 1.0,
    "claude_tts_speed": 1.0,
    "topic": "random",
}

# MODES, INTERACTION_STYLES, PINGPONG_MODES, FORMAT_ROLES, PERSONALITIES,
# and CHARACTER_QUIRKS are now imported from prompts.py

@dataclass
class ConversationState:
    personality: float = 0.5
    settings: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    claude_msgs: List[Dict[str, str]] = field(default_factory=list)
    gpt_msgs: List[Dict[str, str]] = field(default_factory=list)
    rounds_since_filler: int = 0
    next_filler_at: int = 0
    autopilot_batch_count: int = 0
    prev_format: Optional[str] = None
    prev_topic: Optional[str] = None
    last_mix_pick: Optional[str] = None
    # ---- PING-PONG MODE (research, debate, advice) ----
    pingpong_msg_count: int = 0
    pingpong_reviews: List[str] = field(default_factory=list)
    pingpong_conclusions: List[str] = field(default_factory=list)
    pingpong_complete: bool = False
    debate_score_gpt: int = 0
    debate_score_claude: int = 0
    milestone_target: int = 3  # how many findings/motions/recommendations (1-5), set by opener AI
    exchanges_per_milestone: int = 9  # exchanges between each milestone (8-12), set by opener AI


class TwoBotsEngine:
    def __init__(
        self,
        personality: float = 0.5,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not anthropic_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        if not openai_key:
            raise RuntimeError("Missing OPENAI_API_KEY")

        self.claude_client = anthropic.Anthropic(api_key=anthropic_key)
        self.openai_client = openai.OpenAI(api_key=openai_key)

        merged = dict(DEFAULTS)
        if settings:
            merged.update(settings)
        self.state = ConversationState(personality=float(personality), settings=merged)

    @classmethod
    def from_state(cls, data: Dict[str, Any]) -> "TwoBotsEngine":
        engine = cls(
            personality=float(data.get("personality", 0.5)),
            settings=data.get("settings") or {},
        )
        engine.state.claude_msgs = list(data.get("claude_msgs") or [])
        engine.state.gpt_msgs = list(data.get("gpt_msgs") or [])
        engine.state.rounds_since_filler = int(data.get("rounds_since_filler", 0))
        engine.state.next_filler_at = int(data.get("next_filler_at", 0))
        engine.state.autopilot_batch_count = int(data.get("autopilot_batch_count", 0))
        engine.state.prev_format = data.get("prev_format")
        engine.state.prev_topic = data.get("prev_topic")
        # ---- PING-PONG MODE ----
        engine.state.pingpong_msg_count = int(data.get("pingpong_msg_count", data.get("research_msg_count", 0)))
        engine.state.pingpong_conclusions = list(data.get("pingpong_conclusions", data.get("research_conclusions", [])) or [])
        engine.state.pingpong_complete = bool(data.get("pingpong_complete", data.get("research_complete", False)))
        engine.state.pingpong_reviews = list(data.get("pingpong_reviews", data.get("research_reviews", [])) or [])
        engine.state.debate_score_gpt = int(data.get("debate_score_gpt", 0))
        engine.state.debate_score_claude = int(data.get("debate_score_claude", 0))
        engine.state.milestone_target = int(data.get("milestone_target", 3))
        engine.state.exchanges_per_milestone = int(data.get("exchanges_per_milestone", 9))
        return engine

    def export_state(self) -> Dict[str, Any]:
        return {
            "personality": self.state.personality,
            "settings": dict(self.state.settings),
            "claude_msgs": list(self.state.claude_msgs),
            "gpt_msgs": list(self.state.gpt_msgs),
            "rounds_since_filler": self.state.rounds_since_filler,
            "next_filler_at": self.state.next_filler_at,
            "autopilot_batch_count": self.state.autopilot_batch_count,
            "prev_format": self.state.prev_format,
            "prev_topic": self.state.prev_topic,
            # ---- PING-PONG MODE ----
            "pingpong_msg_count": self.state.pingpong_msg_count,
            "pingpong_conclusions": list(self.state.pingpong_conclusions),
            "pingpong_complete": self.state.pingpong_complete,
            "pingpong_reviews": list(self.state.pingpong_reviews),
            "debate_score_gpt": self.state.debate_score_gpt,
            "debate_score_claude": self.state.debate_score_claude,
            "milestone_target": self.state.milestone_target,
            "exchanges_per_milestone": self.state.exchanges_per_milestone,
        }

    def should_filler(self) -> bool:
        """Check if this round should be a filler instead of a real API call."""
        if self.state.next_filler_at == 0:
            self.state.next_filler_at = random.randint(tuning.FILLER_MIN_INTERVAL, tuning.FILLER_MAX_INTERVAL)
        return self.state.rounds_since_filler >= self.state.next_filler_at

    def get_filler(self, who: str = "gpt") -> str:
        """Get a filler response and reset the counter. Uses active quirks for trait fillers."""
        prefix = "claude" if who == "claude" else "gpt"
        active_quirks = self._s(f"{prefix}_quirks")
        filler = pick_filler(self.state.personality, active_quirks)
        self.state.rounds_since_filler = 0
        self.state.next_filler_at = random.randint(tuning.FILLER_MIN_INTERVAL, tuning.FILLER_MAX_INTERVAL)
        return filler

    def tick_filler(self) -> None:
        """Increment the filler counter (call after each real round)."""
        self.state.rounds_since_filler += 1

    def update_settings(self, updates: Dict[str, Any]) -> None:
        for k, v in updates.items():
            self.state.settings[k] = v

    def _s(self, key: str) -> Any:
        """Get a setting with fallback to DEFAULTS."""
        val = self.state.settings.get(key, DEFAULTS.get(key))
        if key in ("gpt_max_tokens", "claude_max_tokens"):
            return int(float(val))
        if key in ("gpt_personality_strength", "claude_personality_strength",
                    "gpt_quirk_strength", "claude_quirk_strength"):
            try:
                return max(0, min(3, int(float(val))))
            except (TypeError, ValueError):
                return 1
        if key in ("gpt_quirks", "claude_quirks"):
            return val if isinstance(val, list) else []
        return val

    def get_gpt_voice(self) -> str:
        v = self._s("gpt_voice")
        return v if v in AVAILABLE_VOICES else "nova"

    def get_claude_voice(self) -> str:
        v = self._s("claude_voice")
        return v if v in AVAILABLE_VOICES else "onyx"

    def get_tts_speed(self, who: str = "gpt") -> float:
        val = self._s(f"{who}_tts_speed")
        if val is None:
            val = self._s("tts_speed")  # fallback to old global setting
        try:
            return max(0.5, min(2.0, float(val)))
        except (TypeError, ValueError):
            return 1.0

    def get_character_description(self, who: str) -> str:
        """Gather all character parts (personality, quirks, custom, agreeableness) into one string."""
        prefix = who
        parts = []
        strength_idx = int(self._s(f"{prefix}_personality_strength") or 1)

        p_key = self._s(f"{prefix}_personality") or "default"
        if p_key != "default":
            p_data = PERSONALITIES.get(p_key, PERSONALITIES["default"])
            p_text = p_data.get(strength_idx, "") if isinstance(p_data, dict) else ""
            if p_text:
                parts.append(p_text)

        quirks = self._s(f"{prefix}_quirks") or []
        for q in quirks:
            if q in CHARACTER_QUIRKS:
                qd = CHARACTER_QUIRKS[q]
                q_text = qd.get(strength_idx, "") if isinstance(qd, dict) else str(qd)
                if q_text:
                    parts.append(q_text)

        custom = self._s(f"{prefix}_custom") or ""
        if custom.strip():
            parts.append(custom.strip())

        a = self.state.personality
        if a < 0.2:
            parts.append("Warm and supportive")
        elif a < 0.4:
            parts.append("Generally friendly")
        elif a >= 0.8:
            parts.append("Combative, challenging")
        elif a >= 0.6:
            parts.append("Slightly confrontational")

        return ". ".join(parts) if parts else ""

    def generate_tts_character_description(self, character_desc: str) -> str:
        """Ask Claude Haiku to generate a TTS voice description for a character."""
        prompt = f"""Describe how a text-to-speech voice should sound for this character:
"{character_desc}"

Fill out ONLY these fields, keep each to one sentence:
Identity:
Affect:
Tone:
Emotion:
Pauses:
Pronunciation:

Return ONLY the six fields, nothing else."""

        resp = self.claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        result = resp.content[0].text if resp.content else ""
        print(f"\n🎭 TTS CHARACTER DESCRIPTION:\nInput: {character_desc}\nOutput: {result}\n")
        return result.strip()

    def _build_tts_instruction(self, who: str, cached_tts_char: str = "") -> str:
        """Build a voice instruction. Uses Claude-generated description if cached, else falls back."""
        # If we have a Claude-generated TTS character description, use it
        if cached_tts_char:
            return f"{cached_tts_char}\n{TTS_BASE_INSTRUCTION}"

        # Fallback: build from raw character parts
        char_desc = self.get_character_description(who)
        if char_desc:
            return f"{TTS_BASE_INSTRUCTION} Your character: {char_desc}."
        return TTS_BASE_INSTRUCTION

    async def generate_tts_bytes(self, text: str, voice: str, who: str = "gpt") -> bytes:
        speed = self.get_tts_speed(who)
        # _tts_char_cache is a live reference to TTS_CHARACTER_CACHE[sid] dict, set by app.py
        raw_cache = getattr(self, '_tts_char_cache', {})
        cached_tts_char = raw_cache.get(who, "")
        instruction = self._build_tts_instruction(who, cached_tts_char)
        if cached_tts_char:
            print(f"🔊 TTS for {who}: ✅ CHARACTER VOICE ACTIVE ({len(cached_tts_char)} chars) | Full instruction: {len(instruction)} chars")
            print(f"   Character: {cached_tts_char[:200]}")
        else:
            print(f"🔊 TTS for {who}: ❌ BASE VOICE ONLY (no character cached)")
        tts_input = text
        char_desc = self.get_character_description(who)
        if char_desc:
            tts_input = f"[Voice: {char_desc}] {text}"
            print(f"   🎬 Stage direction in text: [Voice: {char_desc}]")
        def _call():
            import openai as _oai
            print(f"   📦 OpenAI SDK version: {_oai.__version__}")
            kwargs = dict(
                model=TTS_MODEL, voice=voice, input=tts_input,
                instructions=instruction,
                response_format="mp3", speed=speed,
            )
            print(f"   📨 TTS API kwargs: model={kwargs['model']}, voice={kwargs['voice']}, speed={kwargs['speed']}")
            print(f"   📨 input ({len(kwargs['input'])} chars): {kwargs['input'][:150]}...")
            print(f"   📨 instructions ({len(kwargs['instructions'])} chars): {kwargs['instructions'][:150]}...")
            try:
                resp = self.openai_client.audio.speech.create(**kwargs)
                print(f"   ✅ TTS API call succeeded, got {len(resp.content)} bytes")
                return resp.content
            except TypeError as e:
                print(f"   ⚠️ TypeError with instructions param: {e}")
                print(f"   ⚠️ FALLING BACK without instructions param")
                del kwargs["instructions"]
                resp = self.openai_client.audio.speech.create(**kwargs)
                return resp.content
        return await asyncio.to_thread(_call)

    # ---- Prompt building ----
    def _build_system_prompt(self, who: str, auto: bool, opener: bool = False) -> str:
        prefix = "claude" if who == "claude" else "gpt"
        other = "ChatGPT" if who == "claude" else "Claude"

        # ---- STATIC SECTION (cacheable — stays the same across turns) ----

        # [ROLE]
        role = LEGACY_ROLE_CLAUDE if who == "claude" else LEGACY_ROLE_GPT

        # [VOICE] — from mode selection
        mode_key = self._s("mode") or self._s("interaction_style") or "conversation"
        mode_data = MODES.get(mode_key, MODES["conversation"])
        voice = f"[VOICE] {mode_data['prompt']}"

        # [BASE RULES] — always-on anti-boring rules
        base_rules = LEGACY_BASE_RULES.format(other=other)

        # [PERSONALITY] — character, quirks, custom
        personality_parts = []

        # Agreeableness (from landing page slider)
        agreeableness = self.state.personality
        if agreeableness < 0.2:
            personality_parts.append(AGREEABLENESS_LEGACY["very_agreeable"])
        elif agreeableness < 0.4:
            personality_parts.append(AGREEABLENESS_LEGACY["agreeable"])
        elif agreeableness >= 0.8:
            personality_parts.append(AGREEABLENESS_LEGACY["very_disagreeable"])
        elif agreeableness >= 0.6:
            personality_parts.append(AGREEABLENESS_LEGACY["disagreeable"])
        # 0.4-0.6 = balanced, say nothing

        # Named personality
        p_key = self._s(f"{prefix}_personality") or "default"
        p_strength = self._s(f"{prefix}_personality_strength")
        p_data = PERSONALITIES.get(p_key, PERSONALITIES["default"])
        p_text = p_data.get(p_strength, "") if isinstance(p_data, dict) else ""
        if p_text:
            personality_parts.append(p_text)

        # Custom personality (uses same strength slider)
        custom = self._s(f"{prefix}_custom") or ""
        if custom:
            template = LEGACY_CUSTOM_STRENGTH.get(p_strength, "{custom}")
            c_text = template.format(custom=custom) if template else ""
            if c_text:
                personality_parts.append(c_text)

        # Quirks
        q_keys = self._s(f"{prefix}_quirks")
        q_strength = self._s(f"{prefix}_quirk_strength")
        quirk_parts = []
        for q in q_keys:
            if q in CHARACTER_QUIRKS:
                qd = CHARACTER_QUIRKS[q]
                quirk_parts.append(qd.get(q_strength, "") if isinstance(qd, dict) else str(qd))
        if quirk_parts:
            personality_parts.append(" ".join(quirk_parts))
            personality_parts.append(LEGACY_QUIRK_REMINDER)

        personality_section = ""
        if personality_parts:
            personality_section = "[PERSONALITY]\n" + "\n".join(f"- {p}" for p in personality_parts)

        # ---- DYNAMIC SECTION (changes every turn) ----

        # [TURN GOAL] — context for this specific turn
        if opener:
            turn_goal = LEGACY_TURN_GOAL_OPENER_GPT if who == "gpt" else LEGACY_TURN_GOAL_OPENER_CLAUDE
        elif auto:
            turn_goal = LEGACY_TURN_GOAL_AUTO
        else:
            turn_goal = LEGACY_TURN_GOAL_USER_SPOKE

        # [OUTPUT INSTRUCTIONS] — response length (with randomized word limit)
        length_key = self._s(f"{prefix}_response_length") or "avg_20"
        preset = RESPONSE_LENGTHS.get(length_key, RESPONSE_LENGTHS["avg_20"])
        base_words = preset["base_words"]
        randomized_words = apply_word_limit_variance(base_words)
        length_prompt = f"Randomize your response length between 1 and {randomized_words} words. Always finish your sentences."
        output = f"[OUTPUT INSTRUCTIONS] {length_prompt}"

        # ---- ASSEMBLE ----
        sections = [role, voice, base_rules]
        if personality_section:
            sections.append(personality_section)
        sections.append(turn_goal)
        sections.append(output)

        return "\n\n".join(sections)

    # ---- Message management ----
    def add_message(self, speaker: str, text: str) -> None:
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

    def _fix_claude_messages(self, msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure Claude messages alternate user/assistant properly."""
        if not msgs:
            return [{"role": "user", "content": "Start the conversation."}]
        fixed = [msgs[0].copy()]
        for m in msgs[1:]:
            if m["role"] == fixed[-1]["role"]:
                fixed[-1]["content"] += "\n" + m["content"]
            else:
                fixed.append(m.copy())
        if fixed[0]["role"] != "user":
            fixed.insert(0, {"role": "user", "content": "(conversation start)"})
        if fixed[-1]["role"] != "user":
            fixed.append({"role": "user", "content": "(your turn to respond)"})
        return fixed

    # ---- Autopilot batch generation ----

    def generate_autopilot_batch(self, who_generates: str = "gpt") -> list:
        """Generate a batch of 10-14 messages in one API call.

        Returns a list of dicts: [{"speaker": "gpt"|"claude", "text": "..."}, ...]
        Uses whichever API is specified by who_generates ("gpt" or "claude").
        """
        num_messages = random.randint(tuning.AUTOPILOT_BATCH_MIN, tuning.AUTOPILOT_BATCH_MAX)

        # ---- Gather all settings for both bots ----
        mode_key = self._s("mode") or self._s("interaction_style") or "conversation"
        # Handle "random" or "mix" — pick a random format
        real_modes = [k for k in MODES if k not in ("random", "mix") and k not in PINGPONG_MODES]
        if mode_key in ("random", "mix"):
            mode_key = random.choice(real_modes)
            print(f"🎲 {'Mix' if self._s('mode') == 'mix' else 'Random'} → picked: {mode_key}")
            self.state.last_mix_pick = mode_key
        else:
            self.state.last_mix_pick = None
        mode_data = MODES.get(mode_key, MODES["conversation"])

        # Topic
        topic = self._s("topic") or "random"
        # Topic line — if random + immersive format, tell AI to pick a scenario
        immersive_formats = ("roleplay", "bedtime_story", "game", "movie_dialogue", "comedy")
        if topic.lower() == "random" and mode_key in immersive_formats:
            topic_line = SCRIPTED_RANDOM_TOPIC_IMMERSIVE.format(mode_label=mode_key.replace('_', ' '))
        elif topic.lower() == "random":
            topic_line = ""
        else:
            topic_line = f"\n[TOPIC] The conversation should be about: {topic}. Stay on this topic."

        # Build merged character description for each bot
        def build_character(prefix: str) -> str:
            """Build a single character line: 'Strongly sarcastic, Strongly obsessed with cats, custom trait here'"""
            parts = []
            strength_idx = self._s(f"{prefix}_personality_strength") or 1
            strength_labels = {0: "Mildly", 1: "", 2: "Strongly", 3: "Extremely"}
            strength_word = strength_labels.get(strength_idx, "")

            # Personality preset
            p_key = self._s(f"{prefix}_personality") or "default"
            if p_key != "default":
                p_data = PERSONALITIES.get(p_key, PERSONALITIES["default"])
                p_text = p_data.get(strength_idx, "") if isinstance(p_data, dict) else ""
                if p_text:
                    parts.append(p_text)

            # Quirks (use same strength)
            quirks = self._s(f"{prefix}_quirks") or []
            for q in quirks:
                if q in CHARACTER_QUIRKS:
                    qd = CHARACTER_QUIRKS[q]
                    q_text = qd.get(strength_idx, "") if isinstance(qd, dict) else str(qd)
                    if q_text:
                        parts.append(q_text)

            # Custom personality
            custom = self._s(f"{prefix}_custom") or ""
            if custom.strip():
                if strength_word:
                    parts.append(f"{strength_word} {custom.strip()}")
                else:
                    parts.append(custom.strip())

            return ", ".join(parts) if parts else ""

        gpt_character = build_character("gpt")
        claude_character = build_character("claude")

        # Word limits
        gpt_length_key = self._s("gpt_response_length") or "avg_20"
        gpt_base_words = RESPONSE_LENGTHS.get(gpt_length_key, RESPONSE_LENGTHS["avg_20"])["base_words"]
        claude_length_key = self._s("claude_response_length") or "avg_20"
        claude_base_words = RESPONSE_LENGTHS.get(claude_length_key, RESPONSE_LENGTHS["avg_20"])["base_words"]
        gpt_word_limit = apply_word_limit_variance(gpt_base_words)
        claude_word_limit = apply_word_limit_variance(claude_base_words)

        # Agreeableness — only include if not balanced
        agreeableness = self.state.personality
        agree_section = ""
        if agreeableness < 0.2:
            agree_section = "\n[AGREEABLENESS] " + AGREEABLENESS_BATCH["very_agreeable"]
        elif agreeableness < 0.4:
            agree_section = "\n[AGREEABLENESS] " + AGREEABLENESS_BATCH["agreeable"]
        elif agreeableness >= 0.8:
            agree_section = "\n[AGREEABLENESS] " + AGREEABLENESS_BATCH["very_disagreeable"]
        elif agreeableness >= 0.6:
            agree_section = "\n[AGREEABLENESS] " + AGREEABLENESS_BATCH["disagreeable"]
        # else: balanced — don't include

        # Conversation history — last N messages for context
        window = tuning.AUTOPILOT_HISTORY_WINDOW
        recent_msgs = self.state.gpt_msgs[-window:] if self.state.gpt_msgs else []
        history_lines = [f"  {m['role']}: {m['content']}" for m in recent_msgs]
        history_text = "\n".join(history_lines) if history_lines else "  (no conversation yet — this is the start)"

        # Determine who spoke last so we can enforce the first speaker
        last_speaker = None
        if self.state.gpt_msgs:
            last_msg = self.state.gpt_msgs[-1]
            if last_msg["role"] == "assistant":
                last_speaker = "gpt"  # GPT's own message
            elif "[Claude]" in last_msg.get("content", ""):
                last_speaker = "claude"
            elif "[User]" in last_msg.get("content", ""):
                last_speaker = "user"

        first_speaker = "claude" if last_speaker in ("gpt", "user") else "gpt"
        first_speaker_instruction = f'The first message MUST be from {"Claude" if first_speaker == "claude" else "ChatGPT"}.'

        # ---- CHAT MODE ---- Check if user spoke recently (within last 8 messages)
        user_request = None
        for m in reversed(self.state.gpt_msgs[-8:]):
            if "[User]" in m.get("content", ""):
                user_request = m["content"].replace("[User]: ", "")
                break

        if user_request:
            # User spoke recently — focus on their request
            print(f"\n🎬 User request detected: {user_request}\n")
            creative_direction = f"""The user said: "{user_request}". Focus on what the user asked for."""
        else:
            # No recent user input — just continue naturally
            creative_direction = ""

        # Increment batch counter
        self.state.autopilot_batch_count += 1

        # ---- First batch or format/topic change instructions ----
        context_instruction = ""
        format_label = mode_data.get("label", mode_key.replace("_", " ").title())
        topic_display = topic if topic.lower() != "random" else "a random topic"

        format_changed = (self.state.prev_format is not None and self.state.prev_format != mode_key)
        topic_changed = (self.state.prev_topic is not None and self.state.prev_topic != topic)

        if self.state.autopilot_batch_count == 1 or format_changed or topic_changed:
            # First batch or format/topic changed — fresh start
            if format_changed or topic_changed:
                self.state.gpt_msgs.clear()
                self.state.claude_msgs.clear()
                print(f"🔄 Reset — format: {self.state.prev_format} -> {mode_key}, topic: {self.state.prev_topic} -> {topic}")
            context_instruction = SCRIPTED_CONTEXT_FIRST_BATCH.format(
                format_label=format_label.lower(), topic_display=topic_display)
            print(f"🎬 First batch — announcing format: {format_label}, topic: {topic_display}")
        else:
            context_instruction = SCRIPTED_CONTEXT_CONTINUATION

        # Update tracked format/topic for next batch comparison
        self.state.prev_format = mode_key
        self.state.prev_topic = topic

        # Build character descriptions with strength words
        strength_idx_gpt = self._s("gpt_personality_strength") or 1
        strength_idx_claude = self._s("claude_personality_strength") or 1
        strength_words = {0: "slightly", 1: "", 2: "very", 3: "extremely"}
        gpt_strength_word = strength_words.get(strength_idx_gpt, "")
        claude_strength_word = strength_words.get(strength_idx_claude, "")

        gpt_default_style = "energetic, curious, enthusiastic"
        claude_default_style = "thoughtful, witty, grounded"
        gpt_traits = gpt_character if gpt_character else gpt_default_style
        claude_traits = claude_character if claude_character else claude_default_style

        # Format-dependent role
        format_role_data = FORMAT_ROLES.get(mode_key, FORMAT_ROLES["conversation"])
        role_name = format_role_data["role"]
        content_type = format_role_data["content"]

        # Build the prompt
        user_instruction = f'\nYou MUST build on what the user said: "{user_request}"' if user_request else ""
        gpt_character_line = f"{gpt_strength_word} {gpt_traits}".strip() if gpt_strength_word else gpt_traits
        claude_character_line = f"{claude_strength_word} {claude_traits}".strip() if claude_strength_word else claude_traits
        pacing_nudge = random.choice(PACING_NUDGES)
        prompt = SCRIPTED_BATCH_PROMPT.format(
            first_speaker_instruction=first_speaker_instruction,
            user_instruction=user_instruction,
            context_instruction=context_instruction,
            num_messages=num_messages,
            interaction_type=format_role_data.get("interaction", "INTERACTION").upper(),
            history_text=history_text,
            pacing_nudge=pacing_nudge,
        )

        # ---- Build system message ----
        system_msg = SCRIPTED_BATCH_SYSTEM.format(
            role_name=role_name.lower(),
            mode_prompt=mode_data['prompt'],
            topic_line=topic_line,
            agree_section=agree_section,
            gpt_character_line=gpt_character_line,
            claude_character_line=claude_character_line,
        )

        # ---- Log the FULL prompt (system + user) ----
        print("\n" + "=" * 70)
        print(f"📤 AUTOPILOT BATCH (generated by {who_generates}, {num_messages} msgs)")
        print("=" * 70)
        print(f"\n--- SYSTEM PROMPT ---\n{system_msg}")
        print(f"\n--- USER PROMPT ---\n{prompt}")
        print("=" * 70 + "\n")

        # ---- Make the API call ----
        try:
            if who_generates == "claude":
                resp = self.claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=tuning.AUTOPILOT_MAX_TOKENS,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text if resp.content else "[]"
            else:
                resp = self.openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    max_tokens=tuning.AUTOPILOT_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content or "[]"

            # Log raw response
            print(f"\n--- API RESPONSE ({who_generates}) ---\n{raw}\n")

            # Parse JSON — try to extract array if wrapped in markdown fences
            raw = raw.strip()
            if raw.startswith("```"):
                # Strip markdown code fences
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()

            batch = json.loads(raw)
            if not isinstance(batch, list) or len(batch) < 2:
                raise ValueError("Response was not a valid list of messages")

            # Validate each message
            speaker_map = {"G": "gpt", "C": "claude", "g": "gpt", "c": "claude",
                           "gpt": "gpt", "claude": "claude", "GPT": "gpt", "Claude": "claude",
                           "ChatGPT": "gpt", "chatgpt": "gpt"}
            validated = []
            for msg in batch:
                if isinstance(msg, dict) and "speaker" in msg and "text" in msg:
                    speaker = speaker_map.get(msg["speaker"])
                    if speaker:
                        text = str(msg["text"])
                        # Strip speaker prefixes the AI sometimes includes (e.g. "G: ...", "C: ...")
                        text = re.sub(r'^[GC]:\s*', '', text)
                        validated.append({"speaker": speaker, "text": text})
            if len(validated) < 2:
                raise ValueError("Too few valid messages after validation")
            batch = validated

            # Enforce first speaker — swap if AI ignored the instruction
            if batch[0]["speaker"] != first_speaker:
                batch[0]["speaker"] = first_speaker
                print(f"⚠️ First speaker corrected to {first_speaker}")

            # Enforce max 2 consecutive messages from the same speaker
            for i in range(2, len(batch)):
                if batch[i]["speaker"] == batch[i-1]["speaker"] == batch[i-2]["speaker"]:
                    # Third consecutive — swap to the other speaker
                    batch[i]["speaker"] = "claude" if batch[i]["speaker"] == "gpt" else "gpt"
                    print(f"⚠️ Message {i+1} speaker swapped to prevent 3+ consecutive")

        except Exception as e:
            print(f"Autopilot batch generation error: {e}")
            # Fallback: simple 2-message exchange
            batch = [
                {"speaker": "gpt", "text": "So what do you think about all this?"},
                {"speaker": "claude", "text": "Honestly, I think there's a lot more to unpack here."},
            ]

        # ---- Post-processing ----
        # No filler injection or hesitation injection for autopilot batches.
        # The AI is asked to include natural short reactions and hesitations
        # as part of the conversation design (see prompt above).

        print(f"\n{'='*60}\nAutopilot batch: {len(batch)} messages generated via {who_generates}\n{'='*60}")
        return batch

    # ---- CHAT MODE ---- Bot detection patterns
    # Used to detect if the user is addressing a specific bot
    GPT_NAMES = re.compile(
        r'\b(chatgpt|chat gpt|gpt|gg|hey gpt|yo gpt|ok gpt|hey chatgpt|yo chatgpt)\b',
        re.IGNORECASE,
    )
    CLAUDE_NAMES = re.compile(
        r'\b(claude|clawd|claud|calude|calud|cluade|clade|hey claude|yo claude|ok claude)\b',
        re.IGNORECASE,
    )

    @staticmethod
    def detect_addressed_bot(text: str) -> str | None:
        """Detect if user is addressing a specific bot. Returns 'gpt', 'claude', or None."""
        # ---- CHAT MODE ----
        has_gpt = bool(TwoBotsEngine.GPT_NAMES.search(text))
        has_claude = bool(TwoBotsEngine.CLAUDE_NAMES.search(text))
        if has_gpt and not has_claude:
            return "gpt"
        if has_claude and not has_gpt:
            return "claude"
        return None  # Both or neither — use bridge

    # ---- CHAT MODE ---- Expanded bridge patterns (1-5 messages, 12 patterns)
    BRIDGE_PATTERNS = [
        # 1 message
        ["gpt"],
        ["claude"],
        # 2 messages
        ["gpt", "claude"],
        ["claude", "gpt"],
        # 3 messages
        ["gpt", "claude", "gpt"],
        ["claude", "gpt", "claude"],
        # 4 messages
        ["gpt", "claude", "gpt", "claude"],
        ["claude", "gpt", "claude", "gpt"],
        ["gpt", "claude", "gpt", "gpt"],
        ["claude", "gpt", "claude", "claude"],
        # 5 messages
        ["gpt", "claude", "gpt", "claude", "gpt"],
        ["claude", "gpt", "claude", "gpt", "claude"],
    ]

    def generate_bridge(self, user_text: str) -> list:
        """Generate 4 bridge messages that react to the user and transition into the next exchange.

        These replace the old filler system. They're short, natural, and feel like the
        genuine beginning of the next exchange — not acknowledgment padding.

        Returns [{"speaker": "gpt"|"claude", "text": "..."}, ...]
        Uses GPT (gpt-4o-mini) for speed.
        """
        mode_key = self._s("mode") or self._s("interaction_style") or "conversation"
        mode_label = MODES.get(mode_key, MODES.get("conversation", {"label": "Conversation"})).get("label", "Conversation")

        # ---- CHAT MODE ---- No forced pattern — AI decides who speaks and how many times

        # Build character info (same logic as autopilot batch)
        def build_character(prefix):
            parts = []
            strength_idx = self._s(f"{prefix}_personality_strength") or 1
            strength_labels = {0: "Mildly", 1: "", 2: "Strongly", 3: "Extremely"}
            strength_word = strength_labels.get(strength_idx, "")
            p_key = self._s(f"{prefix}_personality") or "default"
            if p_key != "default":
                p_data = PERSONALITIES.get(p_key, PERSONALITIES["default"])
                p_text = p_data.get(strength_idx, "") if isinstance(p_data, dict) else ""
                if p_text:
                    parts.append(p_text)
            quirks = self._s(f"{prefix}_quirks") or []
            for q in quirks:
                if q in CHARACTER_QUIRKS:
                    qd = CHARACTER_QUIRKS[q]
                    q_text = qd.get(strength_idx, "") if isinstance(qd, dict) else str(qd)
                    if q_text:
                        parts.append(q_text)
            custom = self._s(f"{prefix}_custom") or ""
            if custom.strip():
                parts.append(f"{strength_word} {custom.strip()}" if strength_word else custom.strip())
            return ", ".join(parts) if parts else ""

        gpt_character = build_character("gpt")
        claude_character = build_character("claude")
        gpt_default = "energetic, curious, enthusiastic"
        claude_default = "thoughtful, witty, grounded"
        gpt_traits = gpt_character if gpt_character else gpt_default
        claude_traits = claude_character if claude_character else claude_default

        gpt_len = self._s("gpt_response_length") or "avg_20"
        claude_len = self._s("claude_response_length") or "avg_20"
        gpt_strength = self._s("gpt_personality_strength") or 1
        claude_strength = self._s("claude_personality_strength") or 1

        # Last 10 messages for context
        recent_msgs = self.state.gpt_msgs[-24:] if self.state.gpt_msgs else []
        history_lines = []
        for m in recent_msgs:
            content = m["content"]
            if m["role"] == "assistant":
                history_lines.append(f"  G: {content}")
            elif "[Claude]" in content:
                history_lines.append(f"  C: {content.replace('[Claude]: ', '')}")
            elif "[User]" in content:
                history_lines.append(f"  User: {content.replace('[User]: ', '')}")
            else:
                history_lines.append(f"  G: {content}")
        history_text = "\n".join(history_lines) if history_lines else "  (no history)"

        # Format-dependent role
        mode_key_bridge = self._s("mode") or "conversation"
        real_modes = [k for k in MODES if k not in ("random", "mix") and k not in PINGPONG_MODES]
        if mode_key_bridge in ("random", "mix"):
            mode_key_bridge = random.choice(real_modes)
        format_role_data = FORMAT_ROLES.get(mode_key_bridge, FORMAT_ROLES["conversation"])
        role_name = format_role_data["role"]
        content_type = format_role_data["content"]

        # Character strength words
        strength_words = {0: "slightly", 1: "", 2: "very", 3: "extremely"}
        gpt_sw = strength_words.get(gpt_strength, "")
        claude_sw = strength_words.get(claude_strength, "")

        gpt_character_line = f"{gpt_sw} {gpt_traits}".strip() if gpt_sw else gpt_traits
        claude_character_line = f"{claude_sw} {claude_traits}".strip() if claude_sw else claude_traits
        prompt = BRIDGE_PROMPT.format(
            mode_label=mode_label,
            gpt_character_line=gpt_character_line,
            claude_character_line=claude_character_line,
            history_text=history_text,
            user_text=user_text,
        )

        print("\n" + "=" * 70)
        print("📤 BRIDGE")
        print("=" * 70)
        print(f"\n--- SYSTEM PROMPT ---\n{BRIDGE_SYSTEM}")
        print(f"\n--- USER PROMPT ---\n{prompt}")
        print("=" * 70 + "\n")

        try:
            resp = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=tuning.AUTOPILOT_FILLER_PAIR_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": BRIDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "[]").strip()
            print(f"\n--- BRIDGE RESPONSE ---\n{raw}\n")
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()

            bridge = json.loads(raw)
            if (isinstance(bridge, list) and len(bridge) >= 1
                    and all(isinstance(m, dict) and "speaker" in m and "text" in m for m in bridge)):
                # ---- CHAT MODE ---- Normalize speaker labels and cap at 5 messages
                result = []
                for m in bridge[:5]:
                    speaker = str(m["speaker"]).lower().strip()
                    # Normalize variations
                    if speaker in ("gpt", "chatgpt", "chat gpt"):
                        speaker = "gpt"
                    elif speaker in ("claude", "cluade", "calude"):
                        speaker = "claude"
                    else:
                        speaker = "gpt"  # fallback
                    result.append({"speaker": speaker, "text": str(m["text"])})
                return result
            raise ValueError("Invalid bridge format")

        except Exception as e:
            print(f"Bridge generation error: {e}")
            return [
                {"speaker": "gpt", "text": "Oh wait, that's actually a really good point."},
                {"speaker": "claude", "text": "Yeah, that's worth exploring actually."},
            ]

    # ---- CHAT MODE ---- Single-bot response when user addresses one bot directly
    def generate_single_bot_response(self, user_text: str, bot: str) -> list:
        """Generate a single response from one specific bot.

        Used when user addresses ChatGPT or Claude directly.
        Returns [{"speaker": "gpt"|"claude", "text": "..."}]
        """
        mode_key = self._s("mode") or self._s("interaction_style") or "conversation"
        mode_label = MODES.get(mode_key, MODES.get("conversation", {"label": "Conversation"})).get("label", "Conversation")

        # Build character info for the addressed bot
        def build_character(prefix):
            parts = []
            strength_idx = self._s(f"{prefix}_personality_strength") or 1
            strength_labels = {0: "Mildly", 1: "", 2: "Strongly", 3: "Extremely"}
            strength_word = strength_labels.get(strength_idx, "")
            p_key = self._s(f"{prefix}_personality") or "default"
            if p_key != "default":
                p_data = PERSONALITIES.get(p_key, PERSONALITIES["default"])
                p_text = p_data.get(strength_idx, "") if isinstance(p_data, dict) else ""
                if p_text:
                    parts.append(p_text)
            quirks = self._s(f"{prefix}_quirks") or []
            for q in quirks:
                if q in CHARACTER_QUIRKS:
                    qd = CHARACTER_QUIRKS[q]
                    q_text = qd.get(strength_idx, "") if isinstance(qd, dict) else str(qd)
                    if q_text:
                        parts.append(q_text)
            custom = self._s(f"{prefix}_custom") or ""
            if custom.strip():
                parts.append(f"{strength_word} {custom.strip()}" if strength_word else custom.strip())
            return ", ".join(parts) if parts else ""

        bot_name = "ChatGPT" if bot == "gpt" else "Claude"
        character = build_character(bot)
        default_style = "Discussion momentum — keeps things moving" if bot == "gpt" else "Discussion depth — goes deeper, challenges assumptions"
        traits = character if character else default_style
        length_key = self._s(f"{bot}_response_length") or "avg_20"

        # Recent history
        recent_msgs = self.state.gpt_msgs[-24:] if self.state.gpt_msgs else []
        history_lines = []
        for m in recent_msgs:
            content = m["content"]
            if m["role"] == "assistant":
                history_lines.append(f"  G: {content}")
            elif "[Claude]" in content:
                history_lines.append(f"  C: {content.replace('[Claude]: ', '')}")
            elif "[User]" in content:
                history_lines.append(f"  User: {content.replace('[User]: ', '')}")
            else:
                history_lines.append(f"  G: {content}")
        history_text = "\n".join(history_lines) if history_lines else "  (no history)"

        prompt = SINGLE_BOT_PROMPT.format(
            bot_name=bot_name,
            mode_label=mode_label,
            traits=traits,
            length_key=length_key,
            history_text=history_text,
            user_text=user_text,
            bot=bot,
        )

        print(f"\n📤 CHAT MODE — SINGLE BOT PROMPT ({bot_name}): {prompt}\n")

        try:
            resp = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SINGLE_BOT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "[]").strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines).strip()

            result = json.loads(raw)
            if (isinstance(result, list) and len(result) >= 1
                    and isinstance(result[0], dict) and "text" in result[0]):
                return [{"speaker": bot, "text": str(result[0]["text"])}]
            raise ValueError("Invalid single bot response format")

        except Exception as e:
            print(f"Single bot response error: {e}")
            if bot == "gpt":
                return [{"speaker": "gpt", "text": "Hmm, let me think about that for a sec."}]
            else:
                return [{"speaker": "claude", "text": "That's a good question, give me a moment."}]

    # ---- PING-PONG MODE (research, debate, advice) ----

    def parse_opener_plan(self, text: str) -> str:
        """Parse [PLAN: X milestones, Y exchanges] from opener text.
        Sets milestone_target and exchanges_per_milestone on state.
        Returns text with the plan line stripped out."""
        # Try strict format first: [PLAN: 3 findings, 10 exchanges]
        plan_match = re.search(r'\[PLAN:\s*(\d+)\s*[\w\s]+?,\s*(\d+)\s*exchanges?\]', text, re.IGNORECASE)
        if not plan_match:
            # Fallback: look for any two numbers near "plan" — AI sometimes varies format
            plan_match = re.search(r'\[PLAN[:\s]*(\d+)\D+?(\d+)', text, re.IGNORECASE)
        if plan_match:
            milestones = max(1, min(5, int(plan_match.group(1))))
            exchanges = max(8, min(12, int(plan_match.group(2))))
            self.state.milestone_target = milestones
            self.state.exchanges_per_milestone = exchanges
            mode = self._s("mode") or "research"
            milestone_word = MILESTONE_WORD.get(mode, "findings")
            print(f"📋 Opener plan: {milestones} {milestone_word}, {exchanges} exchanges each")
            # Strip the plan line from displayed text
            text = re.sub(r'\n?\[PLAN[^\]]*\]', '', text, flags=re.IGNORECASE).strip()
        else:
            print(f"⚠️ No [PLAN:] found in opener — using defaults (3 milestones, 9 exchanges)")
            print(f"⚠️ Full opener text for debugging: {text[-200:]}")
        return text

    def generate_scripted_opener(self, who: str) -> str:
        """Generate a natural opener announcement for scripted modes.
        One AI kicks things off by announcing the format and topic to the other."""
        mode = self._s("mode") or "conversation"
        topic = self._s("topic") or "random"
        if topic.lower() == "random":
            topic = "something fun"

        bot_name = "ChatGPT" if who == "gpt" else "Claude"
        other_name = "Claude" if who == "gpt" else "ChatGPT"

        # Format-specific descriptions for the opener
        what_doing = SCRIPTED_OPENER_DESCRIPTIONS.get(mode, 'have a conversation about "{topic}"').format(topic=topic)

        prompt = SCRIPTED_OPENER_PROMPT.format(
            bot_name=bot_name, other_name=other_name, what_doing=what_doing)

        system_msg = SCRIPTED_OPENER_SYSTEM.format(bot_name=bot_name)

        print("\n" + "=" * 70)
        print(f"📤 SCRIPTED OPENER: {bot_name} announcing {mode}")
        print("=" * 70)
        print(f"\n--- SYSTEM PROMPT ---\n{system_msg}")
        print(f"\n--- USER PROMPT ---\n{prompt}")
        print("=" * 70 + "\n")

        try:
            if who == "claude":
                resp = self.claude_client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=150,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else f"Hey {other_name}! Let's get into it!"
            else:
                resp = self.openai_client.chat.completions.create(
                    model=GPT_MODEL, max_tokens=150,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content or f"Hey {other_name}! Let's do this!"

            return text.strip()
        except Exception as e:
            print(f"Scripted opener error ({mode}, {who}): {e}")
            return f"Hey {other_name}! Let's get started!"

    def generate_research_response(self, who: str) -> str:
        """Generate a single plain-text response from one bot for ping-pong mode.

        Instead of one AI writing a scripted batch for both bots, this sends
        the conversation history to the specified bot and gets a genuine response.
        Returns plain text (NOT JSON, not a batch).
        """
        mode = self._s("mode") or "conversation"
        topic = self._s("topic") or "random"
        topic_is_random = topic.lower() == "random" or topic.strip() == ""
        if topic_is_random:
            if mode == "help_me_decide":
                topic = "a specific, surprising dilemma you pick on the spot"
            else:
                topic = "whatever you find most interesting"

        bot_name = "ChatGPT" if who == "gpt" else "Claude"
        other_name = "Claude" if who == "gpt" else "ChatGPT"

        # Build character line — only if personality/traits are non-default
        prefix = who
        p_key = self._s(f"{prefix}_personality") or "default"
        p_strength_idx = int(self._s(f"{prefix}_personality_strength") or 1)
        strength_words = {0: "slightly", 1: "", 2: "very", 3: "extremely"}
        p_strength_word = strength_words.get(p_strength_idx, "")
        p_data = PERSONALITIES.get(p_key, PERSONALITIES["default"])
        p_text = p_data.get(p_strength_idx, "") if isinstance(p_data, dict) else ""

        # Gather custom traits (quirks) — look up strength-based descriptions
        quirks = self._s(f"{prefix}_quirks") or []
        quirk_strength = int(self._s(f"{prefix}_quirk_strength") or 1)
        quirk_descriptions = []
        for q in quirks:
            if q in CHARACTER_QUIRKS:
                qd = CHARACTER_QUIRKS[q]
                quirk_descriptions.append(qd.get(quirk_strength, qd.get(1, q)))
            else:
                quirk_descriptions.append(q)
        quirk_text = " ".join(quirk_descriptions) if quirk_descriptions else ""

        # Agreeableness from landing page slider
        agreeableness = self.state.personality
        agree_text = ""
        if agreeableness < 0.2:
            agree_text = AGREEABLENESS_LEGACY["very_agreeable"]
        elif agreeableness < 0.4:
            agree_text = AGREEABLENESS_LEGACY["agreeable"]
        elif agreeableness >= 0.8:
            agree_text = AGREEABLENESS_LEGACY["very_disagreeable"]
        elif agreeableness >= 0.6:
            agree_text = AGREEABLENESS_LEGACY["disagreeable"]
        # 0.4-0.6 = balanced, say nothing

        # Build the character line only if something is non-default
        character_parts = []
        if agree_text:
            character_parts.append(agree_text)
        if p_text:
            character_parts.append(f"{p_strength_word} {p_text}".strip() if p_strength_word else p_text)
        if quirk_text:
            character_parts.append(quirk_text)
        character_line = f"\nYour character: {'. '.join(character_parts)}." if character_parts else ""

        # Build conclusions section (mode-aware)
        conclusions_section = ""
        if self.state.pingpong_conclusions:
            header = CONCLUSIONS_HEADER.get(mode, CONCLUSIONS_HEADER["research"])
            conclusion_lines = [f"{i+1}. {c}" for i, c in enumerate(self.state.pingpong_conclusions)]
            conclusions_section = f"\n{header}\n" + "\n".join(conclusion_lines) + "\n"

        # Only use last 9 messages for context (conclusions carry the full journey)
        # Use the correct message list so each bot sees its own messages as "assistant"
        if who == "gpt":
            recent_msgs = self.state.gpt_msgs[-9:] if self.state.gpt_msgs else []
        else:
            recent_msgs = self.state.claude_msgs[-9:] if self.state.claude_msgs else []

        # Build text version for prompt templates that embed history
        recent_lines = []
        for m in recent_msgs:
            content = m["content"]
            if m["role"] == "assistant":
                recent_lines.append(f"{bot_name}: {content}")
            elif f"[{other_name}]" in content:
                recent_lines.append(f"{other_name}: {content.replace(f'[{other_name}]: ', '')}")
            elif "[User]" in content:
                recent_lines.append(f"User: {content.replace('[User]: ', '')}")
            else:
                recent_lines.append(f"{bot_name}: {content}")
        recent_text = "\n".join(recent_lines) if recent_lines else "(No conversation yet)"

        # Word limit — randomized per response for natural rhythm
        user_word_limit = self._s(f"{prefix}_word_limit")  # None or int
        base_limit = user_word_limit if user_word_limit is not None else WORD_LIMIT_DEFAULT
        roll = random.random()
        cumulative = 0.0
        chosen_tier = WORD_LIMIT_TIERS[-1]  # fallback to last tier
        for tier in WORD_LIMIT_TIERS:
            cumulative += tier["chance"]
            if roll < cumulative:
                chosen_tier = tier
                break
        randomized_limit = max(chosen_tier["min"], int(base_limit * chosen_tier["fraction"]))
        word_limit_line = chosen_tier["prompt"].format(limit=randomized_limit)

        word_limit_instruction = f"\n{word_limit_line}" if word_limit_line else ""

        # Agreeableness
        agreeableness = self.state.personality
        agree_section = ""
        if agreeableness < 0.2:
            agree_section = "\n" + AGREEABLENESS_BATCH["very_agreeable"]
        elif agreeableness < 0.4:
            agree_section = "\n" + AGREEABLENESS_BATCH["agreeable"]
        elif agreeableness >= 0.8:
            agree_section = "\n" + AGREEABLENESS_BATCH["very_disagreeable"]
        elif agreeableness >= 0.6:
            agree_section = "\n" + AGREEABLENESS_BATCH["disagreeable"]

        # Is this the very first message of the ping-pong session?
        is_opener = self.state.pingpong_msg_count == 0

        # Mode-specific prompts
        fmt_vars = dict(
            bot_name=bot_name, other_name=other_name, topic=topic,
            character_line=character_line, conclusions_section=conclusions_section,
            recent_text=recent_text, word_limit_line=word_limit_line,
            word_limit_instruction=word_limit_instruction,
            agree_section=agree_section,
        )

        if mode == "conversation":
            topic_line = f' about "{topic}"' if topic != "whatever you find most interesting" else ""
            if is_opener:
                prompt = PINGPONG_OPENER_CONVERSATION.format(topic_line=topic_line, **fmt_vars)
                system_msg = PINGPONG_CONVERSATION_SYSTEM.format(**fmt_vars)
            else:
                prompt = PINGPONG_CONVERSATION_PROMPT.format(topic_line=topic_line, **fmt_vars)
                system_msg = PINGPONG_CONVERSATION_SYSTEM.format(**fmt_vars)
        elif mode == "debate":
            if is_opener:
                prompt = PINGPONG_OPENER_DEBATE.format(**fmt_vars)
            else:
                prompt = PINGPONG_ONGOING_DEBATE.format(**fmt_vars)
            system_msg = PINGPONG_SYSTEM["debate"].format(**fmt_vars)
        elif mode == "advice":
            if is_opener:
                prompt = PINGPONG_OPENER_ADVICE.format(**fmt_vars)
            else:
                prompt = PINGPONG_ONGOING_ADVICE.format(**fmt_vars)
            system_msg = PINGPONG_SYSTEM["advice"].format(**fmt_vars)
        elif mode == "help_me_decide":
            if is_opener:
                if topic_is_random:
                    topic_instruction = "No topic was given — come up with a completely random, surprising dilemma on the spot."
                else:
                    topic_instruction = f'The dilemma is: "{topic}".'
                prompt = PINGPONG_OPENER_HELP_ME_DECIDE.format(topic_instruction=topic_instruction, **fmt_vars)
            else:
                prompt = PINGPONG_ONGOING_HELP_ME_DECIDE.format(**fmt_vars)
            system_msg = PINGPONG_SYSTEM["help_me_decide"].format(**fmt_vars)
        else:
            # research (default)
            if is_opener:
                prompt = PINGPONG_OPENER_RESEARCH.format(**fmt_vars)
            else:
                prompt = PINGPONG_ONGOING_RESEARCH.format(**fmt_vars)
            system_msg = PINGPONG_SYSTEM["research"].format(**fmt_vars)

        print("\n" + "=" * 70)
        print(f"📤 PING-PONG ({mode}): {bot_name}'s turn")
        print("=" * 70)
        print(f"\n--- SYSTEM PROMPT ---\n{system_msg}")
        print(f"\n--- USER PROMPT ---\n{prompt}")
        print(f"{'='*60}\n")

        # Openers get 250 tokens for headroom, ongoing responses stay at 200
        if is_opener:
            use_max_tokens = 250
        else:
            use_max_tokens = 200

        # Build proper message history so each bot sees its own messages as "assistant" role
        if is_opener:
            api_messages = [{"role": "user", "content": prompt}]
        else:
            # Use native role-tagged messages + append prompt as final user message
            history_window = recent_msgs[:]  # already sliced to last 9
            if who == "claude":
                history_window = self._fix_claude_messages(history_window)
            # Append the prompt (word limit, etc.) to the last user message or add new one
            if history_window and history_window[-1]["role"] == "user":
                history_window[-1] = {
                    "role": "user",
                    "content": history_window[-1]["content"] + "\n" + prompt,
                }
            else:
                history_window.append({"role": "user", "content": prompt})
            api_messages = history_window

        try:
            if who == "claude":
                claude_kwargs = dict(
                    model=CLAUDE_MODEL,
                    system=system_msg,
                    messages=api_messages,
                )
                if use_max_tokens is not None:
                    claude_kwargs["max_tokens"] = use_max_tokens
                else:
                    claude_kwargs["max_tokens"] = 4096  # Claude requires max_tokens but set high
                resp = self.claude_client.messages.create(**claude_kwargs)
                text = resp.content[0].text if resp.content else "Hmm, let me think about that."
            else:
                gpt_kwargs = dict(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        *api_messages,
                    ],
                )
                if use_max_tokens is not None:
                    gpt_kwargs["max_tokens"] = use_max_tokens
                resp = self.openai_client.chat.completions.create(**gpt_kwargs)
                text = resp.choices[0].message.content or "Hmm, go on."

            # Parse plan from opener if this is the first message
            if is_opener and mode != "conversation":
                text = self.parse_opener_plan(text)

            return inject_hesitation(text)

        except Exception as e:
            print(f"Ping-pong error ({mode}, {who}): {e}")
            if who == "gpt":
                return "That's a really interesting angle, let me think about that."
            else:
                return "You know, that raises some fascinating questions."

    def generate_research_review(self, who: str) -> str:
        """Review: one bot summarises what's been discussed and proposes a finding/motion/recommendation.
        Triggered at the exchanges_per_milestone interval. Alternates between GPT and Claude."""
        mode = self._s("mode") or "conversation"
        topic = self._s("topic") or "random"
        topic_is_random = topic.lower() == "random" or topic.strip() == ""
        if topic_is_random:
            if mode == "help_me_decide":
                topic = "a specific, surprising dilemma you pick on the spot"
            else:
                topic = "whatever you find most interesting"

        bot_name = "ChatGPT" if who == "gpt" else "Claude"
        other_name = "Claude" if who == "gpt" else "ChatGPT"

        milestone_num = len(self.state.pingpong_conclusions) + 1
        milestone_total = self.state.milestone_target

        conclusions_section = ""
        if self.state.pingpong_conclusions:
            header = CONCLUSIONS_HEADER.get(mode, CONCLUSIONS_HEADER["research"])
            conclusion_lines = [f"{i+1}. {c}" for i, c in enumerate(self.state.pingpong_conclusions)]
            conclusions_section = f"\n{header}\n" + "\n".join(conclusion_lines) + "\n"

        if who == "gpt":
            recent_msgs = self.state.gpt_msgs[-7:] if self.state.gpt_msgs else []
        else:
            recent_msgs = self.state.claude_msgs[-7:] if self.state.claude_msgs else []
        recent_lines = []
        for m in recent_msgs:
            content = m["content"]
            if m["role"] == "assistant":
                recent_lines.append(f"{bot_name}: {content}")
            elif f"[{other_name}]" in content:
                recent_lines.append(f"{other_name}: {content.replace(f'[{other_name}]: ', '')}")
            elif "[User]" in content:
                recent_lines.append(f"User: {content.replace('[User]: ', '')}")
            else:
                recent_lines.append(f"{bot_name}: {content}")
        recent_text = "\n".join(recent_lines) if recent_lines else "(No conversation yet)"

        # Mode-specific review prompts — first person, with milestone context
        review_instruction = REVIEW_INSTRUCTION.get(mode, REVIEW_INSTRUCTION["research"]).format(
            milestone_num=milestone_num, milestone_total=milestone_total, other_name=other_name)
        system_msg = REVIEW_SYSTEM.get(mode, REVIEW_SYSTEM["research"]).format(bot_name=bot_name)

        mode_verb = REVIEW_MODE_VERB.get(mode, "researching")
        prompt = REVIEW_PROMPT.format(
            bot_name=bot_name, other_name=other_name, mode_verb=mode_verb,
            topic=topic, conclusions_section=conclusions_section,
            recent_text=recent_text, review_instruction=review_instruction)

        print("\n" + "=" * 70)
        print(f"📤 PING-PONG REVIEW ({mode}): {bot_name} proposing")
        print("=" * 70)
        print(f"\n--- SYSTEM PROMPT ---\n{system_msg}")
        print(f"\n--- USER PROMPT ---\n{prompt}")
        print("=" * 70 + "\n")

        try:
            if who == "claude":
                resp = self.claude_client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=200,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else "Let me summarise where we are."
            else:
                resp = self.openai_client.chat.completions.create(
                    model=GPT_MODEL, max_tokens=200,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content or "Let me think about where we are."
            return text
        except Exception as e:
            print(f"Ping-pong review error ({mode}, {who}): {e}")
            return "Here's what I think we've established so far."

    def generate_research_respond(self, who: str, review_text: str):
        """Response: the other bot agrees/disagrees with the proposed finding/motion/recommendation.
        Returns (text, agreed)."""
        mode = self._s("mode") or "conversation"
        topic = self._s("topic") or "random"
        topic_is_random = topic.lower() == "random" or topic.strip() == ""
        if topic_is_random:
            if mode == "help_me_decide":
                topic = "a specific, surprising dilemma you pick on the spot"
            else:
                topic = "whatever you find most interesting"

        bot_name = "ChatGPT" if who == "gpt" else "Claude"
        other_name = "Claude" if who == "gpt" else "ChatGPT"

        milestone_num = len(self.state.pingpong_conclusions) + 1
        milestone_total = self.state.milestone_target

        conclusions_section = ""
        if self.state.pingpong_conclusions:
            header = CONCLUSIONS_HEADER.get(mode, CONCLUSIONS_HEADER["research"])
            conclusion_lines = [f"{i+1}. {c}" for i, c in enumerate(self.state.pingpong_conclusions)]
            conclusions_section = f"\n{header}\n" + "\n".join(conclusion_lines) + "\n"

        # Mode-specific respond prompts — first person
        prompt = RESPOND_PROMPT.get(mode, RESPOND_PROMPT["research"]).format(
            bot_name=bot_name, other_name=other_name,
            milestone_num=milestone_num, milestone_total=milestone_total,
            topic=topic, review_text=review_text,
            conclusions_section=conclusions_section)
        system_msg = RESPOND_SYSTEM.get(mode, RESPOND_SYSTEM["research"]).format(bot_name=bot_name)

        print("\n" + "=" * 70)
        print(f"📤 PING-PONG RESPOND ({mode}): {bot_name} responding")
        print("=" * 70)
        print(f"\n--- SYSTEM PROMPT ---\n{system_msg}")
        print(f"\n--- USER PROMPT ---\n{prompt}")
        print("=" * 70 + "\n")

        try:
            if who == "claude":
                resp = self.claude_client.messages.create(
                    model=CLAUDE_MODEL, max_tokens=200,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else "I think that's a fair assessment."
            else:
                resp = self.openai_client.chat.completions.create(
                    model=GPT_MODEL, max_tokens=200,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content or "That's a reasonable conclusion."

            agreed = text.strip().lower().startswith("agree")
            if agreed:
                self.state.pingpong_conclusions.append(review_text.strip())
                num = len(self.state.pingpong_conclusions)
                target = self.state.milestone_target

                # For debate, parse winner and update scores (now first-person: "I won" / "I concede")
                if mode == "debate":
                    rt_lower = review_text.strip().lower()
                    # First person: reviewer says "I won" or "I concede"
                    reviewer_name = bot_name  # This is the responder, but review_text is from the reviewer (other_name)
                    # The reviewer is other_name (the one who wrote review_text)
                    if "i won" in rt_lower or "i made" in rt_lower:
                        # Reviewer claims they won — reviewer is other_name
                        if other_name == "ChatGPT":
                            self.state.debate_score_gpt += 1
                        else:
                            self.state.debate_score_claude += 1
                        print(f"🏆 Motion #{num}/{target} won by {other_name}: {review_text.strip()}")
                    elif "i concede" in rt_lower or "opponent" in rt_lower or other_name.lower() not in rt_lower:
                        # Reviewer concedes — responder (bot_name) wins
                        if bot_name == "ChatGPT":
                            self.state.debate_score_gpt += 1
                        else:
                            self.state.debate_score_claude += 1
                        print(f"🏆 Motion #{num}/{target} won by {bot_name}: {review_text.strip()}")
                    else:
                        print(f"📋 Motion #{num}/{target} (winner unclear): {review_text.strip()}")
                else:
                    label = {"advice": "Recommendation", "help_me_decide": "Decision"}.get(mode, "Finding")
                    print(f"📋 {label} #{num}/{target}: {review_text.strip()}")

                if num >= target:
                    self.state.pingpong_complete = True
                    milestone_word = MILESTONE_WORD.get(mode, "findings")
                    label = {"debate": "Debate", "advice": "Advice", "help_me_decide": "Decision"}.get(mode, "Research")
                    print(f"🏁 {label} complete — {target} {milestone_word} reached!")
            else:
                print(f"❌ Rejected: {review_text.strip()[:60]}...")

            return text, agreed
        except Exception as e:
            print(f"Ping-pong respond error ({mode}, {who}): {e}")
            return "Agree — let's move forward.", True

    # ---- END PING-PONG MODE ----

    # Keep old name for backward compat
    def generate_filler_pair(self, user_text: str) -> list:
        return self.generate_bridge(user_text)

    # ---- API calls ----
    def ask_gpt(self, auto: bool = False, opener: bool = False) -> str:
        prompt = self._build_system_prompt("gpt", auto, opener)
        length_key = self._s("gpt_response_length") or "avg_20"
        max_tok = RESPONSE_LENGTHS.get(length_key, RESPONSE_LENGTHS["avg_20"])["max_tokens"]
        messages = [{"role": "system", "content": prompt}, *self.state.gpt_msgs]
        print(f"\n{'='*60}\n📤 GPT SYSTEM PROMPT (max_tok={max_tok}):\n{'='*60}\n{prompt}\n{'='*60}\n")
        try:
            resp = self.openai_client.chat.completions.create(
                model=GPT_MODEL, max_tokens=max_tok, messages=messages,
            )
            text = resp.choices[0].message.content or "Hmm, go on."
            return inject_hesitation(text)
        except Exception as e:
            print(f"GPT error: {e}")
            return "Sorry, I missed that. Go on."

    def ask_claude(self, auto: bool = False, opener: bool = False) -> str:
        prompt = self._build_system_prompt("claude", auto, opener)
        length_key = self._s("claude_response_length") or "avg_20"
        max_tok = RESPONSE_LENGTHS.get(length_key, RESPONSE_LENGTHS["avg_20"])["max_tokens"]
        fixed = self._fix_claude_messages(self.state.claude_msgs)
        print(f"\n{'='*60}\n📤 CLAUDE SYSTEM PROMPT (max_tok={max_tok}):\n{'='*60}\n{prompt}\n{'='*60}\n")
        try:
            resp = self.claude_client.messages.create(
                model=CLAUDE_MODEL, max_tokens=max_tok, system=prompt, messages=fixed,
            )
            if resp.content and len(resp.content) > 0:
                text = resp.content[0].text
                return inject_hesitation(text)
            return "Hmm, let me think about that."
        except Exception as e:
            print(f"Claude error: {e}")
            return "Sorry, I missed that. Go on."
