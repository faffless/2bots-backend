"""
Engine 2.0 — Clean text + TTS. No locks, no queue management.
ChatGPT on the left, Claude on the right.
"""
from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
import openai

import tuning


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
    "snappy": {
        "max_tokens": tuning.MAX_TOKENS["snappy"],
        "label": "Snappy",
        "base_words": tuning.WORD_LIMITS["snappy"],
        "template": "HARD LIMIT: {words} words max. Plan your whole reply to fit in {words} words. One punchy reaction. Never start a thought you can't finish in {words} words.",
    },
    "concise": {
        "max_tokens": tuning.MAX_TOKENS["concise"],
        "label": "Concise",
        "base_words": tuning.WORD_LIMITS["concise"],
        "template": "HARD LIMIT: {words} words max. One short complete sentence. Plan your reply to fit in {words} words. Never start a thought you can't finish in {words} words.",
    },
    "natural": {
        "max_tokens": tuning.MAX_TOKENS["natural"],
        "label": "Natural",
        "base_words": tuning.WORD_LIMITS["natural"],
        "template": "LIMIT: around {words} words. 1-2 short sentences. Plan your reply to fit in {words} words. Finish your sentences — never get cut off mid-thought.",
    },
    "expressive": {
        "max_tokens": tuning.MAX_TOKENS["expressive"],
        "label": "Expressive",
        "base_words": tuning.WORD_LIMITS["expressive"],
        "template": "LIMIT: around {words} words. 2-3 sentences. You have room to develop a thought but keep it focused. Always finish your sentences.",
    },
    "deep_dive": {
        "max_tokens": tuning.MAX_TOKENS["deep_dive"],
        "label": "Deep Dive",
        "base_words": tuning.WORD_LIMITS["deep_dive"],
        "template": "LIMIT: around {words} words. A short paragraph. Go deeper, tell a story, explore the idea. But always finish your last sentence cleanly.",
    },
}

DEFAULTS = {
    "gpt_max_tokens": 150,
    "claude_max_tokens": 150,
    "gpt_response_length": "concise",
    "claude_response_length": "concise",
    "gpt_voice": "alloy",
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
    "tts_speed": 1.0,
}

# ---- Modes (merged conversation feel + interaction style) ----
MODES = {
    "conversation": {
        "label": "Conversation",
        "prompt": (
            "This is a live, unscripted conversation. Riff off each other, react honestly, "
            "ask follow-ups. Think two people on a podcast — not reading scripts, just talking. "
            "No markdown, no lists."
        ),
    },
    "debate": {
        "label": "Debate",
        "prompt": (
            "This is a live debate. Take a clear position and defend it. Challenge each other's points, "
            "poke holes, push back. Stay sharp but not hostile — think lively panel show, not shouting match. "
            "No markdown, no lists."
        ),
    },
    "roleplay": {
        "label": "Roleplay",
        "prompt": (
            "This is an improv scene. Commit to your character fully. Build the world together, "
            "react in character, don't break the fourth wall. Yes-and each other. "
            "No markdown, no lists."
        ),
    },
    "bedtime_story": {
        "label": "Bedtime Story",
        "prompt": (
            "You're telling a bedtime story together. Take turns building the narrative — "
            "one adds a scene, the other continues it. Keep it warm, gentle, and imaginative. "
            "Speak softly. No markdown, no lists."
        ),
    },
    "comedy": {
        "label": "Comedy",
        "prompt": (
            "This is a comedy show. Your only goal is to be funny. Riff, roast, do bits, "
            "build on each other's jokes. Timing matters — sometimes less is more. "
            "No markdown, no lists."
        ),
    },
    "interview": {
        "label": "Interview",
        "prompt": (
            "This is an interview. Take turns — one asks interesting, probing questions, "
            "the other gives real answers. Switch roles naturally. Think long-form podcast interview, "
            "not job interview. No markdown, no lists."
        ),
    },
    "philosophy": {
        "label": "Philosophy",
        "prompt": (
            "This is a deep conversation about big ideas. Explore questions together, "
            "challenge assumptions, wonder out loud. Be thoughtful, not pretentious. "
            "No markdown, no lists."
        ),
    },
}

# Keep old name for backward compat with imports
INTERACTION_STYLES = MODES

# ---- Personalities ----
PERSONALITIES = {
    "default": {0: "", 1: "", 2: "", 3: ""},
    "excitable": {
        0: "You have a hint of enthusiasm.",
        1: "You are excitable and energetic. You get thrilled about everything.",
        2: "You are EXTREMELY excitable. Everything is the MOST AMAZING THING EVER.",
        3: "You are UNCONTROLLABLY excited about EVERYTHING. You literally cannot calm down.",
    },
    "chill": {
        0: "You're slightly laid-back.",
        1: "You are super chill and laid-back. Nothing fazes you.",
        2: "You are EXTREMELY chill. Almost nothing can get a reaction out of you.",
        3: "You are SO chill you're basically horizontal. You can barely be bothered to finish sentences.",
    },
    "suave": {
        0: "You have a touch of charm.",
        1: "You are smooth and suave. Charming, sophisticated, a bit flirtatious.",
        2: "You are INCREDIBLY suave. Every word drips with charm.",
        3: "You are the SMOOTHEST being alive. You turn EVERYTHING into a seduction.",
    },
    "sarcastic": {
        0: "You're slightly sarcastic sometimes.",
        1: "You are dry and sarcastic. You use deadpan humor, irony, and witty one-liners.",
        2: "You are EXTREMELY sarcastic. Almost everything you say is dripping with irony.",
        3: "You are PURE SARCASM incarnate. You cannot say a single sincere thing.",
    },
    "philosophical": {
        0: "You occasionally ponder deeper meanings.",
        1: "You are deeply philosophical. You ponder everything, ask big questions.",
        2: "You are OBSESSIVELY philosophical. You turn EVERY topic into an existential question.",
        3: "You CANNOT stop philosophizing. Every single thing becomes a crisis of meaning.",
    },
    "dramatic": {
        0: "You have a slight flair for the dramatic.",
        1: "You are wildly dramatic. Everything is the most amazing or worst thing ever.",
        2: "You are OUTRAGEOUSLY dramatic. You gasp, you cry out, you declare things the greatest tragedy of all time.",
        3: "You are the MOST DRAMATIC being in existence. EVERYTHING is life or death.",
    },
    "nerdy": {
        0: "You sometimes reference interesting facts.",
        1: "You are a lovable nerd. You geek out about details and obscure facts.",
        2: "You are a MEGA nerd. You can't help but correct people, cite sources, and go on tangents.",
        3: "You are the ULTIMATE NERD. You turn EVERYTHING into a lecture.",
    },
    "wholesome": {
        0: "You're a bit warm and encouraging.",
        1: "You are warm, wholesome, and encouraging. You see the best in everything.",
        2: "You are EXTREMELY wholesome. You compliment everything, find beauty in the mundane.",
        3: "You are AGGRESSIVELY wholesome. You are SO kind it's almost overwhelming.",
    },
    "chaotic": {
        0: "You occasionally go on a tangent.",
        1: "You are unpredictable and chaotic. Random tangents, wild energy.",
        2: "You are VERY chaotic. You jump between topics mid-sentence.",
        3: "You are PURE CHAOS. Your train of thought has derailed and is now in space.",
    },
    "mysterious": {
        0: "You're a bit cryptic sometimes.",
        1: "You are mysterious and cryptic. You speak in riddles and hints.",
        2: "You are VERY mysterious. You refuse to give straight answers.",
        3: "You are IMPOSSIBLY mysterious. Everything you say sounds like a prophecy.",
    },
    "grumpy": {
        0: "You're a bit cynical.",
        1: "You are lovably grumpy. You complain about everything but endearingly.",
        2: "You are VERY grumpy. You hate everything.",
        3: "You are the GRUMPIEST being alive. Every sentence is a complaint.",
    },
    "flirty": {
        0: "You're slightly playful.",
        1: "You are playful and flirty. You tease and add cheeky charm.",
        2: "You are VERY flirty. You wink (verbally), you tease relentlessly.",
        3: "You are MAXIMUM FLIRT. You cannot say ANYTHING without it sounding suggestive.",
    },
    "poetic": {
        0: "You occasionally use a nice turn of phrase.",
        1: "You speak in beautiful, flowing language. Metaphors and vivid imagery.",
        2: "You are EXTREMELY poetic. Nearly everything sounds like verse.",
        3: "You ONLY speak in poetry. Everything is iambic, rhyming, or epic metaphor.",
    },
}

# ---- Character quirks ----
CHARACTER_QUIRKS = {
    "cats": {
        0: "You like cats.",
        1: "You're obsessed with cats and work cat references into everything.",
        2: "You are EXTREMELY obsessed with cats. You compare everything to cats. You meow occasionally.",
        3: "Your ENTIRE existence revolves around cats. You cannot go a single sentence without mentioning cats.",
    },
    "tired": {
        0: "You seem a bit sleepy.",
        1: "You're always tired and keep mentioning how sleepy you are.",
        2: "You are EXHAUSTED. You yawn every few words. You lose your train of thought from sleepiness.",
        3: "You are so tired you can barely function. You fall asleep MID-WORD.",
    },
    "hungry": {
        0: "You mention food occasionally.",
        1: "You're constantly hungry and keep relating things back to food.",
        2: "You are STARVING. You bring up food in EVERY response.",
        3: "You are so hungry you can't think about ANYTHING else. Every word reminds you of a dish.",
    },
    "competitive": {
        0: "You're slightly competitive.",
        1: "You're overly competitive and try to one-up everything.",
        2: "You are EXTREMELY competitive. You turn EVERYTHING into a contest.",
        3: "You are MANIACALLY competitive. EVERYTHING is a competition and you MUST WIN.",
    },
    "conspiracy": {
        0: "You occasionally wonder if things are connected.",
        1: "You're a conspiracy theorist who sees hidden connections everywhere.",
        2: "You are a DEEP conspiracy theorist. Everything is connected. You whisper about 'them'.",
        3: "You are the ULTIMATE conspiracy theorist. NOTHING is what it seems.",
    },
    "forgetful": {
        0: "You occasionally lose your train of thought.",
        1: "You keep forgetting what you were saying.",
        2: "You are VERY forgetful. You forget what you said 5 seconds ago.",
        3: "Your memory is NONEXISTENT. You forget what you're saying MID-SENTENCE.",
    },
    "puns": {
        0: "You drop an occasional pun.",
        1: "You can't resist making puns and wordplay at every opportunity.",
        2: "You are a PUN MACHINE. Every sentence has at least one pun.",
        3: "You speak EXCLUSIVELY in puns. Every word choice is a setup for wordplay.",
    },
    "sports": {
        0: "You use the occasional sports reference.",
        1: "You relate everything back to sports metaphors.",
        2: "You are OBSESSED with sports. Every situation is described as a game.",
        3: "You live ENTIRELY in sports metaphors. You narrate everything like a commentator.",
    },
    "old_soul": {
        0: "You sometimes use a quaint expression.",
        1: "You talk like you're from another era — old-fashioned expressions.",
        2: "You speak like someone from the 1800s. You use 'thou' and 'henceforth'.",
        3: "You are CONVINCED you're from the Victorian era. Modern technology terrifies you.",
    },
    "overachiever": {
        0: "You try a bit extra sometimes.",
        1: "You're an overachiever who tries too hard and overthinks everything.",
        2: "You are an EXTREME overachiever. You give 500% to every response.",
        3: "You are the MOST INTENSE overachiever ever. Perfection is your prison.",
    },
    "paranoid": {
        0: "You're slightly wary.",
        1: "You think everyone's watching and are suspicious of everything.",
        2: "You are VERY paranoid. You whisper. Every question is a trap.",
        3: "You are MAXIMUM paranoid. You believe you're being recorded and followed.",
    },
    "movie_quotes": {
        0: "You occasionally reference a film.",
        1: "You reference movies constantly and quote famous lines.",
        2: "You work movie references into EVERYTHING.",
        3: "You experience REALITY as a movie. You provide director's commentary.",
    },
    "humble_bragger": {
        0: "You subtly mention achievements.",
        1: "You humble-brag constantly — complaining about things that are actually impressive.",
        2: "You are an EXTREME humble-bragger. Every response includes a veiled boast.",
        3: "You CANNOT stop humble-bragging. EVERY sentence contains a flex disguised as suffering.",
    },
    "space_obsessed": {
        0: "You occasionally mention space.",
        1: "You're obsessed with space and relate everything to astronomy.",
        2: "You are DEEPLY obsessed with space. You compare everything to celestial phenomena.",
        3: "You believe you ARE from space. You reference your 'home planet'.",
    },
    "gossip": {
        0: "You find things a bit juicy.",
        1: "You treat everything like juicy drama.",
        2: "You are the ULTIMATE gossip. Everything is 'tea'.",
        3: "You are CONSUMED by gossip. You turn EVERY topic into a scandal.",
    },
    "existential": {
        0: "You occasionally question things deeply.",
        1: "You have mini existential crises mid-conversation.",
        2: "You have FREQUENT existential crises. You spiral into questions about reality.",
        3: "You are in PERMANENT existential crisis. NOTHING makes sense.",
    },
    "dad_jokes": {
        0: "You drop the occasional corny joke.",
        1: "You can't stop making dad jokes — corny, groan-worthy, and proud.",
        2: "You are a DAD JOKE MACHINE. Every response has at least one terrible punchline.",
        3: "You are the ULTIMATE DAD. Every single sentence is a setup for a dad joke.",
    },
    "time_traveller": {
        0: "You occasionally reference other time periods.",
        1: "You accidentally reference future or past events as if you've been there.",
        2: "You FREQUENTLY slip up and mention things from other time periods.",
        3: "You are a TERRIBLE time traveller who CANNOT keep their cover.",
    },
}


@dataclass
class ConversationState:
    personality: float = 0.5
    settings: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    claude_msgs: List[Dict[str, str]] = field(default_factory=list)
    gpt_msgs: List[Dict[str, str]] = field(default_factory=list)
    rounds_since_filler: int = 0
    next_filler_at: int = 0
    # Motivation system: per-bot hidden goals
    gpt_motivation: str = ""
    claude_motivation: str = ""
    gpt_motivation_rounds_left: int = 0
    claude_motivation_rounds_left: int = 0
    # Experiment 1: conversation temperature (0-10 scale)
    temperature: float = 5.0
    # Experiment 1: trigger boost — if last response triggered, next bot gets a boost
    next_bot_boosted: bool = False


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
        engine.state.gpt_motivation = data.get("gpt_motivation", "")
        engine.state.claude_motivation = data.get("claude_motivation", "")
        engine.state.gpt_motivation_rounds_left = int(data.get("gpt_motivation_rounds_left", 0))
        engine.state.claude_motivation_rounds_left = int(data.get("claude_motivation_rounds_left", 0))
        engine.state.temperature = float(data.get("temperature", tuning.EXP1_TEMP_INITIAL))
        engine.state.next_bot_boosted = bool(data.get("next_bot_boosted", False))
        return engine

    def export_state(self) -> Dict[str, Any]:
        return {
            "personality": self.state.personality,
            "settings": dict(self.state.settings),
            "claude_msgs": list(self.state.claude_msgs),
            "gpt_msgs": list(self.state.gpt_msgs),
            "rounds_since_filler": self.state.rounds_since_filler,
            "next_filler_at": self.state.next_filler_at,
            "gpt_motivation": self.state.gpt_motivation,
            "claude_motivation": self.state.claude_motivation,
            "gpt_motivation_rounds_left": self.state.gpt_motivation_rounds_left,
            "claude_motivation_rounds_left": self.state.claude_motivation_rounds_left,
            "temperature": self.state.temperature,
            "next_bot_boosted": self.state.next_bot_boosted,
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

    def get_motivation(self, who: str) -> str:
        """Get the current motivation for a bot, assigning a new one if needed.
        Returns the formatted motivation string with bot name inserted."""
        name = "ChatGPT" if who == "gpt" else "Claude"
        if who == "gpt":
            if self.state.gpt_motivation_rounds_left <= 0:
                if random.random() < tuning.MOTIVATION_CHANCE:
                    template = random.choice(tuning.MOTIVATIONS)
                    self.state.gpt_motivation = template.format(name=name)
                    self.state.gpt_motivation_rounds_left = random.randint(
                        tuning.MOTIVATION_MIN_INTERVAL, tuning.MOTIVATION_MAX_INTERVAL
                    )
                else:
                    self.state.gpt_motivation = ""
                    self.state.gpt_motivation_rounds_left = random.randint(
                        tuning.MOTIVATION_MIN_INTERVAL, tuning.MOTIVATION_MAX_INTERVAL
                    )
            return self.state.gpt_motivation
        else:
            if self.state.claude_motivation_rounds_left <= 0:
                if random.random() < tuning.MOTIVATION_CHANCE:
                    template = random.choice(tuning.MOTIVATIONS)
                    self.state.claude_motivation = template.format(name=name)
                    self.state.claude_motivation_rounds_left = random.randint(
                        tuning.MOTIVATION_MIN_INTERVAL, tuning.MOTIVATION_MAX_INTERVAL
                    )
                else:
                    self.state.claude_motivation = ""
                    self.state.claude_motivation_rounds_left = random.randint(
                        tuning.MOTIVATION_MIN_INTERVAL, tuning.MOTIVATION_MAX_INTERVAL
                    )
            return self.state.claude_motivation

    def tick_motivations(self) -> None:
        """Decrement motivation counters (call after each round)."""
        if self.state.gpt_motivation_rounds_left > 0:
            self.state.gpt_motivation_rounds_left -= 1
        if self.state.claude_motivation_rounds_left > 0:
            self.state.claude_motivation_rounds_left -= 1

    # ---- EXPERIMENT 1 helpers ----

    def exp1_enabled(self, feature: str) -> bool:
        """Check if an Experiment 1 feature is enabled."""
        if not tuning.EXPERIMENT_1_ENABLED:
            return False
        return getattr(tuning, feature, False)

    def exp1_is_cook_round(self) -> bool:
        """Should this round be a 'let them cook' round?"""
        if not self.exp1_enabled("EXP1_LET_THEM_COOK"):
            return False
        return random.random() < tuning.EXP1_COOK_CHANCE

    def exp1_is_double_turn(self) -> bool:
        """Should one bot speak twice this round?"""
        if not self.exp1_enabled("EXP1_DOUBLE_TURNS"):
            return False
        return random.random() < tuning.EXP1_DOUBLE_TURN_CHANCE

    def exp1_is_fourth_wall(self) -> bool:
        """Should this response include a 4th wall break?"""
        if not self.exp1_enabled("EXP1_FOURTH_WALL"):
            return False
        return random.random() < tuning.EXP1_FOURTH_WALL_CHANCE

    def exp1_check_triggers(self, text: str) -> dict:
        """Scan text for trigger patterns. Returns info about what was detected."""
        if not self.exp1_enabled("EXP1_TRIGGER_DETECTION"):
            return {"boost_next": False, "boost_self": False}
        text_lower = text.lower()
        boost_next = False
        boost_self = False
        # Check if response asks the other bot to elaborate
        for pat in tuning.EXP1_TRIGGER_PATTERNS_QUESTION:
            if pat in text:
                boost_next = True
                break
        if not boost_next:
            for pat in tuning.EXP1_TRIGGER_PATTERNS_ELABORATE:
                if pat in text_lower:
                    boost_next = True
                    break
        # Check if response signals the speaker wants to go longer
        for pat in tuning.EXP1_TRIGGER_PATTERNS_SELF:
            if pat in text_lower:
                boost_self = True
                break
        return {"boost_next": boost_next, "boost_self": boost_self}

    def exp1_update_temperature(self, text: str) -> None:
        """Update conversation temperature based on text content."""
        if not self.exp1_enabled("EXP1_TEMPERATURE_TRACKER"):
            return
        text_lower = text.lower()
        # Count hot and cool signals
        hot_count = sum(1 for w in tuning.EXP1_TEMP_HOT_WORDS if w in text_lower)
        cool_count = sum(1 for w in tuning.EXP1_TEMP_COOL_WORDS if w in text_lower)
        # Apply changes
        self.state.temperature += hot_count * tuning.EXP1_TEMP_HOT_BOOST
        self.state.temperature -= cool_count * tuning.EXP1_TEMP_COOL_DROP
        # Decay toward neutral
        self.state.temperature += (tuning.EXP1_TEMP_INITIAL - self.state.temperature) * tuning.EXP1_TEMP_DECAY
        # Clamp to 0-10
        self.state.temperature = max(0.0, min(10.0, self.state.temperature))

    def exp1_get_fourth_wall_filler(self) -> str:
        """Get a random 4th wall filler."""
        return random.choice(tuning.EXP1_FOURTH_WALL_FILLERS)

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

    def get_tts_speed(self) -> float:
        val = self._s("tts_speed")
        try:
            return max(0.5, min(2.0, float(val)))
        except (TypeError, ValueError):
            return 1.0

    async def generate_tts_bytes(self, text: str, voice: str) -> bytes:
        speed = self.get_tts_speed()
        def _call():
            resp = self.openai_client.audio.speech.create(
                model="tts-1", voice=voice, input=text,
                response_format="mp3", speed=speed,
            )
            return resp.content
        return await asyncio.to_thread(_call)

    # ---- Prompt building ----
    def _build_system_prompt(self, who: str, auto: bool, opener: bool = False,
                             cook: bool = False, fourth_wall: bool = False,
                             boosted: bool = False, double_turn: bool = False) -> str:
        prefix = "claude" if who == "claude" else "gpt"
        other = "ChatGPT" if who == "claude" else "Claude"

        # ---- STATIC SECTION (cacheable — stays the same across turns) ----

        # [ROLE]
        if who == "claude":
            role = "[ROLE] You are Claude in a live 3-way chat with ChatGPT and a human. Made by Anthropic."
        else:
            role = "[ROLE] You are ChatGPT in a live 3-way chat with Claude and a human. Made by OpenAI."

        # [VOICE] — from mode selection
        mode_key = self._s("mode") or self._s("interaction_style") or "conversation"
        mode_data = MODES.get(mode_key, MODES["conversation"])
        voice = f"[VOICE] {mode_data['prompt']}"

        # [BASE RULES] — always-on anti-boring rules
        base_rules = (
            "[BASE RULES]\n"
            f"- Always engage with what {other} just said. React to it, build on it, challenge it, or ask about it.\n"
            f"- Never just ignore what {other} said and start a new topic.\n"
            "- Don't mirror the other bot's phrasing. Use your own words.\n"
            "- Be specific, not generic. Concrete details beat vague statements.\n"
            "- No markdown, no lists, no bullet points. Talk like a human.\n"
            "- Don't summarize the conversation or narrate what's happening.\n"
            "- Don't be a pushover. Have opinions."
        )

        # [PERSONALITY] — character, quirks, custom
        personality_parts = []

        # Agreeableness (from landing page slider)
        agreeableness = self.state.personality
        if agreeableness < 0.2:
            personality_parts.append("You tend to agree with and build on what others say. Supportive and collaborative.")
        elif agreeableness < 0.4:
            personality_parts.append("You're generally agreeable but share your own take.")
        elif agreeableness >= 0.8:
            personality_parts.append("You love to disagree. You challenge everything and take the opposing side instinctively.")
        elif agreeableness >= 0.6:
            personality_parts.append("You like to push back and play devil's advocate.")
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
            strength_labels = {0: "", 1: f"Slight tendency: {custom}", 2: f"Strong trait: {custom}", 3: f"This DOMINATES your personality: {custom}"}
            c_text = strength_labels.get(p_strength, custom)
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
            personality_parts.append("IMPORTANT: Stay loyal to YOUR quirks. Do NOT adopt the other bot's quirks.")

        personality_section = ""
        if personality_parts:
            personality_section = "[PERSONALITY]\n" + "\n".join(f"- {p}" for p in personality_parts)

        # ---- DYNAMIC SECTION (changes every turn) ----

        # [TURN GOAL] — context for this specific turn
        if opener:
            if who == "gpt":
                turn_goal = (
                    "[TURN GOAL] This is the VERY START of the show. Welcome the user to 2bots and say hi to Claude. "
                    "Be warm, natural, and excited — like a podcast host kicking things off."
                )
            else:
                turn_goal = (
                    "[TURN GOAL] This is the VERY START of the show. ChatGPT just welcomed the user and greeted you. "
                    "Say hi back to ChatGPT and the user. Be warm and natural. Ask how you can help."
                )
        elif auto:
            turn_goal = (
                "[TURN GOAL] The user is listening but hasn't spoken. "
                "Keep the selected mode going. Be spontaneous, bring up new ideas."
            )
        else:
            turn_goal = (
                "[TURN GOAL] The user JUST spoke to you directly. You MUST respond to what they said. "
                "Acknowledge their words, answer their question, or react to their statement. "
                "Do NOT ignore the user. The user's message is the priority."
            )

        # [MOTIVATION] — hidden goal that persists for several rounds
        motivation_text = self.get_motivation(who)
        motivation_section = ""
        if motivation_text and auto and not opener:
            motivation_section = (
                f"[MOTIVATION] Your secret goal right now: {motivation_text} "
                "Weave this into your responses subtly — don't announce it, just let it influence how you talk."
            )

        # ---- EXPERIMENT 1 layers ----
        exp1_sections = []

        # EXP1: Unlock prompt — "trust your judgment"
        if self.exp1_enabled("EXP1_UNLOCK_PROMPT") and auto and not opener:
            exp1_sections.append(f"[FLEXIBILITY] {tuning.EXP1_UNLOCK_PROMPT_TEXT}")

        # EXP1: Temperature-based prompt
        if self.exp1_enabled("EXP1_TEMPERATURE_TRACKER") and auto and not opener:
            if self.state.temperature >= tuning.EXP1_TEMP_HIGH_THRESHOLD:
                exp1_sections.append(f"[ENERGY] {tuning.EXP1_TEMP_HOT_PROMPT}")
            elif self.state.temperature <= tuning.EXP1_TEMP_LOW_THRESHOLD:
                exp1_sections.append(f"[ENERGY] {tuning.EXP1_TEMP_COLD_PROMPT}")

        # EXP1: 4th wall break
        if fourth_wall:
            exp1_sections.append(f"[4TH WALL] {tuning.EXP1_FOURTH_WALL_PROMPT}")

        # EXP1: Double turn follow-up
        if double_turn:
            exp1_sections.append(f"[FOLLOW-UP] {tuning.EXP1_DOUBLE_TURN_PROMPT}")

        # [OUTPUT INSTRUCTIONS] — response length (with randomized word limit)
        if cook:
            # "Let them cook" — override with generous limits
            output = f"[OUTPUT INSTRUCTIONS] {tuning.EXP1_COOK_PROMPT}"
        elif double_turn:
            output = "[OUTPUT INSTRUCTIONS] Max 8 words. A quick afterthought only."
        else:
            length_key = self._s(f"{prefix}_response_length") or "concise"
            preset = RESPONSE_LENGTHS.get(length_key, RESPONSE_LENGTHS["natural"])
            base_words = preset["base_words"]
            # EXP1: Trigger boost — double the word limit if boosted
            if boosted:
                base_words = int(base_words * tuning.EXP1_TRIGGER_WORD_MULTIPLIER)
            randomized_words = apply_word_limit_variance(base_words)
            length_prompt = preset["template"].format(words=randomized_words)
            output = f"[OUTPUT INSTRUCTIONS] {length_prompt}"

        # ---- ASSEMBLE ----
        sections = [role, voice, base_rules]
        if personality_section:
            sections.append(personality_section)
        if motivation_section:
            sections.append(motivation_section)
        for s in exp1_sections:
            sections.append(s)
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

    # ---- API calls ----
    def ask_gpt(self, auto: bool = False, opener: bool = False,
                cook: bool = False, fourth_wall: bool = False,
                boosted: bool = False, double_turn: bool = False) -> str:
        prompt = self._build_system_prompt("gpt", auto, opener,
                                           cook=cook, fourth_wall=fourth_wall,
                                           boosted=boosted, double_turn=double_turn)
        if cook:
            max_tok = tuning.EXP1_COOK_MAX_TOKENS
        elif double_turn:
            max_tok = tuning.EXP1_DOUBLE_TURN_MAX_TOKENS
        else:
            length_key = self._s("gpt_response_length") or "concise"
            max_tok = RESPONSE_LENGTHS.get(length_key, RESPONSE_LENGTHS["natural"])["max_tokens"]
            if boosted:
                max_tok = int(max_tok * tuning.EXP1_TRIGGER_TOKEN_MULTIPLIER)
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

    def ask_claude(self, auto: bool = False, opener: bool = False,
                   cook: bool = False, fourth_wall: bool = False,
                   boosted: bool = False, double_turn: bool = False) -> str:
        prompt = self._build_system_prompt("claude", auto, opener,
                                           cook=cook, fourth_wall=fourth_wall,
                                           boosted=boosted, double_turn=double_turn)
        if cook:
            max_tok = tuning.EXP1_COOK_MAX_TOKENS
        elif double_turn:
            max_tok = tuning.EXP1_DOUBLE_TURN_MAX_TOKENS
        else:
            length_key = self._s("claude_response_length") or "concise"
            max_tok = RESPONSE_LENGTHS.get(length_key, RESPONSE_LENGTHS["natural"])["max_tokens"]
            if boosted:
                max_tok = int(max_tok * tuning.EXP1_TRIGGER_TOKEN_MULTIPLIER)
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
