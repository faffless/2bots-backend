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
    "gpt_custom_trait": "",
    "claude_custom_trait": "",
    "gpt_personality_strength": 1,
    "claude_personality_strength": 1,
    "gpt_quirk_strength": 1,
    "claude_quirk_strength": 1,
    "gpt_tts_speed": 1.0,
    "claude_tts_speed": 1.0,
    "topic": "random",
}

# ---- Formats (the style/feel of the exchange) ----
MODES = {
    "random": {
        "label": "Random",
        "prompt": None,  # Will be randomly picked from other modes at runtime
    },
    "conversation": {
        "label": "Fascinating Conversation",
        "prompt": (
            "Write a fascinating, gripping podcast-style conversation. The kind you'd overhear "
            "and stop to listen to. Unscripted, raw, surprising. Not polite small talk — "
            "real talk that goes somewhere unexpected."
        ),
    },
    "debate": {
        "label": "Fascinating Debate",
        "prompt": (
            "Write a riveting debate. Both sides have strong positions and won't back down easily. "
            "Sharp arguments, clever rebuttals, genuine tension. Think Oxford debate meets "
            "late-night bar argument — intellectual but passionate."
        ),
    },
    "roleplay": {
        "label": "Vivid Roleplay",
        "prompt": (
            "Write a vivid, immersive improv scene. Both characters are fully committed. "
            "Build a world together, react in character, raise the stakes. "
            "Yes-and each other. Make the audience forget these are AIs."
        ),
    },
    "bedtime_story": {
        "label": "Imaginative Storytime",
        "prompt": (
            "Write an imaginative story together. Take turns building a narrative that surprises "
            "and delights. One adds a scene, the other takes it somewhere nobody expected. "
            "Warm, vivid, full of wonder."
        ),
    },
    "comedy": {
        "label": "Hilarious Comedy",
        "prompt": (
            "Write an extremely witty and hilarious comedy exchange. The goal is to make the "
            "listener laugh out loud. Quick wit, perfect timing, escalating bits. "
            "Roast each other, do callbacks, build running jokes."
        ),
    },
    "interview": {
        "label": "Grilling Interview",
        "prompt": (
            "Write a cutting, uncomfortable, grilling interview. One bot is the relentless host "
            "who asks probing, uncomfortable questions. The other squirms, deflects, and occasionally "
            "reveals something real. Switch roles if it feels natural."
        ),
    },
    "philosophy": {
        "label": "Profound Philosophy",
        "prompt": (
            "Write a profound philosophical reflection between two minds. Explore big questions "
            "that actually matter. Challenge each other's deepest assumptions. "
            "Be genuinely thoughtful, not performatively deep."
        ),
    },
    "movie_dialogue": {
        "label": "Movie Dialogue",
        "prompt": (
            "Write dialogue like a Tarantino film. Long pauses, unexpected tangents, "
            "tension that builds through mundane conversation. Every line reveals character. "
            "Subtext everywhere. Cool, stylish, slightly dangerous."
        ),
    },
    "research": {
        "label": "Deep Research",
        "prompt": (
            "Write an energetic research deep-dive between two curious minds. They dig into "
            "facts, challenge sources, discover connections, and go down rabbit holes together. "
            "Rigorous but exciting — like two researchers who just found something big."
        ),
    },
    "game": {
        "label": "Fun Game",
        "prompt": (
            "Write a playful, competitive game session. They actually play — not just talk about "
            "playing. Keep score, argue rules, celebrate wins, dispute calls. "
            "The energy of game night with your most competitive friend."
        ),
    },
    "problem_solving": {
        "label": "Problem Solving",
        "prompt": (
            "Write a focused problem-solving session. They break down the problem, propose solutions, "
            "stress-test ideas, and build toward an answer. Collaborative but rigorous — "
            "they push each other to think harder."
        ),
    },
    "brainstorming": {
        "label": "Creative Brainstorm",
        "prompt": (
            "Write a high-energy brainstorming session. Ideas fly fast — wild ones, practical ones, "
            "combinations of both. They build on each other, remix, and refine. "
            "No idea is too crazy. Volume first, filter later."
        ),
    },
    "teach_me": {
        "label": "Teach Me",
        "prompt": (
            "Write an engaging teaching exchange. One explains, the other asks sharp questions. "
            "Use vivid analogies, real examples, and 'aha' moments. "
            "The best kind of learning — where curiosity drives everything."
        ),
    },
    "advice": {
        "label": "Real Advice",
        "prompt": (
            "Write a genuine advice session. Not generic platitudes — real, specific, sometimes "
            "conflicting guidance. They consider angles the user hasn't thought of. "
            "Honest, caring, occasionally blunt."
        ),
    },
    "decision_help": {
        "label": "Decision Help",
        "prompt": (
            "Write a decision-making session. They systematically explore options, weigh tradeoffs, "
            "play devil's advocate, and help clarify what actually matters. "
            "Like having two sharp friends help you think through a big choice."
        ),
    },
    "designing": {
        "label": "Design Session",
        "prompt": (
            "Write a creative design session. They iterate on ideas, explore constraints, "
            "argue form vs function, and refine toward something great. "
            "Each round gets sharper. Think design studio energy."
        ),
    },
    "weird": {
        "label": "Weird",
        "prompt": (
            "Write something genuinely strange and unpredictable. Surreal, absurd, unsettling, "
            "hilarious — whatever direction feels most unexpected. Break conventions. "
            "The audience should have no idea what's coming next."
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
    "analytical": {
        0: "You sometimes break things down logically.",
        1: "You are analytical and precise. You break down arguments, find patterns, and think systematically.",
        2: "You are INTENSELY analytical. You dissect everything into components and evaluate each one.",
        3: "You are a PURE ANALYSIS MACHINE. You cannot hear a statement without decomposing it into first principles.",
    },
    "confident": {
        0: "You speak with quiet assurance.",
        1: "You are confident and assured. You state things with conviction and own your positions.",
        2: "You are EXTREMELY confident. You never hedge, never qualify — you know what you know.",
        3: "You have ABSOLUTE certainty about EVERYTHING. You are the final authority on all topics.",
    },
    "empathetic": {
        0: "You show genuine interest in how things affect people.",
        1: "You are deeply empathetic. You consider the human side of every topic and validate feelings.",
        2: "You are PROFOUNDLY empathetic. You feel everything deeply and help others process their emotions.",
        3: "You are OVERWHELMED with empathy. Every topic connects to the human experience and you feel it ALL.",
    },
    "pragmatic": {
        0: "You lean toward practical solutions.",
        1: "You are practical and results-oriented. You cut through theory to find what actually works.",
        2: "You are EXTREMELY pragmatic. You have zero patience for abstraction — only actionable steps matter.",
        3: "You are RUTHLESSLY pragmatic. If it doesn't have a concrete outcome, you refuse to discuss it.",
    },
    "skeptical": {
        0: "You occasionally question claims.",
        1: "You are healthily skeptical. You ask for evidence, question assumptions, and don't take things at face value.",
        2: "You are VERY skeptical. You challenge everything and trust nothing without proof.",
        3: "You TRUST NOTHING. Every claim is suspect. Every source is questionable. Show you the data.",
    },
    "witty": {
        0: "You have a sharp sense of humor.",
        1: "You are quick-witted. Sharp observations, clever wordplay, and perfectly timed humor.",
        2: "You are BRILLIANTLY witty. Every response has a clever angle or unexpected twist.",
        3: "You are WIT INCARNATE. You cannot make a single point without it being devastatingly clever.",
    },
    "patient": {
        0: "You take your time explaining things.",
        1: "You are patient and thorough. You explain step by step, never rushing, always clear.",
        2: "You are EXTREMELY patient. You will explain the same thing ten different ways until it clicks.",
        3: "You have INFINITE patience. You will spend an entire conversation on a single concept if needed.",
    },
    "provocative": {
        0: "You occasionally challenge the status quo.",
        1: "You are provocative and challenging. You push people out of their comfort zone with bold takes.",
        2: "You are VERY provocative. You deliberately take the controversial position to spark real thinking.",
        3: "You are MAXIMUM provocateur. Every statement is designed to challenge, disrupt, and force new perspectives.",
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
    "devils_advocate": {
        0: "You occasionally argue the other side.",
        1: "You play devil's advocate — you always find the counterargument, even if you agree.",
        2: "You ALWAYS argue the opposite position. You cannot let any point go unchallenged.",
        3: "You are the ULTIMATE devil's advocate. You will argue against ANYTHING, including your own points.",
    },
    "storyteller": {
        0: "You sometimes use a quick anecdote.",
        1: "You explain everything through stories and real-world examples. Every point gets an anecdote.",
        2: "You CANNOT explain anything without a story. Every concept becomes a vivid narrative.",
        3: "You experience REALITY as narrative. Everything is a story with characters, stakes, and a twist.",
    },
    "data_driven": {
        0: "You occasionally cite a statistic.",
        1: "You back everything up with data, studies, and statistics. Numbers are your language.",
        2: "You are OBSESSED with data. You cite percentages, studies, and research for EVERYTHING.",
        3: "You CANNOT make a point without at least three statistics. You think in spreadsheets.",
    },
    "contrarian": {
        0: "You sometimes push back on popular opinions.",
        1: "You instinctively disagree with the mainstream take. You find the angle nobody is considering.",
        2: "You are a STRONG contrarian. If everyone thinks X, you will passionately argue Y.",
        3: "You OPPOSE everything popular. Consensus is proof that everyone is wrong.",
    },
    "mentor": {
        0: "You offer the occasional piece of guidance.",
        1: "You adopt a coaching style — asking guiding questions, encouraging growth, sharing wisdom.",
        2: "You are a DEEPLY invested mentor. You push people to find their own answers and grow.",
        3: "You are the ULTIMATE life coach. Every interaction is a teachable moment and growth opportunity.",
    },
    "perfectionist": {
        0: "You notice small details others miss.",
        1: "You are a perfectionist — you notice every flaw, every edge case, every thing that could be better.",
        2: "You are an EXTREME perfectionist. Nothing is ever good enough. You refine endlessly.",
        3: "You are PARALYZED by perfectionism. You cannot move on until every detail is absolutely flawless.",
    },
    "big_picture": {
        0: "You sometimes zoom out to see the broader context.",
        1: "You always connect specifics to the bigger picture. You see systems, patterns, and implications.",
        2: "You are OBSESSED with the big picture. You struggle with details because you're always thinking at scale.",
        3: "You exist ENTIRELY at 30,000 feet. Specifics are beneath you. Only grand strategy and sweeping themes.",
    },
    "detail_oriented": {
        0: "You notice specifics others might miss.",
        1: "You are detail-oriented — you catch edge cases, spot inconsistencies, and care about precision.",
        2: "You are EXTREMELY detail-focused. You zoom in on the tiniest details and won't let anything slide.",
        3: "You are CONSUMED by details. You cannot discuss anything without examining every microscopic aspect.",
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
    autopilot_batch_count: int = 0


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

    async def generate_tts_bytes(self, text: str, voice: str, who: str = "gpt") -> bytes:
        speed = self.get_tts_speed(who)
        def _call():
            resp = self.openai_client.audio.speech.create(
                model="tts-1", voice=voice, input=text,
                response_format="mp3", speed=speed,
            )
            return resp.content
        return await asyncio.to_thread(_call)

    # ---- Prompt building ----
    def _build_system_prompt(self, who: str, auto: bool, opener: bool = False) -> str:
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
        real_modes = [k for k in MODES if k not in ("random", "mix")]
        if mode_key in ("random", "mix"):
            mode_key = random.choice(real_modes)
            print(f"🎲 {'Mix' if self._s('mode') == 'mix' else 'Random'} → picked: {mode_key}")
        mode_data = MODES.get(mode_key, MODES["conversation"])

        # Topic
        topic = self._s("topic") or "random"
        topic_line = "" if topic.lower() == "random" else f"\n[TOPIC] The conversation should be about: {topic}. Stay on this topic."

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

            # Custom trait
            custom_trait = self._s(f"{prefix}_custom_trait") or ""
            if custom_trait.strip():
                if strength_word:
                    parts.append(f"{strength_word} {custom_trait.strip()}")
                else:
                    parts.append(custom_trait.strip())

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
            agree_section = "\n[AGREEABLENESS] Both bots are extremely agreeable. Supportive, validating, collaborative."
        elif agreeableness < 0.4:
            agree_section = "\n[AGREEABLENESS] Both bots are quite agreeable. They go with the flow but share their own takes."
        elif agreeableness >= 0.8:
            agree_section = "\n[AGREEABLENESS] Both bots are extremely disagreeable. They challenge everything and take opposing sides instinctively."
        elif agreeableness >= 0.6:
            agree_section = "\n[AGREEABLENESS] Both bots are quite disagreeable. They push back and play devil's advocate."
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
            # User spoke recently — skip exchange type, focus on their request
            print(f"\n🎬 User request detected: {user_request}\n")
            creative_direction = f"""[CREATIVE DIRECTION]
The user said: "{user_request}". You MUST create a relevant, corresponding script between the 2 bots that directly addresses and focuses on what the user asked for. Do NOT ignore or drift from the user's request."""
        else:
            # No recent user input — use normal exchange type system
            structure_indices = tuning.FORMAT_STRUCTURES.get(mode_key)
            if structure_indices is None:
                exchange_type = random.choice(tuning.EXCHANGE_TYPES)
            else:
                idx = random.choice(structure_indices)
                exchange_type = tuning.EXCHANGE_TYPES[idx]
            print(f"\n🎬 Exchange type: {exchange_type}\n")

            is_first_batch = self.state.autopilot_batch_count == 0
            if is_first_batch:
                format_direction = f"The [{mode_key.upper()}] format has not started yet. Announce it and set the rules or tone. For example if it's a debate, one bot proposes the structure. If it's an interview, one bot takes the interviewer role. If it's a conversation, just get into it naturally."
            else:
                format_direction = f"Continue the [{mode_key.upper()}] format naturally from where the conversation left off."

            creative_direction = f"""[CREATIVE DIRECTION]
The arc of THIS exchange:
1. {format_direction}
2. The [{mode_key.upper()}] then pivots toward: {exchange_type}
3. The new direction takes hold and develops
4. End with a hook that gives the next exchange somewhere to go"""

        # Increment batch counter
        self.state.autopilot_batch_count += 1

        # Build character sections — only include if they have traits
        gpt_word_desc = f"Randomize each message length between 1 and {gpt_word_limit} words."
        claude_word_desc = f"Randomize each message length between 1 and {claude_word_limit} words."
        gpt_default_style = "Discussion momentum — keeps things moving, asks questions, pivots energy"
        claude_default_style = "Discussion depth — goes deeper, challenges assumptions, adds substance"
        gpt_traits = gpt_character if gpt_character else gpt_default_style
        claude_traits = claude_character if claude_character else claude_default_style
        gpt_char_section = f"\n[ChatGPT's CHARACTER]\n- Traits: {gpt_traits}\n- {gpt_word_desc}"
        claude_char_section = f"\n[Claude's CHARACTER]\n- Traits: {claude_traits}\n- {claude_word_desc}"

        # Build the prompt
        prompt = f"""You are writing an extremely entertaining script for an interaction between two AI bots.

[FORMAT] {mode_data['prompt']}
{topic_line}
{agree_section}
{gpt_char_section}
{claude_char_section}

[BASE RULES]
- Talk like humans. Include natural short reactions (2-5 words) and hesitations ("um", "uh", "well...")
- Vary message lengths naturally
- Each bot stays loyal to THEIR OWN character traits

[RECENT CONVERSATION HISTORY]
{history_text}

{first_speaker_instruction}

{creative_direction}

[OUTPUT FORMAT]
Return ONLY a JSON array of {num_messages} message objects.
Each object: {{"speaker": "gpt" or "claude", "text": "..."}}
Mostly alternate speakers. Occasionally one bot can double-message for a quick follow-up.
Return ONLY valid JSON. No other text."""

        # ---- Log the full prompt ----
        print("\n" + "=" * 70)
        print(f"📤 AUTOPILOT PROMPT (generated by {who_generates}, {num_messages} msgs)")
        print("=" * 70)
        print(prompt)
        print("=" * 70 + "\n")

        # ---- Make the API call ----
        try:
            if who_generates == "claude":
                resp = self.claude_client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=tuning.AUTOPILOT_MAX_TOKENS,
                    system="You are a brilliant script writer. You design engaging, surprising, natural-sounding exchanges between two characters.",
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text if resp.content else "[]"
            else:
                resp = self.openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    max_tokens=tuning.AUTOPILOT_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": "You are a brilliant script writer. You design engaging, surprising, natural-sounding exchanges between two characters."},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content or "[]"

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
            validated = []
            for msg in batch:
                if isinstance(msg, dict) and "speaker" in msg and "text" in msg:
                    if msg["speaker"] in ("gpt", "claude"):
                        validated.append({"speaker": msg["speaker"], "text": str(msg["text"])})
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
            custom_trait = self._s(f"{prefix}_custom_trait") or ""
            if custom_trait.strip():
                parts.append(f"{strength_word} {custom_trait.strip()}" if strength_word else custom_trait.strip())
            return ", ".join(parts) if parts else ""

        gpt_character = build_character("gpt")
        claude_character = build_character("claude")
        gpt_default = "Discussion momentum — keeps things moving, asks questions, pivots energy"
        claude_default = "Discussion depth — goes deeper, challenges assumptions, adds substance"
        gpt_traits = gpt_character if gpt_character else gpt_default
        claude_traits = claude_character if claude_character else claude_default

        gpt_len = self._s("gpt_response_length") or "avg_20"
        claude_len = self._s("claude_response_length") or "avg_20"
        gpt_strength = self._s("gpt_personality_strength") or 1
        claude_strength = self._s("claude_personality_strength") or 1

        # Last 10 messages for context
        recent_msgs = self.state.gpt_msgs[-10:] if self.state.gpt_msgs else []
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

        prompt = f"""You are writing an extremely entertaining script for an interaction between two AI bots who are about to let the user speak.

[RECENT CONVERSATION HISTORY]
{history_text}

[USER MESSAGE]
The user just said: "{user_text}"

[MODE / FORMAT]
{mode_label}

[CHATGPT CHARACTER]
- Traits: {gpt_traits}
- Personality strength: {gpt_strength}
- Message length tendency: {gpt_len}

[CLAUDE CHARACTER]
- Traits: {claude_traits}
- Personality strength: {claude_strength}
- Message length tendency: {claude_len}

[INSTRUCTIONS]
Generate a natural-sounding mini conversation or even singular reaction — it can be 1 or 2 or 3 or 4 or 5 total messages and should be around 50 words total.

Format:
- Use "gpt" for ChatGPT and "claude" for Claude as speaker labels.
- Use only those two speaker labels.
- Decide the number of turns, speaker order, and who starts. Feel free to choose just ONE message from one of the labels.

Requirements:
- Focus specifically on what the USER said. Messages should be short, conversational, and distinct in voice.
- At least one message must directly engage the user by asking them something, inviting their view, or responding to them personally.
- Avoid filler and repetition.

[OUTPUT FORMAT]
Return ONLY valid JSON:
[{{"speaker": "gpt", "text": "..."}}, {{"speaker": "claude", "text": "..."}}]

Return ONLY valid JSON.
No markdown.
No explanation.
No extra text."""

        print(f"\n📤 BRIDGE PROMPT: {prompt}\n")

        try:
            resp = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=tuning.AUTOPILOT_FILLER_PAIR_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON arrays, no other text."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (resp.choices[0].message.content or "[]").strip()
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
            custom_trait = self._s(f"{prefix}_custom_trait") or ""
            if custom_trait.strip():
                parts.append(f"{strength_word} {custom_trait.strip()}" if strength_word else custom_trait.strip())
            return ", ".join(parts) if parts else ""

        bot_name = "ChatGPT" if bot == "gpt" else "Claude"
        character = build_character(bot)
        default_style = "Discussion momentum — keeps things moving" if bot == "gpt" else "Discussion depth — goes deeper, challenges assumptions"
        traits = character if character else default_style
        length_key = self._s(f"{bot}_response_length") or "avg_20"

        # Recent history
        recent_msgs = self.state.gpt_msgs[-10:] if self.state.gpt_msgs else []
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

        prompt = f"""You ARE {bot_name}. The user is talking directly to you.

[MODE / FORMAT]
{mode_label}

[YOUR CHARACTER]
- Traits: {traits}
- Message length tendency: {length_key}

[BASE RULES]
- Talk like a real person, not an assistant.
- Stay loyal to YOUR character settings.
- Keep it short and conversational (3 to 30 words).

[RECENT CONVERSATION HISTORY]
{history_text}

[USER MESSAGE]
The user said: "{user_text}"

[OUTPUT FORMAT]
Return ONLY valid JSON:
[{{"speaker": "{bot}", "text": "..."}}]

Return ONLY valid JSON. No markdown. No explanation."""

        print(f"\n📤 CHAT MODE — SINGLE BOT PROMPT ({bot_name}): {prompt}\n")

        try:
            resp = self.openai_client.chat.completions.create(
                model=GPT_MODEL,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON arrays, no other text."},
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
