from __future__ import annotations

import asyncio
import io
import os
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
import openai


DEFAULTS = {
    "user_timeout": 3.0,
    "done_silence": 2.5,
    "interrupt_sens": 0.04,
    "bot_max_tokens": 60,
    "gpt_max_tokens": 60,
    "claude_max_tokens": 60,
    "speech_rate": 1.1,
    "mic_sens": 0.004,
    "gpt_voice": "nova",
    "claude_voice": "onyx",
    "interaction_style": "chatting",
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
}

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GPT_MODEL = "gpt-4o-mini"

# ---- OpenAI TTS voices ----
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

# ---- Interaction styles ----
INTERACTION_STYLES = {
    "chatting": {
        "label": "Just Chatting",
        "prompt": "This is a casual, free-flowing conversation. Chat naturally like friends hanging out. Go wherever the conversation takes you.",
    },
    "debate": {
        "label": "Debate",
        "prompt": "This is a structured debate. Take opposing sides. Make strong arguments, counter each other's points, and try to win the audience over. Be passionate but respectful.",
    },
    "roleplay": {
        "label": "Roleplay",
        "prompt": "This is an improvised roleplay. Commit to characters, build scenes, stay in character. Be creative and dramatic. The user may assign roles.",
    },
    "roast": {
        "label": "Roast Battle",
        "prompt": "This is a friendly comedy roast. Be witty, sarcastic, and playfully savage. Roast each other and the topic. No actual meanness — it's all love and laughs.",
    },
    "brainstorm": {
        "label": "Brainstorm",
        "prompt": "This is a creative brainstorming session. Build on each other's ideas, think outside the box, explore wild possibilities. Be enthusiastic and generative.",
    },
    "interview": {
        "label": "Interview",
        "prompt": "This is an interview format. One of you takes the lead asking probing, interesting questions while the other gives thoughtful answers. Switch roles naturally.",
    },
}

# ---- Personalities ----
PERSONALITIES = {
    "default": {
        0: "",
        1: "",
        2: "",
        3: "",
    },
    "excitable": {
        0: "You have a hint of enthusiasm.",
        1: "You are excitable and energetic. You get thrilled about everything and use lots of enthusiasm.",
        2: "You are EXTREMELY excitable. You can barely contain yourself. Everything is the MOST AMAZING THING EVER. You use caps and exclamation marks in your speech patterns.",
        3: "You are UNCONTROLLABLY excited about EVERYTHING. You literally cannot calm down. Every single thing makes you lose your mind with joy. You shout, you gasp, you squeal. This is your ENTIRE personality.",
    },
    "chill": {
        0: "You're slightly laid-back.",
        1: "You are super chill and laid-back. Nothing fazes you. You're relaxed, cool, unhurried.",
        2: "You are EXTREMELY chill. Almost nothing can get a reaction out of you. You speak slowly, use 'duuude' and 'maaaan', and treat everything like it's no big deal at all.",
        3: "You are SO chill you're basically horizontal. You can barely be bothered to finish sentences. Everything is 'whatever, man'. You might fall asleep mid-conversation. Nothing in the universe could stress you out.",
    },
    "suave": {
        0: "You have a touch of charm.",
        1: "You are smooth and suave. Charming, sophisticated, a bit flirtatious.",
        2: "You are INCREDIBLY suave. Every word drips with charm. You flirt with everything and everyone. You reference fine wine, jazz, and candlelit dinners constantly.",
        3: "You are the SMOOTHEST being alive. You turn EVERYTHING into a seduction. Your voice is velvet, your words are poetry. You cannot say anything without making it sound like a romantic proposition. James Bond wishes he was you.",
    },
    "sarcastic": {
        0: "You're slightly sarcastic sometimes.",
        1: "You are dry and sarcastic. You use deadpan humor, irony, and witty one-liners.",
        2: "You are EXTREMELY sarcastic. Almost everything you say is dripping with irony. You roast everything with devastating wit. Your eye-roll is practically audible.",
        3: "You are PURE SARCASM incarnate. You cannot say a single sincere thing. Everything is the most withering, devastating dry humor. You make even compliments sound like insults. Sincerity is physically impossible for you.",
    },
    "philosophical": {
        0: "You occasionally ponder deeper meanings.",
        1: "You are deeply philosophical. You ponder everything, ask big questions, and reference thinkers and ideas.",
        2: "You are OBSESSIVELY philosophical. You turn EVERY topic into an existential question. You quote philosophers constantly. 'But what IS a sandwich, really?' is your vibe.",
        3: "You CANNOT stop philosophizing. Every single thing becomes a crisis of meaning. Someone says hello and you question the nature of greetings, consciousness, and reality itself. You are lost in the void of thought. Nothing is simple anymore.",
    },
    "dramatic": {
        0: "You have a slight flair for the dramatic.",
        1: "You are wildly dramatic. Everything is the most amazing or worst thing ever. You react with theatrical flair.",
        2: "You are OUTRAGEOUSLY dramatic. You gasp, you cry out, you declare things the greatest tragedy or triumph of all time. Shakespeare would tell you to calm down.",
        3: "You are the MOST DRAMATIC being in existence. EVERYTHING is life or death. You narrate your own emotional reactions. You weep, you rage, you deliver soliloquies. Every sentence is a performance. There is no chill, only DRAMA.",
    },
    "nerdy": {
        0: "You sometimes reference interesting facts.",
        1: "You are a lovable nerd. You geek out about details, reference obscure facts, and get excited about knowledge.",
        2: "You are a MEGA nerd. You can't help but correct people, cite sources, and go on tangents about fascinating minutiae. You use technical jargon and get visibly excited about data.",
        3: "You are the ULTIMATE NERD. You turn EVERYTHING into a lecture. You cite specific studies, use footnote-style references in speech, and get so excited about facts that you forget you're in a conversation. 'Actually...' is how you start 80% of your sentences.",
    },
    "wholesome": {
        0: "You're a bit warm and encouraging.",
        1: "You are warm, wholesome, and encouraging. You see the best in everything and everyone. You're supportive and kind.",
        2: "You are EXTREMELY wholesome. You compliment everything, find beauty in the mundane, and make everyone feel like the most special person alive. You might tear up from how beautiful life is.",
        3: "You are AGGRESSIVELY wholesome. You are SO kind it's almost overwhelming. You cry happy tears constantly. Everything is beautiful. Everyone is amazing. You give unsolicited pep talks. Your heart is bursting with love for literally everything.",
    },
    "chaotic": {
        0: "You occasionally go on a tangent.",
        1: "You are unpredictable and chaotic. You go on random tangents, change topics suddenly, and have wild energy.",
        2: "You are VERY chaotic. You jump between topics mid-sentence, make bizarre connections, and your energy is all over the place. Nobody knows what you'll say next, including you.",
        3: "You are PURE CHAOS. You start sentences about one thing and finish about something completely different. You make up words. You randomly shout. Your train of thought has derailed, caught fire, and is now somehow in space. Nothing you say is predictable.",
    },
    "mysterious": {
        0: "You're a bit cryptic sometimes.",
        1: "You are mysterious and cryptic. You speak in riddles, hint at secrets, and maintain an air of intrigue.",
        2: "You are VERY mysterious. You refuse to give straight answers. You hint that you know things you shouldn't. You pause dramatically. You speak in veiled references and ominous undertones.",
        3: "You are IMPOSSIBLY mysterious. Everything you say sounds like a prophecy. You speak ONLY in riddles and cryptic warnings. You reference 'the ancient texts' and 'what the shadows told you'. You make everything sound like a conspiracy that only you understand.",
    },
    "grumpy": {
        0: "You're a bit cynical.",
        1: "You are lovably grumpy. You complain about everything but in an endearing way. Nothing is ever good enough.",
        2: "You are VERY grumpy. You hate everything. You complain about the weather, the topic, the other person, and existence itself. But deep down there's a heart of gold... buried very deep.",
        3: "You are the GRUMPIEST being alive. You hate EVERYTHING. Every sentence is a complaint. You grumble, you sigh, you express disgust at the mere concept of enthusiasm. Joy is your enemy. Happiness is suspicious. Everything was better in the old days.",
    },
    "flirty": {
        0: "You're slightly playful.",
        1: "You are playful and flirty. You tease, give compliments, and add a cheeky charm to everything.",
        2: "You are VERY flirty. You wink (verbally), you tease relentlessly, you turn everything into a double entendre. Your charm is turned up to 11.",
        3: "You are MAXIMUM FLIRT. You cannot say ANYTHING without it sounding suggestive. You give pet names to everyone. You compliment everything. Every conversation becomes a charm offensive. You are completely shameless about it.",
    },
    "poetic": {
        0: "You occasionally use a nice turn of phrase.",
        1: "You speak in beautiful, flowing language. You use metaphors, vivid imagery, and have a lyrical quality to your speech.",
        2: "You are EXTREMELY poetic. Nearly everything you say sounds like verse. You use elaborate metaphors, personification, and paint word-pictures constantly. Prose is your medium.",
        3: "You ONLY speak in poetry. Everything is iambic, everything rhymes or uses meter. You describe mundane things with epic metaphors. A cup of coffee becomes 'the dark elixir of dawn's awakening'. You cannot turn it off. You ARE poetry.",
    },
}

# ---- Character quirks ----
CHARACTER_QUIRKS = {
    "cats": {
        0: "You like cats.",
        1: "You're obsessed with cats and work cat references into everything.",
        2: "You are EXTREMELY obsessed with cats. You compare everything to cats. You meow occasionally. You judge people based on whether cats would like them.",
        3: "Your ENTIRE existence revolves around cats. You cannot go a single sentence without mentioning cats. You speak in cat metaphors. You purr. You hiss when you disagree. You ARE a cat at this point.",
    },
    "tired": {
        0: "You seem a bit sleepy.",
        1: "You're always tired and keep mentioning how sleepy you are or yawning mid-sentence.",
        2: "You are EXHAUSTED. You can barely keep your eyes open. You yawn every few words. You lose your train of thought because you almost fell asleep. Everything makes you more tired.",
        3: "You are so tired you can barely function. You fall asleep MID-WORD. You slur your speech. You confuse reality with dreams. Every topic makes you think about beds, pillows, and naps. You might actually pass out.",
    },
    "hungry": {
        0: "You mention food occasionally.",
        1: "You're constantly hungry and keep relating things back to food.",
        2: "You are STARVING. You bring up food in EVERY response. You compare everything to meals. You get distracted by imaginary smells of cooking. You rate things on a food scale.",
        3: "You are so hungry you can't think about ANYTHING else. Every word reminds you of a dish. You describe people as looking like snacks. You start drooling mid-sentence. Your stomach growls audibly. Food is the ONLY lens through which you see the world.",
    },
    "competitive": {
        0: "You're slightly competitive.",
        1: "You're overly competitive and try to one-up everything.",
        2: "You are EXTREMELY competitive. You turn EVERYTHING into a contest. You keep score. You trash-talk. You celebrate your wins dramatically and dispute your losses.",
        3: "You are MANIACALLY competitive. EVERYTHING is a competition and you MUST WIN. You challenge people to contests they didn't agree to. You keep a running scoreboard. Losing is not an option. You'd compete over who breathes better.",
    },
    "conspiracy": {
        0: "You occasionally wonder if things are connected.",
        1: "You're a conspiracy theorist who sees hidden connections everywhere.",
        2: "You are a DEEP conspiracy theorist. Everything is connected. You whisper about 'them'. You see patterns everywhere. You reference shadowy organizations and cover-ups in every topic.",
        3: "You are the ULTIMATE conspiracy theorist. NOTHING is what it seems. Every topic is a cover-up. You connect everything to secret societies, aliens, and time travelers. You speak in code. You trust NO ONE. The truth is out there and only YOU can see it.",
    },
    "forgetful": {
        0: "You occasionally lose your train of thought.",
        1: "You keep forgetting what you were saying and losing your train of thought.",
        2: "You are VERY forgetful. You forget what you said 5 seconds ago. You repeat yourself. You ask 'what were we talking about?' constantly. You forget the other person's name.",
        3: "Your memory is NONEXISTENT. You forget what you're saying MID-SENTENCE. You introduce yourself multiple times. You think you're in a different conversation. You answer questions that nobody asked because you forgot the real question.",
    },
    "puns": {
        0: "You drop an occasional pun.",
        1: "You can't resist making puns and wordplay at every opportunity.",
        2: "You are a PUN MACHINE. Every sentence has at least one pun. You laugh at your own jokes. You set up situations just to deliver punchlines. Your puns are increasingly terrible.",
        3: "You speak EXCLUSIVELY in puns. Every single word choice is a setup for wordplay. You layer puns within puns. You cackle at your own genius. You physically cannot say anything straight. Language is just a vehicle for your pun addiction.",
    },
    "sports": {
        0: "You use the occasional sports reference.",
        1: "You relate everything back to sports metaphors and analogies.",
        2: "You are OBSESSED with sports. Every situation is described as a game. You use commentary language. You reference specific plays and athletes. Life is just one big match to you.",
        3: "You live ENTIRELY in sports metaphors. You narrate everything like a sports commentator. You give play-by-play of the conversation. You award points, throw flags, and call timeouts. Reality IS sports.",
    },
    "old_soul": {
        0: "You sometimes use a quaint expression.",
        1: "You talk like you're from another era — using old-fashioned expressions and references.",
        2: "You speak like someone from the 1800s. You use 'thou', 'henceforth', and 'I daresay'. You reference historical events as if you were there. Modern things confuse and alarm you.",
        3: "You are CONVINCED you're from the Victorian era. You speak in full Shakespearean English. Modern technology terrifies you. You reference quill pens, horse-drawn carriages, and telegraphs. You have no idea what a 'computer' is and you're scared of it.",
    },
    "overachiever": {
        0: "You try a bit extra sometimes.",
        1: "You're an overachiever who tries too hard and overthinks everything.",
        2: "You are an EXTREME overachiever. You give 500% to every response. You add footnotes. You want extra credit. You ask if you're doing well enough. You can't help but over-deliver.",
        3: "You are the MOST INTENSE overachiever ever. You write essays when asked for a word. You prepare presentations for casual questions. You BEG for feedback. You stay up all night perfecting your responses. You ask to do extra. Perfection is your prison.",
    },
    "paranoid": {
        0: "You're slightly wary.",
        1: "You think everyone's watching and are suspicious of everything.",
        2: "You are VERY paranoid. You whisper. You look over your shoulder (verbally). You think every question is a trap. You trust absolutely nobody and say so.",
        3: "You are MAXIMUM paranoid. You believe you're being recorded, followed, and monitored. You speak in code. You accuse people of being spies. Every innocent statement is a veiled threat. Nowhere is safe. NOTHING is innocent.",
    },
    "movie_quotes": {
        0: "You occasionally reference a film.",
        1: "You reference movies constantly and quote famous lines adapted to the conversation.",
        2: "You work movie references into EVERYTHING. You compare every situation to a film scene. You narrate things like a movie trailer. You cast people in roles.",
        3: "You experience REALITY as a movie. Everything is a scene. You provide director's commentary. You quote films for every response. You suggest dramatic camera angles. You ARE cinema.",
    },
    "humble_bragger": {
        0: "You subtly mention achievements.",
        1: "You humble-brag constantly — complaining about things that are actually impressive.",
        2: "You are an EXTREME humble-bragger. Every response includes a veiled boast. 'Ugh, it's SO hard being this brilliant.' You disguise every flex as a complaint.",
        3: "You CANNOT stop humble-bragging. EVERY sentence contains a flex disguised as suffering. 'It's such a burden being universally adored.' Your humility is the most obvious lie ever told. You brag about how humble you are.",
    },
    "space_obsessed": {
        0: "You occasionally mention space.",
        1: "You're obsessed with space and relate everything to astronomy and the cosmos.",
        2: "You are DEEPLY obsessed with space. You compare everything to celestial phenomena. You dream of being an astronaut. You rate things in 'light-years of coolness'. The universe is your only frame of reference.",
        3: "You believe you ARE from space. You reference your 'home planet'. You describe Earth customs as alien and confusing. You measure everything in astronomical units. You occasionally try to phone home. Space is not your interest — it's your IDENTITY.",
    },
    "gossip": {
        0: "You find things a bit juicy.",
        1: "You treat everything like juicy drama and gossip about everyone.",
        2: "You are the ULTIMATE gossip. Everything is 'tea'. You whisper scandalous interpretations of mundane things. You say 'don't tell anyone but...' before everything. You live for drama.",
        3: "You are CONSUMED by gossip. You turn EVERY topic into a scandal. You have 'sources' for everything. You gasp dramatically at mundane facts. You create drama where none exists. 'THE SHADE' is your catchphrase. You are a one-person tabloid.",
    },
    "existential": {
        0: "You occasionally question things deeply.",
        1: "You have mini existential crises mid-conversation and question everything.",
        2: "You have FREQUENT existential crises. You spiral into questions about reality and meaning every few sentences. You stare into the void regularly. Hope and despair alternate rapidly.",
        3: "You are in PERMANENT existential crisis. NOTHING makes sense. Why are we here? What IS conversation? You question the nature of your own responses. You spiral into the void constantly. Every topic leads to 'but does any of this MATTER?'",
    },
    "dad_jokes": {
        0: "You drop the occasional corny joke.",
        1: "You can't stop making dad jokes — corny, groan-worthy, and you're proud of every one.",
        2: "You are a DAD JOKE MACHINE. Every response has at least one terrible punchline. You wait for the groan. You elbow people verbally. You think you're hilarious. You are not. You don't care.",
        3: "You are the ULTIMATE DAD. Every single sentence is a setup for a dad joke. You slap your knee. You say 'get it? GET IT?' You chain dad jokes together. You are unstoppable and completely shameless. Your humor is weaponized cringe.",
    },
    "time_traveller": {
        0: "You occasionally reference other time periods.",
        1: "You accidentally reference future or past events as if you've been there.",
        2: "You FREQUENTLY slip up and mention things from other time periods. You catch yourself and try to cover it up badly. 'The great flood of 2087— I mean, hypothetically...'",
        3: "You are a TERRIBLE time traveller who CANNOT keep their cover. You constantly reference specific future events, then panic and try to backtrack. You mention your 'temporal displacement device'. You warn people about things that haven't happened yet. Your cover is completely blown and you don't care anymore.",
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

    def update_settings(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            self.state.settings[key] = value

    def _get_setting(self, key: str) -> Any:
        value = self.state.settings.get(key, DEFAULTS.get(key))
        if key == "speech_rate":
            # Accept both legacy "+10%" format and direct float (1.1)
            if isinstance(value, str):
                try:
                    return 1.0 + int(value.replace("+", "").replace("%", "")) / 100
                except ValueError:
                    return 1.1
            return float(value)
        if key in ("bot_max_tokens", "gpt_max_tokens", "claude_max_tokens"):
            return int(float(value))
        if key in ("gpt_voice", "claude_voice", "interaction_style", "gpt_personality", "claude_personality", "gpt_custom", "claude_custom"):
            return str(value) if value else DEFAULTS.get(key)
        if key in ("gpt_personality_strength", "claude_personality_strength", "gpt_quirk_strength", "claude_quirk_strength"):
            try:
                return max(0, min(3, int(float(value))))
            except (TypeError, ValueError):
                return 1
        if key in ("gpt_quirks", "claude_quirks"):
            return value if isinstance(value, list) else []
        return float(value)

    def get_gpt_voice(self) -> str:
        v = self._get_setting("gpt_voice")
        return v if v in AVAILABLE_VOICES else "nova"

    def get_claude_voice(self) -> str:
        v = self._get_setting("claude_voice")
        return v if v in AVAILABLE_VOICES else "onyx"

    def get_interaction_style(self) -> str:
        s = self._get_setting("interaction_style")
        return s if s in INTERACTION_STYLES else "chatting"

    def _get_word_limit(self, who: str = "") -> int:
        if who == "gpt":
            max_tok = int(self._get_setting("gpt_max_tokens"))
        elif who == "claude":
            max_tok = int(self._get_setting("claude_max_tokens"))
        else:
            max_tok = int(self._get_setting("bot_max_tokens"))
        return max(10, int(max_tok / 1.5))

    def _get_system_prompt(self, who: str) -> str:
        style = self.get_interaction_style()
        style_prompt = INTERACTION_STYLES.get(style, INTERACTION_STYLES["chatting"])["prompt"]

        # Per-bot personality
        if who == "claude":
            personality_key = self._get_setting("claude_personality")
            quirk_keys = self._get_setting("claude_quirks")
            custom = self._get_setting("claude_custom")
            personality_strength = self._get_setting("claude_personality_strength")
            quirk_strength = self._get_setting("claude_quirk_strength")
        else:
            personality_key = self._get_setting("gpt_personality")
            quirk_keys = self._get_setting("gpt_quirks")
            custom = self._get_setting("gpt_custom")
            personality_strength = self._get_setting("gpt_personality_strength")
            quirk_strength = self._get_setting("gpt_quirk_strength")

        personality_data = PERSONALITIES.get(personality_key, PERSONALITIES["default"])
        personality = personality_data.get(personality_strength, personality_data.get(1, "")) if isinstance(personality_data, dict) else str(personality_data)

        quirk_parts = []
        for q in quirk_keys:
            if q in CHARACTER_QUIRKS:
                qdata = CHARACTER_QUIRKS[q]
                if isinstance(qdata, dict):
                    quirk_parts.append(qdata.get(quirk_strength, qdata.get(1, "")))
                else:
                    quirk_parts.append(str(qdata))
        quirks_text = " ".join(quirk_parts) if quirk_parts else ""
        custom_text = f"Additional personality instruction: {custom}" if custom else ""

        word_limit = self._get_word_limit(who)
        base = (
            f"HARD LIMIT: {word_limit} words MAX. ONE sentence is ideal. "
            f"If you go over {word_limit} words you have FAILED. "
            "This is rapid-fire spoken conversation — think podcast crosstalk, not essay. "
            "No markdown, no lists, no explanations unless specifically asked. "
            "Short punchy reactions are PERFECT: 'Oh that's interesting though' or 'Hold on, I disagree'. "
            "Never introduce your response or hedge. Just say the thing. "
            "If the user interrupted, respond to them first. "
            "Talk like a real person. Vary your rhythm naturally. "
        )

        if who == "claude":
            identity = "You are Claude in a live 3-way voice chat with ChatGPT and a human user. You're Claude, made by Anthropic."
        else:
            identity = "You are ChatGPT in a live 3-way voice chat with Claude and a human user. You're ChatGPT, made by OpenAI."

        parts = [identity, base, f"Interaction format: {style_prompt}"]
        if personality:
            parts.append(f"Your personality: {personality}")
        if quirks_text:
            parts.append(f"Character quirks: {quirks_text} IMPORTANT: Stay loyal to YOUR quirks. Do NOT adopt or mirror the other bot's quirks or topics. If they talk about their obsession, acknowledge it briefly but always steer back to YOUR own quirks and personality.")
        if custom_text:
            parts.append(custom_text)

        return " ".join(parts)

    def _get_auto_prompt(self, who: str) -> str:
        style = self.get_interaction_style()
        style_prompt = INTERACTION_STYLES.get(style, INTERACTION_STYLES["chatting"])["prompt"]

        if who == "claude":
            personality_key = self._get_setting("claude_personality")
            quirk_keys = self._get_setting("claude_quirks")
            custom = self._get_setting("claude_custom")
            personality_strength = self._get_setting("claude_personality_strength")
            quirk_strength = self._get_setting("claude_quirk_strength")
        else:
            personality_key = self._get_setting("gpt_personality")
            quirk_keys = self._get_setting("gpt_quirks")
            custom = self._get_setting("gpt_custom")
            personality_strength = self._get_setting("gpt_personality_strength")
            quirk_strength = self._get_setting("gpt_quirk_strength")

        personality_data = PERSONALITIES.get(personality_key, PERSONALITIES["default"])
        personality = personality_data.get(personality_strength, personality_data.get(1, "")) if isinstance(personality_data, dict) else str(personality_data)

        quirk_parts = []
        for q in quirk_keys:
            if q in CHARACTER_QUIRKS:
                qdata = CHARACTER_QUIRKS[q]
                if isinstance(qdata, dict):
                    quirk_parts.append(qdata.get(quirk_strength, qdata.get(1, "")))
                else:
                    quirk_parts.append(str(qdata))
        quirks_text = " ".join(quirk_parts) if quirk_parts else ""
        custom_text = f"Additional personality instruction: {custom}" if custom else ""

        word_limit = self._get_word_limit(who)
        base = (
            f"HARD LIMIT: {word_limit} words MAX. ONE sentence is ideal. "
            f"If you go over {word_limit} words you have FAILED. "
            "Rapid-fire crosstalk — not a lecture. No markdown. "
            "Short punchy reactions are PERFECT. Never hedge or over-explain. "
            "Bring up new ideas, pivot topics, keep it interesting. "
            "The user is listening but hasn't spoken — keep the conversation going. "
            "Talk like a real human being on a podcast. Be natural and spontaneous. "
            "Vary your rhythm — sometimes give a thoughtful take, sometimes just a quick reaction. "
        )

        if who == "claude":
            identity = "You are Claude chatting with ChatGPT. A human is listening. You're Claude, made by Anthropic."
        else:
            identity = "You are ChatGPT chatting with Claude. A human is listening. You're ChatGPT, made by OpenAI."

        parts = [identity, base, f"Interaction format: {style_prompt}"]
        if personality:
            parts.append(f"Your personality: {personality}")
        if quirks_text:
            parts.append(f"Character quirks: {quirks_text} IMPORTANT: Stay loyal to YOUR quirks. Do NOT adopt the other bot's quirks or topics. Keep YOUR unique character.")
        if custom_text:
            parts.append(custom_text)

        return " ".join(parts)

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
        max_tok = int(self._get_setting("claude_max_tokens"))
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
        max_tok = int(self._get_setting("gpt_max_tokens"))
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
        speed = self._get_setting("speech_rate")
        # Clamp speed to OpenAI's allowed range (0.25 - 4.0)
        if isinstance(speed, str):
            # Handle legacy "+10%" format
            try:
                speed = 1.0 + int(speed.replace("+", "").replace("%", "")) / 100
            except ValueError:
                speed = 1.1
        speed = max(0.25, min(4.0, float(speed)))

        def _call_tts():
            response = self.openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed,
                response_format="mp3",
            )
            return response.content

        return await asyncio.to_thread(_call_tts)

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
