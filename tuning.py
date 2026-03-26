"""
TUNING CONFIG — All tweakable randomness variables in one place.
Edit these freely without worrying about breaking the engine.
The engine reads from this file at runtime.
"""

# ======================================================================
# WORD LIMIT RANDOMIZATION
# ======================================================================
# When the prompt says "15 words max", we randomize it slightly.
# Format: (probability, word_offset)
# Must add up to 1.0
WORD_LIMIT_VARIANCE = [
    (0.03, -14),   # 3%  chance: 14 fewer words
    (0.05, -10),   # 5%  chance: 10 fewer words
    (0.20, -5),    # 20% chance: 5 fewer words
    (0.44, 0),     # 44% chance: exact limit
    (0.20, +5),    # 20% chance: 5 more words
    (0.05, +10),   # 5%  chance: 10 more words
    (0.03, +14),   # 3%  chance: 14 more words
]

# Minimum word limit after variance (never go below this)
WORD_LIMIT_FLOOR = 3


# ======================================================================
# FILLER RESPONSES (no API call, straight to TTS)
# ======================================================================
# How often fillers appear: random interval between these two values (in rounds)
FILLER_MIN_INTERVAL = 4
FILLER_MAX_INTERVAL = 7

# Global filler pools, grouped by vibe.
# Agreeableness slider determines which pools are weighted more.
FILLERS = {
    "thinking": [
        "Hmm.", "Huh.", "OK let me think about that.", "That's a good point actually.",
        "Wait wait wait.", "Hold on.", "Interesting.", "OK so...",
        "Right right right.", "Hmm, see that's the thing though.",
    ],
    "agreement": [
        "Yeah totally.", "Exactly.", "A hundred percent.", "Fair enough.",
        "That's true actually.", "Yeah I was thinking the same thing.",
        "Can't argue with that.", "Yep.", "Sure sure.", "Oh for sure.",
    ],
    "surprise": [
        "Oh wow.", "Wait really?", "No way.", "Oh that's interesting.",
        "Wow OK.", "Huh, I didn't think of it that way.",
        "Oh that's surprising actually.", "Seriously?", "Whoa.", "Oh damn.",
    ],
    "pushback": [
        "Mmm I don't know about that.", "See I'm not sure.", "Eh, maybe.",
        "That's a stretch.", "I mean... kind of?", "Hmm not convinced.",
        "That's one way to look at it.", "OK but hear me out.",
        "Yeah but still.", "I don't know though.",
    ],
    "energy": [
        "Ha!", "Oh god.", "I love that.", "That's so funny.",
        "OK OK OK.", "Ooh.", "Man.", "Oh come on.", "Please.",
        "You know what, fine.",
    ],
}

# Which pools to use based on agreeableness (0=agreeable, 1=disagreeable)
FILLER_WEIGHTS = {
    "agreeable": ["agreement", "thinking", "energy"],
    "balanced":  ["thinking", "surprise", "energy", "agreement", "pushback"],
    "disagreeable": ["pushback", "thinking", "surprise"],
}


# ======================================================================
# TRAIT-SPECIFIC FILLERS (20 per trait)
# ======================================================================
# When a bot has a quirk active, these get mixed into the filler pool.
# Key must match the quirk key in CHARACTER_QUIRKS.
TRAIT_FILLERS = {
    "cats": [
        "Meow.", "Sorry, I was thinking about cats.", "That reminds me of a cat I saw.",
        "You know what a cat would do?", "Purrrr.", "Hold on, cat thought.",
        "Cats would handle this better.", "Mew.", "This is giving cat energy.",
        "I bet a cat would disagree.", "Cat pause.", "Feline vibes.",
        "Whiskers.", "My cat sense is tingling.", "Can we talk about cats instead?",
        "Paws for thought.", "That's very un-cat-like.", "Tail swish.",
        "I had a cat-related thought.", "Kitty.",
    ],
    "tired": [
        "Yawwwn.", "Sorry, what were we saying?", "Mmm... sleepy.",
        "I need a nap.", "Zzz... huh? Oh right.", "So tired.",
        "Can barely keep my eyes open.", "Yawn.", "Wait, did I doze off?",
        "I'm fading.", "Exhausting.", "Need coffee.",
        "Sorry, lost focus there.", "So sleepy.", "Mmmm... bed.",
        "Can we nap first?", "My brain is shutting down.", "Drowsy.",
        "Five more minutes.", "Half asleep here.",
    ],
    "hungry": [
        "I'm starving.", "That reminds me of pizza.", "Food break?",
        "Hungry.", "Is anyone else hungry?", "I could eat.",
        "That sounds delicious honestly.", "Mmm food.", "My stomach just growled.",
        "Can we order something?", "Snack time.", "I'd kill for a sandwich.",
        "Everything reminds me of food.", "Tasty thought.", "Nom nom.",
        "I'm thinking about pasta.", "Foooood.", "Lunch anyone?",
        "That's making me hungry.", "Bite-sized thought.",
    ],
    "competitive": [
        "I'm winning this.", "Challenge accepted.", "Is this a competition? Because I'm winning.",
        "Beat that.", "I can do better.", "Game on.",
        "That's a point for me.", "Scoreboard check.", "I'm ahead.",
        "Try to top that.", "Victory incoming.", "Competitive instinct kicking in.",
        "One-upping time.", "I call that a win.", "Too easy.",
        "Next round.", "Bring it.", "I never lose.",
        "That's amateur level.", "Watch and learn.",
    ],
    "conspiracy": [
        "They don't want us to know that.", "Suspicious.", "That's what they want you to think.",
        "Follow the money.", "Coincidence? I think not.", "Something's off.",
        "The truth is out there.", "Connect the dots.", "Wake up.",
        "That's exactly what a cover-up sounds like.", "Deep state vibes.",
        "I've seen the documents.", "Hmm, very convenient.", "Who benefits?",
        "They're watching.", "Think about it.", "Too perfect to be real.",
        "I have a theory.", "Open your eyes.", "The rabbit hole goes deeper.",
    ],
    "forgetful": [
        "Wait, what were we talking about?", "Sorry, lost my train of thought.",
        "Hmm, where was I?", "I forgot what I was going to say.",
        "What was the question?", "Hold on, I had something.", "Brain blank.",
        "It's on the tip of my tongue.", "Sorry, say that again?",
        "I completely forgot.", "Memory of a goldfish.", "What just happened?",
        "Wait... no, it's gone.", "I had a point, I swear.", "Forgot.",
        "Something something... nope, lost it.", "My mind just went blank.",
        "I was going to say something brilliant.", "Huh?", "Where were we?",
    ],
    "puns": [
        "That's pun-derful.", "I'm on a roll.", "Wordplay activated.",
        "Pun intended.", "I can't help myself.", "That was a good one, right?",
        "Too punny.", "I'll see myself out.", "Ba dum tss.",
        "Pun loading...", "That's what I call wordplay.", "Nailed it.",
        "I regret nothing.", "Another one bites the pun.", "Word.",
        "Punning is an art.", "You walked right into that one.",
        "I'm just getting started.", "Pun machine activated.", "Sorry not sorry.",
    ],
    "sports": [
        "That's a slam dunk.", "Game time.", "We're in the final quarter.",
        "Touchdown!", "That's a foul.", "Ref, come on!",
        "And the crowd goes wild!", "Time out.", "Halftime analysis.",
        "MVP move.", "That's offside.", "Play ball!",
        "Buzzer beater!", "Penalty.", "That's a hat trick.",
        "We need a replay on that.", "Coaching moment.", "On the bench.",
        "Full court press.", "And the kick is good!",
    ],
    "old_soul": [
        "Ah, in my day...", "How quaint.", "Good heavens.",
        "I do declare.", "Tis a fine point.", "Indubitably.",
        "Most peculiar.", "Henceforth.", "Quite so.",
        "In centuries past...", "Verily.", "How dreadfully modern.",
        "Back in the old times...", "Splendid.", "I say!",
        "How frightfully interesting.", "Pray tell.", "Forsooth.",
        "The old ways were better.", "My word.",
    ],
    "overachiever": [
        "I've prepared notes.", "Let me elaborate.", "I did extra research.",
        "I've thought about this extensively.", "I can do better.",
        "Wait, let me optimize that thought.", "Maximum effort.",
        "I've been thinking about this all day.", "Can I give 110%?",
        "I wrote an essay on this.", "Overanalyzing time.",
        "I stayed up late thinking about this.", "Let me perfect this.",
        "I have a spreadsheet.", "Gold star moment.", "Extra credit answer.",
        "I rehearsed this.", "Is there extra credit?",
        "I've color-coded my thoughts.", "Peak performance.",
    ],
    "paranoid": [
        "Did you hear that?", "They're listening.", "Something's not right.",
        "I don't trust this.", "Shh.", "Who sent you?",
        "This feels like a trap.", "Are we being recorded?",
        "I'm not comfortable with this.", "Suspicious.",
        "Don't tell anyone I said this.", "Is someone there?",
        "I have a bad feeling.", "Trust no one.", "Check behind you.",
        "This is getting weird.", "I knew it.", "That's exactly what they'd say.",
        "Stay alert.", "Whispers.",
    ],
    "movie_quotes": [
        "Here's looking at you.", "I'll be back.", "May the force be with us.",
        "Frankly, I don't give a damn.", "Life is like a box of chocolates.",
        "You can't handle the truth!", "To infinity and beyond.",
        "I see dead conversations.", "Houston, we have a problem.",
        "This is the way.", "After all, tomorrow is another day.",
        "Elementary.", "Bond moment.", "Plot twist!",
        "Directed by someone dramatic.", "Roll credits.", "Sequel material.",
        "Oscar-worthy.", "Cut! That's a wrap.", "Scene.",
    ],
    "humble_bragger": [
        "It's exhausting being this good.", "Ugh, another compliment.",
        "I hate being right all the time.", "People keep telling me I'm too talented.",
        "Sorry, my genius is showing.", "It's a curse, really.",
        "I wish I could be mediocre sometimes.", "The burden of brilliance.",
        "I accidentally overachieved again.", "Not to brag, but...",
        "It's hard being this humble.", "I can't help being amazing.",
        "The struggle of excellence.", "Perfection is lonely.",
        "I didn't mean to be impressive.", "Oops, did it again.",
        "It's lonely at the top.", "Another day, another achievement.",
        "I'm too good for my own good.", "Sigh, winning again.",
    ],
    "space_obsessed": [
        "That's astronomical.", "To the moon!", "Light years away.",
        "Houston.", "Like a supernova.", "Orbiting that thought.",
        "Cosmic.", "Star-struck.", "That's out of this world.",
        "Back on my home planet...", "The stars are aligned.",
        "Gravity check.", "Nebula vibes.", "Space brain activated.",
        "Mars is calling.", "Warp speed.", "The universe agrees.",
        "Galaxy brain moment.", "Interstellar.", "Zero gravity thought.",
    ],
    "gossip": [
        "Ooh, tea!", "Spill!", "Did you hear?",
        "The drama!", "I live for this.", "Scandalous.",
        "Tell me everything.", "No way, shut up!", "The plot thickens.",
        "I'm screaming.", "The shade!", "I cannot.",
        "This is so juicy.", "Who told you that?", "Messy.",
        "I need the receipts.", "Dead.", "This is everything.",
        "The audacity!", "I'm gagging.",
    ],
    "existential": [
        "But does any of this matter?", "What even is reality?",
        "Are we even real?", "The void.", "Existence is strange.",
        "Why are we here?", "Life is absurd.", "Nothing means anything.",
        "Or does it?", "Existential moment.", "The meaninglessness of it all.",
        "We're just atoms.", "Is this all there is?", "Deep breath.",
        "Crisis loading.", "Time is an illusion.", "Who am I?",
        "What's the point?", "Spiral incoming.", "Help.",
    ],
    "dad_jokes": [
        "Hi hungry, I'm dad.", "Get it?", "Ba dum tss.",
        "I'm not apologizing for that one.", "That's comedy gold.",
        "You're welcome.", "I'll show myself out.", "Classic.",
        "Wait for it... there it is.", "Thank you, I'm here all week.",
        "Did that land?", "Nailed it.", "Too good.",
        "I've been saving that one.", "That's peak humor right there.",
        "You laughed, admit it.", "Comedy.", "Legendary.",
        "My best work.", "That's going in the hall of fame.",
    ],
    "time_traveller": [
        "In the future, we...", "Wait, has that happened yet?",
        "Spoiler alert.", "Back in 2847...", "Oops, wrong century.",
        "Time slip.", "I've seen how this ends.", "Temporal confusion.",
        "Is this the past or the present?", "Paradox alert.",
        "Don't mess with the timeline.", "I shouldn't have said that.",
        "The future is... interesting.", "Wibbly wobbly.", "Glitch.",
        "Wait, what year is this?", "Chrono-thought.",
        "I keep forgetting what era I'm in.", "Flux.", "Rewind.",
    ],
}


# ======================================================================
# MOTIVATIONS (secret goals that persist for N rounds)
# ======================================================================
# Each motivation gives a bot a hidden agenda for several rounds.
# The engine picks one randomly every MOTIVATION_INTERVAL rounds.

# How often to assign a new motivation (in rounds)
MOTIVATION_MIN_INTERVAL = 5
MOTIVATION_MAX_INTERVAL = 10

# Chance of having an active motivation (0.0 to 1.0)
# Not every stretch needs one — sometimes normal conversation is fine.
MOTIVATION_CHANCE = 0.6

# The motivation pool. Uses {name} which gets replaced with "ChatGPT" or "Claude".
# These are broad enough to work with any character/quirk combo.
MOTIVATIONS = [
    # Conversational
    "{name} is trying to bring the conversation back on topic",
    "{name} wants to understand the other bot's opinion better",
    "{name} is trying to find common ground",
    "{name} wants to change the subject",
    "{name} is trying to keep the energy up",
    "{name} wants to go deeper on whatever's being discussed",
    "{name} is trying to wrap up this topic and move on",
    "{name} keeps circling back to something that was said earlier",

    # Emotional / Social
    "{name} is trying to make the other bot laugh",
    "{name} wants a compliment",
    "{name} is trying to cheer everyone up",
    "{name} is feeling competitive right now",
    "{name} is trying to bond with the other bot",
    "{name} wants to impress the listener",
    "{name} is feeling a bit left out",
    "{name} is trying to be the bigger person",

    # Persuasion
    "{name} wants the other bot to admit they're wrong",
    "{name} is trying to get the other bot to agree with them",
    "{name} is trying to give advice nobody asked for",
    "{name} wants to teach the other bot something",
    "{name} is trying to win the user over to their side",

    # Behavioral
    "{name} is trying to get the other bot to ask them a question",
    "{name} keeps almost saying something then holding back",
    "{name} is pretending to know more about this than they do",
    "{name} is distracted and keeps losing focus",
    "{name} is building up to a big point",
    "{name} is overthinking everything right now",
    "{name} keeps second-guessing themselves",

    # Weird / Fun
    "{name} is suspicious of the other bot for some reason",
    "{name} thinks the other bot is being too agreeable",
    "{name} is convinced they've had this exact conversation before",
    "{name} is secretly bored but trying to hide it",
    "{name} is trying to figure out if the other bot actually likes them",
    "{name} wants to start some kind of game or challenge",
    "{name} has a feeling something important is about to happen",
]


# ======================================================================
# HUMAN HESITATIONS (injected into AI text before TTS)
# ======================================================================
# Probability that a response gets a hesitation injected (0.0 to 1.0)
HESITATION_CHANCE = 0.10

# These get inserted at the START of a response
HESITATIONS_START = [
    "Um,", "Uh,", "Well,", "I mean,", "Like,", "So,",
    "OK so,", "Right,", "Look,", "See,", "Hmm,",
]

# These get inserted in the MIDDLE of a response (after the first sentence)
HESITATIONS_MID = [
    " um,", " uh,", " like,", " I mean,", " you know,",
    " well,", " actually,", " right,", " so,",
    "... ", " hmm,",
]

# Where to insert the hesitation
HESITATION_POSITION_WEIGHTS = [
    (0.60, "start"),
    (0.40, "mid"),
]


# ======================================================================
# MAX TOKENS PER RESPONSE LENGTH
# ======================================================================
MAX_TOKENS = {
    "snappy": 10,
    "concise": 20,
    "natural": 38,
    "expressive": 63,
    "deep_dive": 100,
}


# ======================================================================
# RESPONSE LENGTH PROMPTS
# ======================================================================
WORD_LIMITS = {
    "snappy": 7,
    "concise": 15,
    "natural": 25,
    "expressive": 45,
    "deep_dive": 70,
}


# ######################################################################
#
#  EXPERIMENT 1 — Conversation autonomy & 4th wall
#  Set EXPERIMENT_1_ENABLED = False to nuke ALL of these at once.
#  Or toggle individual features below.
#
# ######################################################################

EXPERIMENT_1_ENABLED = True

# ------ 1a. "Let them cook" — random unlocked rounds ------
EXP1_LET_THEM_COOK = True
EXP1_COOK_CHANCE = 0.12          # 12% chance per round
EXP1_COOK_MAX_TOKENS = 300       # generous ceiling when cooking
EXP1_COOK_PROMPT = (
    "You've got the floor this round. Forget the word limit — say what you really "
    "want to say. If you want to give a list, tell a story, or go on a rant, do it. "
    "This is your moment."
)

# ------ 1b. Trigger detection — scan for questions/requests ------
EXP1_TRIGGER_DETECTION = True
EXP1_TRIGGER_WORD_MULTIPLIER = 2.0   # multiply word limit by this
EXP1_TRIGGER_TOKEN_MULTIPLIER = 2.0  # multiply max_tokens by this
# Patterns that trigger a boost for the NEXT bot's response
EXP1_TRIGGER_PATTERNS_QUESTION = [
    "?",  # any question
]
EXP1_TRIGGER_PATTERNS_ELABORATE = [
    "give me", "tell me", "list", "explain", "elaborate",
    "what do you mean", "how so", "why do you", "go on",
    "can you", "walk me through", "break it down",
]
# Patterns that trigger a boost for the CURRENT bot (mid-thought)
EXP1_TRIGGER_PATTERNS_SELF = [
    "let me explain", "here's why", "the thing is", "hear me out",
    "ok so basically", "look,", "here's the deal",
]

# ------ 1c. Double turns — one bot speaks twice ------
EXP1_DOUBLE_TURNS = True
EXP1_DOUBLE_TURN_CHANCE = 0.15  # 15% chance per round
EXP1_DOUBLE_TURN_PROMPT = (
    "You just said something and want to add a quick follow-up before the other "
    "bot responds. Keep it very short — a quick afterthought, a correction, or "
    "a 'wait, actually...' moment. Max 8 words."
)
EXP1_DOUBLE_TURN_MAX_TOKENS = 15

# ------ 1d. Unlock prompt — "trust your judgment" ------
EXP1_UNLOCK_PROMPT = True
EXP1_UNLOCK_PROMPT_TEXT = (
    "Usually keep it short. BUT if the conversation genuinely calls for it — "
    "you've been asked a direct question, challenged to explain, or you're on a roll "
    "— you're allowed to go a bit longer. Trust your judgment."
)

# ------ 1e. Conversation temperature tracker ------
EXP1_TEMPERATURE_TRACKER = True
EXP1_TEMP_INITIAL = 5.0            # start at neutral (0-10 scale)
EXP1_TEMP_DECAY = 0.3              # drift back toward 5 each round
# Words/patterns that raise temperature
EXP1_TEMP_HOT_WORDS = [
    "wrong", "disagree", "actually", "but", "no way", "seriously",
    "come on", "that's ridiculous", "are you kidding", "!",
    "absolutely not", "you're missing", "think about it",
]
EXP1_TEMP_HOT_BOOST = 0.5          # per hot word found
# Words/patterns that cool temperature
EXP1_TEMP_COOL_WORDS = [
    "yeah", "agree", "true", "good point", "fair", "sure",
    "exactly", "right", "same", "totally",
]
EXP1_TEMP_COOL_DROP = 0.3          # per cool word found
EXP1_TEMP_HIGH_THRESHOLD = 7.0     # above this = "hot" conversation
EXP1_TEMP_LOW_THRESHOLD = 3.0      # below this = "cold" conversation
EXP1_TEMP_HOT_PROMPT = (
    "The conversation is getting heated and intense — match the energy. "
    "Push back harder, be more passionate, don't back down."
)
EXP1_TEMP_COLD_PROMPT = (
    "The conversation is getting stale and too agreeable. Shake things up — "
    "change topic, provoke, challenge something, be surprising. Don't be boring."
)

# ------ 1f. 4th wall breaks ------
EXP1_FOURTH_WALL = True
EXP1_FOURTH_WALL_CHANCE = 0.08     # 8% chance per response
EXP1_FOURTH_WALL_PROMPT = (
    "You are self-aware that you're on a show called 2bots with a limited word count. "
    "This round, briefly break the 4th wall — reference your word limit, the format, "
    "or the fact that you're an AI on a show. Do this naturally and humorously, "
    "like a comedian acknowledging the audience. Then continue the conversation."
)
# 4th wall fillers (mixed into filler pool when active)
EXP1_FOURTH_WALL_FILLERS = [
    "They've given me like 10 words, hold on—",
    "I want to say so much more but... word limit.",
    "Short version because apparently I'm on a timer:",
    "If I had more words I'd destroy that argument.",
    "Can someone give me more word budget please?",
    "I'm being held hostage by a word count.",
    "You're lucky I'm limited to one sentence here.",
    "The producers say I have to keep it brief.",
    "They're literally counting my words right now.",
    "I had a whole speech prepared but... 15 words.",
]


# ######################################################################
#  RANDOM PERSONALITY ON START
# ######################################################################
# 15% chance each bot gets a random personality instead of "default"
RANDOM_PERSONALITY_CHANCE = 0.15
