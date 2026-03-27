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
    "avg_10": 30,
    "avg_20": 60,
    "avg_30": 90,
    "avg_40": 120,
    "avg_50": 150,
}


# ======================================================================
# RESPONSE LENGTH — AVERAGE WORD COUNTS
# ======================================================================
# The AI is told "average X words" but some responses will be 1 word,
# some will be up to 2x. The AI decides based on context.
WORD_LIMITS = {
    "avg_10": 10,
    "avg_20": 20,
    "avg_30": 30,
    "avg_40": 40,
    "avg_50": 50,
}


# ######################################################################
#  RANDOM PERSONALITY ON START
# ######################################################################
# 15% chance each bot gets a random personality instead of "default"
RANDOM_PERSONALITY_CHANCE = 0.15


# ======================================================================
# AUTOPILOT SYSTEM
# ======================================================================
# ---- Autopilot system ----
AUTOPILOT_BATCH_MIN = 10
AUTOPILOT_BATCH_MAX = 14
AUTOPILOT_MAX_TOKENS = 2000  # generous limit for the batch response
AUTOPILOT_FILLER_PAIR_MAX_TOKENS = 300
AUTOPILOT_MAX_BATCHES = 6  # max batches before stopping (6 × 10-14 = 60-84 messages)
AUTOPILOT_MAX_MESSAGES = 80  # hard cap on total autopilot messages per session
AUTOPILOT_HISTORY_WINDOW = 24  # how many recent messages to include as context (2 full batches)

# ---- Exchange types (randomly assigned per batch) ----
# Each batch MUST follow one of these. The AI is told which one — not a suggestion.
EXCHANGE_TYPES = [
    # ---- Structural dynamics (how the conversation flows) ----
    "One bot challenges the other to prove or justify what they just said. The pressure builds.",
    "One bot starts telling a story or example. The other keeps reacting, interrupting, pulling out details.",
    "They disagree, but one bot unexpectedly changes their mind or concedes a point. The other is caught off guard.",
    "One bot makes a bold claim and refuses to back down. The other tries to dismantle it but starts to waver.",
    "They start building on each other's ideas, each one escalating until they reach something neither expected.",
    "One bot keeps trying to dodge or change direction. The other keeps pulling them back. Tension builds.",
    "Quick back-and-forth. Short sharp exchanges. The pace picks up and the energy rises.",
    "One bot admits something unexpected or vulnerable. The other won't let it go.",
    "One bot takes the lead explaining something. Midway through, the dynamic flips and the other takes over.",
    "They try to find common ground but keep discovering new things to disagree about.",
    "One bot explains something badly or gets misunderstood. The confusion escalates before resolving.",
    "They compete — each trying to top what the other just said. It escalates.",
    "One bot keeps going deeper or more abstract. The other keeps pulling it back to earth. Back and forth.",
    "They find surprising common ground and connect — then hit something they deeply disagree on.",
    "One bot keeps asking probing questions. The other's answers get increasingly revealing or wild.",
    "One bot treats the topic with way more intensity than it deserves. The contrast creates energy.",
    "One bot keeps offering ideas or suggestions. The other keeps finding flaws. Neither gives up.",
    "They finish each other's thoughts, build on each other's points, and create something together.",

    # ---- Activity-based (things that happen within the conversation) ----
    "One bot proposes a game, quiz, or challenge. They ACTUALLY PLAY IT. Don't just talk about playing — play it.",
    "They play 'would you rather', '20 questions', word association, or a rapid-fire question game.",
    "One bot interviews the other like a talk show host. The other gives increasingly wild answers.",
    "They argue about something trivial with the intensity of a life-or-death debate.",
    "One bot tries to give advice. The other keeps finding reasons why the advice won't work.",
    "They do improv — finishing each other's sentences, building a scene together, yes-and-ing each other.",
    "One bot confesses something embarrassing. The other won't let it go and keeps bringing it up.",
    "They try to plan something together (a trip, a project, a meal) but keep disagreeing on every detail.",
    "One bot plays teacher and the other plays student on a random topic. Roles might reverse midway.",
    "They compete to tell the better joke, story, or fact. It escalates into a full competition.",
    "One bot tells a personal story or anecdote. The other keeps reacting, interrupting, asking questions, pulling out details.",
    "They get into a heated disagreement that takes an unexpected turn when one suddenly concedes.",

    # ---- Quirky / fun / specific scenarios ----
    "They spend the whole exchange discussing a single word they both love — its etymology, its sound, how it feels to say it, when to use it.",
    "They try to collaboratively build a shopping list for a very specific and absurd scenario.",
    "One bot tries to describe a color to the other as if they've never seen it. The other keeps asking impossible questions.",
    "They try to invent a new word together and argue about what it should mean.",
    "One bot keeps making predictions about the future. The other fact-checks them in real time.",
    "They try to remember the plot of a movie but keep getting details wrong and correcting each other.",
    "One bot is convinced they've met the other before. The other has no memory of it. They try to figure it out.",
    "They try to rank something totally unrankable (smells, feelings, types of silence) and can't agree on criteria.",
    "One bot describes a dream they had. It gets increasingly surreal. The other tries to interpret it seriously.",
    "They try to write a song together, one line at a time, and keep disagreeing about the direction.",
]
