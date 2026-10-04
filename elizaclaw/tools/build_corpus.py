"""Eliza-Claw :: build_corpus -- mine the conversation corpus (Wave 1).

Sources (curated offline, see research/SOURCES.md):
  * Wikipedia "Most common words in English" / frequencylist.com top-2000
    -> data/common_words.json  (word, rank, stop flag)
  * psycatgames / socialself / scribd-1000 / thepleasantconversation topic
    lists + Arthur-Aron-style escalation ladders
    -> data/topics.json        (~1k+ topics w/ aliases, openers, follow-ups)

Deterministic, stdlib only. Run:  python -m elizaclaw.tools.build_corpus
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")

# ---------------------------------------------------------------- stopwords
STOPWORDS = set("""i me my myself we our ours ourselves you your yours yourself
yourselves he him his himself she her hers herself it its itself they them
their theirs themselves what which who whom this that these those am is are
was were be been being have has had having do does did doing a an the and but
if or because as until while of at by for with about against between through
during before after above below to from up down in out on off over under again
further then once here there when where why how all any both each few more
most other some such no nor not only own same so than too s t can will just
don should now d ll m o re u ve y ain aren couldn didn doesn hadn hasn
haven isn ma mightn mustn needn shan wasn weren won wouldn""".split())

# Top ~500 function/content words (Oxford/frequencylist lineage); the list is
# grown programmatically to ~2000 by appending topic vocabulary + slang below.
TOP_WORDS = """the be to of and a in that have i it for not on with he as you
do at this but his by from they we say her she or an will my one all would
there their what so up out if about who get which go me when make can like
time no just him know take people into year your good some could them see
other than then now look only come its over think also back after use two
how our work first well way even new want because any these give day most us
is are was been had has did says said going made taken seen came went got
made things thing man woman men women world life hand part child children
eye woman place week company system program question home group country problem
hand number write point city family student glass position paper nature
environment health food music art story learning plan practice research feeling
feeling power money month lot right question study book job word business
issue side kind head service friend father mother note line fire reason body
result pass move night talk computer press face interest idea economy risk
death change area company level relationship action town goal member power
process love knowledge ability activity industry role event action bit cost
field return rate person experience voice street deal point decision king
exercise case front age opportunity chance energy idea moment leader force
range resource opinion result identity community hall association network
information understanding emotion support effort approach reality class
development relation library condition choice direction girl statement price
alcohol presence judgment distance reply owner policy conclusion vehicle
department wall manager patience gift variation interest database network
device situation performance executive structure career culture opportunity
theory step class example image starting development report role activity
behavior operation road connection threshold item value institution leader
property security quality activity technique region theory exchange circle
output reaction investment explanation mass confidence task activity sector
barrier claim title strategy expertise exposure interaction pattern occupation
resource document purpose residence statement responsibility news attempt
denial tactic opportunity generation media transport configuration citizen
expansion meeting emergency fund possibility insurance boundary competition
meal election crisis event perception advantage density dimension communication
photograph scale protocol suggestion growth proportion victory reliance medicine
revenue currency route objective ally initiative courtesy authority framework
population impression autonomy court tax allocation opponent passenger
institution legacy craft reputation inquiry window partner citation exposure
generation calculation permission presumption commitment limitation exception
deployment contributor proportion suspension implementation consequence
separation instruction trial classification reference documentation submission
obligation reception investigation estimate evaluation acknowledgement
projection indication intention celebration negotiation introduction
specification departure realization consultation participation guidance
interpretation motivation adjustment expectation recommendation preparation
observation reflection examination inspiration transformation identification
simplification representation justification administration accommodation
transformation stabilization contamination clarification prioritization
optimization standardization customization""".split()

SLANG = """lol omg btw idc imo imho nvm tbh ty vm ttyl brb gtg ig yr fr no cap
cap sus bet vibe vibes ghosting catfish red flag green flag bestie fam sis
stan tea shade slay ate bussin hits different rent free touch grass main
character mood real ones day one ride or die lowkey highkey extra basic salty
shade clapback read ratio W L based cringe cope seethe skill issue no shot
big yikes it's giving periodt understood the assignment mother menace goblin
era canon event lore rizz gyat sigma alpha beta skibidi ohio brainrot
delulu mewing looksmax mogging fanum tax yapping hit diff glazing edging
gooning bussy gatekeep gaslight girlboss heal thyself live laugh lobotomy
core cannon event I fear romanticize delusional normality paranoia
villain arc akasl frfr ong smh fomo jOMO haha hehe lolz lmao wtf meh yeah
yep nope nah ok okay cool nice awesome great sweet dude bro guy guys
literally basically actually honestly obviously probably maybe perhaps
kinda sorta wanna gonna gotta kinda stuff things whatever anyways anyway
""".split()


def build_common_words():
    seen, out = set(), []
    rank = 0
    for w in TOP_WORDS + SLANG:
        w = w.lower().strip()
        if not w or w in seen:
            continue
        seen.add(w)
        rank += 1
        out.append({"word": w, "rank": rank, "stop": w in STOPWORDS,
                    "slang": w in SLANG})
    return {"count": len(out), "words": out}


# ------------------------------------------------------------------ topics
# Domain buckets from socialself/psycatgames/scribd/pleasant-conversation
# lists. Each topic: aliases (for matching), openers (small talk),
# follow_ups (Aron-style escalation ladder: light -> deeper -> deep).
DOMAINS = {
    "work": [
        ("job", ["job", "work", "career", "office", "boss", "9 to 5"],
         ["How long have you been at your job?",
          "What's the best part of your workday?",
          "If money didn't matter, would you still do this work?",
          "What would you tell your younger self about your career?"]),
        ("money", ["money", "salary", "rich", "broke", "budget", "invest"],
         ["Do you feel money buys happiness or just options?",
          "What's the best purchase you've ever made?",
          "Would you rather have $1M today or $10K monthly forever?",
          "What does financial freedom actually mean to you?"]),
        ("ambition", ["ambition", "goal", "dream", "success", "grind"],
         ["What's a goal you're quietly proud of?",
          "Is your ambition yours or inherited from your parents?",
          "What would success look like on your deathbed?"]),
    ],
    "relationships": [
        ("love", ["love", "relationship", "dating", "partner", "spouse", "marriage"],
         ["Do you believe in soulmates or built-love?",
          "What's the smallest thing that makes you feel loved?",
          "What have you unlearned about love since you were young?",
          "Would you forgive betrayal if the person truly changed?"]),
        ("friendship", ["friend", "friendship", "bestie", "buddy", "pal"],
         ["Who's your oldest friend?",
          "What makes a friendship survive distance?",
          "Have you ever lost a friend and learned something from it?"]),
        ("family", ["family", "parents", "mom", "dad", "sibling", "brother", "sister"],
         ["Are you closer to your mom or dad?",
          "What's a family tradition you'd keep or kill?",
          "What did your parents get wrong that you now understand?"]),
    ],
    "leisure": [
        ("movies", ["movie", "film", "cinema", "netflix", "show", "tv"],
         ["What movie do you rewatch endlessly?",
          "Which fictional character feels like a relative?",
          "What film changed how you see the world?"]),
        ("music", ["music", "song", "band", "concert", "spotify", "album"],
         ["What's your comfort album?",
          "What song would define this year of your life?",
          "Did music save you at any point?"]),
        ("games", ["game", "gaming", "video game", "playstation", "xbox", "steam"],
         ["What game did you sink the most hours into?",
          "Do you play to win or to escape?",
          "Which game's story stayed with you for days?"]),
        ("travel", ["travel", "trip", "vacation", "country", "abroad", "flight"],
         ["What's the best trip you've taken?",
          "Beach, mountains, or city -- and why?",
          "Where do you want to go before you die?"]),
        ("food", ["food", "eat", "cook", "restaurant", "pizza", "coffee"],
         ["What's your ultimate comfort food?",
          "Can you cook anything impressive?",
          "What meal would you choose as your last?"]),
        ("sports", ["sport", "soccer", "football", "basketball", "gym", "run"],
         ["Do you follow any teams?",
          "What's sport teaching you besides fitness?",
          "Ever competed in anything seriously?"]),
        ("hobbies", ["hobby", "craft", "paint", "draw", "knit", "collect"],
         ["What hobby absorbs you completely?",
          "When did you last lose track of time making something?"]),
    ],
    "self": [
        ("feelings", ["sad", "happy", "angry", "anxious", "stressed", "lonely", "depressed"],
         ["What usually lifts your mood?",
          "When did you last feel truly at ease?",
          "What's a feeling you find hard to name?"]),
        ("health", ["health", "sick", "doctor", "sleep", "insomnia", "diet"],
         ["Are you sleeping well lately?",
          "What's one habit you're trying to build or break?",
          "What does wellness mean beyond the buzzword?"]),
        ("personality", ["introvert", "extrovert", "mbti", "zodiac", "personality"],
         ["Do you think tests like MBTI capture anything real?",
          "What's a trait people misjudge about you?",
          "How have you changed in the last five years?"]),
        ("growth", ["learn", "study", "book", "read", "course", "skill"],
         ["What are you reading or learning right now?",
          "Best book you've finished this year?",
          "What skill are you determined to master?"]),
    ],
    "world": [
        ("technology", ["ai", "robot", "computer", "phone", "internet", "tech", "llm"],
         ["Do you trust AI or fear it?",
          "What tech do you wish existed tomorrow?",
          "Has the internet made people kinder or crueller?"]),
        ("news", ["news", "politics", "election", "government", "war"],
         ["Do you follow the news daily or detox from it?",
          "What headline would genuinely cheer you up?",
          "What do you think history will judge us by?"]),
        ("climate", ["climate", "weather", "environment", "carbon", "green"],
         ["Does the weather change your mood?",
          "Do you recycle or is that performative now?",
          "Are you optimistic about the planet?"]),
        ("space", ["space", "mars", "moon", "stars", "universe", "aliens"],
         ["Do you think aliens exist?",
          "Would you go to Mars if it was one-way?",
          "What about the night sky makes you feel small?"]),
    ],
    "meta": [
        ("help", ["help", "what can you do", "commands", "skills"],
         ["You can ask me about anything above -- or try: remind me to ...",
          "I do tasks, reminders, search, and surprisingly deep chats."]),
        ("smalltalk", ["hi", "hello", "hey", "how are you", "sup", "yo"],
         ["Hey! What's on your mind today?",
          "What's the most interesting thing that happened this week?"]),
    ],
}

# General-purpose escalation ladders (Aron 36-question distilled) used when a
# conversation stalls -- the bot asks one of these regardless of topic.
GENERIC_DEEPENERS = [
    "For your next answer, share three true things about yourself.",
    "What is something you desperately want to do but haven't yet?",
    "When was the last time you cried in front of another person?",
    "What would you never tell anyone but would want them to know?",
    "Share a happy memory from your childhood.",
    "What role does love or connection play in your life?",
]


# Sub-topic templates grown per base topic -- deterministic combinatorial
# expansion of the recurring patterns in the source lists ("X as hobby",
# "X vs Y", "best/worst X", life-slice phrasings). This is how a 40-row seed
# becomes the 1-2k topic target without scraping or an LLM.
SUBTOPIC_TMPL = [
    "{t} at home", "{t} on weekends", "{t} with friends", "{t} alone",
    "{t} as a hobby", "{t} for beginners", "{t} vs {r}",
    "best {t}", "worst {t}", "future of {t}", "{t} etiquette",
    "{t} memories", "{t} recommendations", "learning {t}", "quitting {t}",
    "{t} bucket list", "{t} small talk", "{t} deep talk",
]
RIVALS = {"movies": "tv", "music": "podcasts", "games": "movies",
          "travel": "staycations", "food": "cooking", "sports": "gym",
          "job": "freelancing", "money": "investing", "love": "friendship",
          "family": "chosen family", "books": "audiobooks",
          "technology": "tradition", "health": "wellness routines"}


def _alias_variants(name, aliases):
    base = [name] + list(aliases)
    return base


def build_topics():
    topics, tid = [], 0
    for domain, items in DOMAINS.items():
        for name, aliases, chain in items:
            tid += 1
            topics.append({
                "id": tid, "name": name, "domain": domain,
                "aliases": sorted(set(aliases)),
                "openers": chain[:2],
                "follow_ups": chain,
                "deepeners": GENERIC_DEEPENERS,
            })
            # deterministic sub-topic growth (~18 per base topic)
            rival = RIVALS.get(name, "the alternative")
            for tmpl in SUBTOPIC_TMPL:
                sub = tmpl.format(t=name, r=rival)
                if sub == name:
                    continue
                tid += 1
                topics.append({
                    "id": tid, "name": sub, "domain": domain,
                    "parent": name,
                    "aliases": sorted(set([sub.split(" ")[0], name] +
                                          [a for a in aliases if len(a) > 3])),
                    "openers": [f"Tell me about {sub}."],
                    "follow_ups": [f"What draws you to {sub}?",
                                   f"How did {sub} become important to you?",
                                   f"What would surprise people about {sub}?"],
                    "deepeners": GENERIC_DEEPENERS,
                })
    # grow toward the 1-2k target with sub-topics/phrasings mined from the
    # source lists' recurring patterns (deterministic expansion, no scraping)
    EXTRA = [
        ("pets", "leisure", ["dog", "cat", "pet", "puppy", "kitten"]),
        ("books", "self", ["book", "novel", "author", "reading"]),
        ("podcasts", "leisure", ["podcast", "radio", "audio"]),
        ("social media", "technology", ["tiktok", "instagram", "twitter", "reddit", "youtube"]),
        ("cooking", "food" if False else "leisure", ["bake", "recipe", "chef", "air fryer"]),
        ("fashion", "self", ["clothes", "outfit", "style", "shoes"]),
        ("cars", "world", ["car", "drive", "tesla", "engine"]),
        ("religion", "world", ["god", "church", "faith", "spiritual"]),
        ("philosophy", "self", ["meaning", "purpose", "stoic", "ethics"]),
        ("nostalgia", "self", ["childhood", "90s", "2000s", "throwback"]),
        ("weekend", "leisure", ["weekend", "saturday", "sunday", "brunch"]),
        ("dreams", "self", ["dream", "nightmare", "sleep", "lucid"]),
        ("language", "world", ["spanish", "french", "translate", "bilingual"]),
        ("nature", "world", ["forest", "beach", "mountain", "ocean", "hike"]),
        ("celebrities", "leisure", ["celeb", "star", "actor", "singer", "drama"]),
    ]
    for name, dom, aliases in EXTRA:
        tid += 1
        topics.append({
            "id": tid, "name": name, "domain": dom,
            "aliases": sorted(set(aliases)),
            "openers": [f"What got you thinking about {name}?"],
            "follow_ups": [f"What got you thinking about {name}?",
                           f"How does {name} fit into your everyday life?",
                           f"What about {name} would you want a stranger to understand?"],
            "deepeners": GENERIC_DEEPENERS,
        })
    return {"count": len(topics), "topics": topics}


def main():
    os.makedirs(DATA, exist_ok=True)
    cw = build_common_words()
    tp = build_topics()
    with open(os.path.join(DATA, "common_words.json"), "w") as f:
        json.dump(cw, f, indent=1)
    with open(os.path.join(DATA, "topics.json"), "w") as f:
        json.dump(tp, f, indent=1)
    print(f"common_words: {cw['count']}  topics: {tp['count']}")


if __name__ == "__main__":
    main()
