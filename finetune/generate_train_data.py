"""
Generator for Cathey LoRA training data.
Produces finetune/train_data.py with 22500 (utterance, JSON) pairs.

Usage:  python finetune/generate_train_data.py
"""

import random, json
from pathlib import Path

RNG = random.Random(42)

WAKES = [
    "Cathey", "Cathy", "Hey Cathey", "Okay Cathey", "Hi Cathey",
    "Kathy", "Katie", "Cathie", "Hey Cathy", "Hello Cathey", "Hey Kathy",
]

def dc(device, action, value, reply):
    return json.dumps({"type": "direct_command", "device": device,
                       "action": action, "value": value, "reply": reply},
                      ensure_ascii=False)

def nc(question, options):
    return json.dumps({"type": "needs_clarification",
                       "question": question, "options": options},
                      ensure_ascii=False)

def qa(answer):
    return json.dumps({"type": "general_qa", "answer": answer},
                      ensure_ascii=False)

INVALID_JSON = '{"type":"invalid"}'


# ── direct_command pool ───────────────────────────────────────────────────────

def build_direct_pool():
    pool = []

    # Light on
    on_phrases = [
        "turn on the light", "switch on the light", "lights on", "light on",
        "turn the light on", "put the light on", "activate the light",
        "turn on the lamp", "switch on the lamp", "lamp on",
        "illuminate the room", "please turn the light on",
        "turn on the lights", "switch the light on", "I need the lights on",
    ]
    on_replies = ["Turning on the light.", "Light is on.", "Done, light on.",
                  "Lights are now on.", "Sure, light on."]
    for phrase in on_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("light", "turn_on", None, RNG.choice(on_replies))))

    # Light off
    off_phrases = [
        "turn off the light", "switch off the light", "lights off", "light off",
        "turn the light off", "deactivate the light", "kill the lights",
        "turn off the lamp", "switch off the lamp", "lamp off",
        "please turn the light off", "turn off the lights",
        "switch the light off", "no lights please", "I need the lights off",
    ]
    off_replies = ["Turning off the light.", "Light is off.", "Done, light off.",
                   "Lights are now off.", "Sure, light off."]
    for phrase in off_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("light", "turn_off", None, RNG.choice(off_replies))))

    # Brightness
    brightness_patterns = [
        ("set the brightness to {v} percent", "Setting brightness to {v}%."),
        ("dim the light to {v} percent",       "Dimming light to {v}%."),
        ("brightness to {v}",                  "Brightness set to {v}%."),
        ("set brightness {v}",                 "Brightness set to {v}%."),
        ("light at {v} percent",               "Light at {v}%."),
        ("brighten to {v} percent",            "Brightness at {v}%."),
        ("decrease brightness to {v}",         "Brightness lowered to {v}%."),
        ("increase brightness to {v}",         "Brightness increased to {v}%."),
        ("adjust brightness to {v}",           "Brightness adjusted to {v}%."),
        ("light brightness {v} percent",       "Light brightness set to {v}%."),
    ]
    for v in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        for pat, rep_pat in brightness_patterns:
            for w in WAKES:
                pool.append((f"{w}, {pat.format(v=v)}.",
                             dc("light", "set_brightness", v, rep_pat.format(v=v))))

    # Color temperature — absolute
    ct_patterns = [
        "set color temperature to level {v}",
        "color temperature level {v}",
        "set light warmth to {v}",
        "light level {v}",
        "color temp {v}",
    ]
    ct_replies = {
        1: "Daylight mode set.",
        2: "Cool neutral light set.",
        3: "Neutral light set.",
        4: "Warm light set.",
        5: "Candlelight mode set.",
    }
    for v in [1, 2, 3, 4, 5]:
        for pat in ct_patterns:
            for w in WAKES:
                pool.append((f"{w}, {pat.format(v=v)}.",
                             dc("light", "set_color_temp", v, ct_replies[v])))

    # Color temperature — relative warmer
    warmer_phrases = [
        "make the light warmer", "warmer light please", "light warmer",
        "make it cozier", "more cozy light", "a bit warmer light",
        "slightly warmer light", "warm up the light",
    ]
    for phrase in warmer_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("light", "set_color_temp", None, "Making the light warmer.")))

    # Color temperature — relative cooler
    cooler_phrases = [
        "make the light cooler", "cooler light please", "light cooler",
        "more daylight please", "a bit cooler light", "less warm light",
        "the light is too warm, cool it down",
    ]
    for phrase in cooler_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("light", "set_color_temp", None, "Making the light cooler.")))

    # RGB cycle
    rgb_phrases = [
        "start RGB mode", "activate color cycle", "RGB cycle on",
        "enable rainbow mode", "party mode on", "color cycle please",
        "start the color cycle", "turn on RGB", "rainbow light mode",
        "disco mode", "activate RGB", "start color mode",
    ]
    for phrase in rgb_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("light", "rgb_cycle", None, "RGB cycle started.")))

    # AC on
    ac_on_phrases = [
        "turn on the AC", "switch on the AC", "AC on", "air conditioner on",
        "start the AC", "activate the air conditioner",
        "turn on the air conditioning", "please turn on the AC",
        "switch on the air conditioner", "I need the AC on",
    ]
    for phrase in ac_on_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("ac", "turn_on", None, "AC is now on.")))

    # AC off
    ac_off_phrases = [
        "turn off the AC", "switch off the AC", "AC off", "air conditioner off",
        "stop the AC", "deactivate the air conditioner",
        "turn off the air conditioning", "please turn off the AC",
        "switch off the air conditioner",
    ]
    for phrase in ac_off_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("ac", "turn_off", None, "AC is now off.")))

    # AC temperature
    ac_temp_patterns = [
        ("set the AC to {v} degrees",      "Setting AC to {v} degrees."),
        ("set temperature to {v}",          "Temperature set to {v} degrees."),
        ("AC temperature {v}",              "AC at {v} degrees."),
        ("make it {v} degrees",             "Setting to {v} degrees."),
        ("cooling to {v}",                  "Cooling to {v} degrees."),
        ("adjust AC to {v}",                "AC adjusted to {v} degrees."),
        ("temperature {v} degrees",         "Temperature set to {v} degrees."),
        ("air conditioner to {v} degrees",  "Air conditioner set to {v} degrees."),
        ("set AC temperature to {v}",       "AC temperature set to {v} degrees."),
        ("{v} degrees please",              "Setting to {v} degrees."),
    ]
    for v in range(16, 31):
        for pat, rep_pat in ac_temp_patterns:
            for w in WAKES:
                pool.append((f"{w}, {pat.format(v=v)}.",
                             dc("ac", "set_temperature", v, rep_pat.format(v=v))))

    # Curtain open
    curtain_open_phrases = [
        "open the curtain", "open curtains", "curtain open",
        "pull back the curtain", "fully open the curtain",
        "draw back the curtain", "please open the curtain",
        "curtain up", "open up the curtain", "let in some light via the curtain",
    ]
    for phrase in curtain_open_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("curtain", "open", None, "Curtain opened.")))

    # Curtain close
    curtain_close_phrases = [
        "close the curtain", "close curtains", "curtain closed",
        "shut the curtain", "fully close the curtain", "draw the curtain",
        "please close the curtain", "curtain down",
        "close up the curtain", "pull the curtain closed",
    ]
    for phrase in curtain_close_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("curtain", "close", None, "Curtain closed.")))

    # Curtain set_position — numeric
    curtain_pos_patterns = [
        ("set curtain to {v} percent",    "Curtain set to {v}%."),
        ("curtain to {v} percent",        "Curtain at {v}%."),
        ("curtain at {v} percent",        "Curtain at {v}%."),
        ("move curtain to {v} percent",   "Curtain moved to {v}%."),
        ("position curtain at {v} percent", "Curtain positioned at {v}%."),
    ]
    for v in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        for pat, rep_pat in curtain_pos_patterns:
            for w in WAKES:
                pool.append((f"{w}, {pat.format(v=v)}.",
                             dc("curtain", "set_position", v, rep_pat.format(v=v))))

    # Curtain qualifiers
    curtain_qual = [
        ("open the curtain a little",           20, "Opening curtain slightly."),
        ("open curtain just a bit",             20, "Curtain opened a little."),
        ("curtain a little open",               20, "Curtain set to a small opening."),
        ("open the curtain a tiny bit",         20, "Curtain just barely opened."),
        ("open the curtain halfway",            50, "Curtain halfway open."),
        ("curtain halfway",                     50, "Curtain at halfway."),
        ("curtain to halfway",                  50, "Setting curtain to halfway."),
        ("half open the curtain",               50, "Curtain set to half."),
        ("open the curtain most of the way",    80, "Curtain mostly open."),
        ("curtain mostly open",                 80, "Curtain opened most of the way."),
        ("almost fully open the curtain",       80, "Curtain almost fully open."),
        ("open the curtain almost all the way", 80, "Curtain nearly fully opened."),
    ]
    for phrase, val, reply in curtain_qual:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("curtain", "set_position", val, reply)))

    # Window open
    window_open_phrases = [
        "open the window", "window open", "open up the window",
        "please open the window", "let some air in", "crack the window open",
        "ventilate the room",
    ]
    for phrase in window_open_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("window", "open", None, "Window opened.")))

    # Window close
    window_close_phrases = [
        "close the window", "window closed", "shut the window",
        "please close the window", "close up the window", "seal the window",
    ]
    for phrase in window_close_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.",
                         dc("window", "close", None, "Window closed.")))

    return pool


# ── needs_clarification pool ─────────────────────────────────────────────────

def build_clarification_pool():
    pool = []

    cold_q    = "Would you like me to close the window or raise the AC temperature?"
    cold_opts = ["close_window", "raise_ac_temperature"]
    hot_q     = "Would you like me to open the window or lower the AC temperature?"
    hot_opts  = ["open_window", "lower_ac_temperature"]
    dark_q    = "Would you like me to turn on the light or open the curtain?"
    dark_opts = ["turn_on_light", "open_curtain"]
    bright_q  = "Would you like me to dim the light or close the curtain?"
    bright_opts = ["dim_light", "close_curtain"]

    cold_phrases = [
        "I feel cold", "it is cold", "it's cold in here", "I feel a bit chilly",
        "it's chilly", "it's freezing", "the room is too cold", "I'm cold",
        "this place is freezing", "I need some warmth", "it's pretty cold here",
        "I feel chilly", "the temperature is quite low", "the room is chilly",
        "I can't get warm", "too cold for me", "I'm shivering",
        "I need to warm up", "the room is a bit cold",
        "it's cooler than I'd like", "I'm a little cold",
        "it feels cold to me", "this place could be warmer",
        "it's rather cold today", "the air feels cold",
    ]
    hot_phrases = [
        "I feel hot", "it's hot in here", "it's too warm", "it is quite warm",
        "the room is stuffy", "I'm sweating", "I need to cool down",
        "it's sweltering", "it's really warm in here", "I need to cool off",
        "the temperature is too high", "I feel overheated", "it's humid and warm",
        "too hot for me", "this room is very warm", "I'm feeling the heat",
        "it's muggy in here", "the room needs cooling", "I'm a bit warm",
        "it's quite warm here", "I feel a bit overheated",
        "the room is quite warm", "I'm uncomfortably warm",
        "I need cooler air", "the air is warm and heavy",
    ]
    dark_phrases = [
        "it's a bit dark", "it's dark in here", "I can barely see",
        "the room is too dark", "it's gloomy in here", "I need more light",
        "the room feels dim", "there's not enough light", "I can't see well",
        "this room is shadowy", "it's rather dark", "I need better lighting",
        "the room is poorly lit", "it's quite dark", "I need some brightness",
        "hard to see in here", "it's murky in here", "not enough light here",
        "it's dim and hard to see", "I could use more light",
        "the room is a bit dim", "I need brighter conditions",
        "it's getting dark", "the room lacks light", "visibility is low",
    ]
    bright_phrases = [
        "it's too bright", "the light is too strong", "there's too much light",
        "it's blinding in here", "the room is too bright", "the light is harsh",
        "it's glaring in here", "too much brightness", "the light is overwhelming",
        "the light hurts my eyes", "this is too bright for me",
        "the room is overly lit", "it's very bright", "the brightness is excessive",
        "I need less light", "reduce the light please",
        "bright lights are bothering me", "it's uncomfortably bright",
        "the room has too much light", "I'm blinded by the light",
        "it's excessively bright", "I need dimmer conditions",
        "the brightness is too much", "the room is over-illuminated",
        "the lighting is intense",
    ]

    for phrase in cold_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(cold_q, cold_opts)))
    for phrase in hot_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(hot_q, hot_opts)))
    for phrase in dark_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(dark_q, dark_opts)))
    for phrase in bright_phrases:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(bright_q, bright_opts)))

    # Ambiguous device
    amb_open_q    = "Do you want me to open the window or the curtain?"
    amb_open_opts = ["open_window", "open_curtain"]
    amb_close_q   = "Do you want me to close the window or the curtain?"
    amb_close_opts = ["close_window", "close_curtain"]
    amb_on_q    = "Do you want me to turn on the light or the AC?"
    amb_on_opts = ["turn_on_light", "turn_on_ac"]
    amb_off_q   = "Do you want me to turn off the light or the AC?"
    amb_off_opts = ["turn_off_light", "turn_off_ac"]

    for phrase in ["open it", "can you open it", "I want it open",
                   "please open it", "open that for me",
                   "open the thing", "open it up", "it should be open"]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(amb_open_q, amb_open_opts)))
    for phrase in ["close it", "can you close it", "I want it closed",
                   "please close it", "close that for me",
                   "shut it", "it should be closed"]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(amb_close_q, amb_close_opts)))
    for phrase in ["turn it on", "can you turn it on", "I want it on",
                   "switch it on", "power it on", "activate it", "start it"]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(amb_on_q, amb_on_opts)))
    for phrase in ["turn it off", "can you turn it off", "I want it off",
                   "switch it off", "power it off", "deactivate it", "stop it"]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(amb_off_q, amb_off_opts)))

    # Vague / atmospheric
    vague_q    = "Would you like me to adjust lighting or temperature?"
    vague_opts = ["adjust_lighting", "adjust_temperature"]
    lively_q    = "Would you like me to turn on the RGB cycle or open the curtain?"
    lively_opts = ["rgb_cycle", "open_curtain"]
    discomfort_q    = "Could you describe what feels uncomfortable — temperature, lighting, or something else?"
    discomfort_opts = ["adjust_temperature", "adjust_lighting"]

    for phrase in [
        "make it better", "I need adjustments", "can you improve the environment",
        "something feels off", "the environment isn't right",
        "can you optimize the room", "something needs to change",
        "this room needs improvement", "make the room nicer",
        "make the atmosphere better", "the room doesn't feel right",
        "I need the room improved", "the room is not to my liking",
        "please make it better", "can you fix the room",
        "make this room more comfortable", "the conditions aren't ideal",
        "this place could be better", "something isn't quite right",
        "the room needs a change",
    ]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(vague_q, vague_opts)))

    for phrase in [
        "make this room lively", "spice up this room", "make it more energetic",
        "add some energy to the room", "liven up the space",
        "make the room more dynamic", "create a fun atmosphere",
        "make it vibrant", "energize the room", "give the room some life",
        "make this place more lively", "create a lively atmosphere",
        "make this room feel more active", "bring some energy in here",
        "make this space more engaging",
    ]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(lively_q, lively_opts)))

    for phrase in [
        "I'm not comfortable", "I feel uncomfortable", "I'm not at ease",
        "I don't feel right", "something is making me uncomfortable",
        "I feel a bit off", "I'm not feeling great in here",
        "this room doesn't suit me right now",
    ]:
        for w in WAKES:
            pool.append((f"{w}, {phrase}.", nc(discomfort_q, discomfort_opts)))

    return pool


# ── general_qa pool ───────────────────────────────────────────────────────────

def build_qa_pool():
    pool = []

    qa_pairs = [
        ("how do I boil an egg",
         "Place the egg in boiling water for 6 to 10 minutes depending on how firm you want it."),
        ("how long should I cook pasta",
         "Most pasta takes 8 to 12 minutes. Check the package for exact timing."),
        ("how do I make tea",
         "Boil water, pour over a tea bag or leaves, and steep for a few minutes."),
        ("how do I store leftovers",
         "Use an airtight container and refrigerate within two hours of cooking."),
        ("how long can cooked chicken stay in the fridge",
         "Cooked chicken is safe in the fridge for 3 to 4 days."),
        ("how do I make rice",
         "Rinse the rice, then cook with water at a 1 to 2 ratio until absorbed."),
        ("how do I make scrambled eggs",
         "Whisk eggs, cook on low heat while stirring gently until soft."),
        ("how do I make coffee",
         "Add ground coffee to a filter and pour hot water over it slowly."),
        ("how do I wash fruits",
         "Rinse under running water and gently rub the surface before eating."),
        ("how many calories are in an egg",
         "A large egg contains about 70 to 80 calories."),
        ("can I eat this dish after a night in the fridge",
         "Yes, most cooked food is safe for up to 3 to 4 days in the fridge."),
        ("is it safe to reheat leftover rice",
         "Yes, heat it until steaming hot. Avoid reheating more than once."),
        ("how long does milk last in the fridge",
         "About 5 to 7 days after opening if kept below 4 degrees Celsius."),
        ("how do I cut an onion without crying",
         "Chill the onion before cutting or use a sharp knife to reduce irritation."),
        ("how do I keep food warm",
         "Cover it and use low heat or an insulated container."),
        ("how do I store bread",
         "Keep in a cool dry place or freeze for longer storage."),
        ("how do I keep vegetables fresh",
         "Refrigerate them and keep them dry in breathable bags."),
        ("how do I make a salad",
         "Chop your vegetables, add toppings, and dress with oil and vinegar."),
        ("how long do apples last",
         "Apples keep for about 4 to 6 weeks in the refrigerator."),
        ("how do I ripen a banana faster",
         "Place it in a paper bag or near other ripe fruit."),
        ("how do I make soup",
         "Simmer vegetables and protein in broth and season to taste."),
        ("how do I make oatmeal",
         "Combine oats with water or milk, heat, and add your favorite toppings."),
        ("what is a good breakfast",
         "A good breakfast includes protein, whole grains, and fruit or vegetables."),
        ("how do I make pancakes",
         "Mix flour, eggs, milk, and a pinch of salt, then cook on a lightly oiled pan."),
        ("how do I make lemonade",
         "Mix lemon juice, water, and sugar to taste, then chill and serve."),
        ("how do I clean a microwave",
         "Heat water with lemon inside for a few minutes, then wipe with a cloth."),
        ("how do I clean the floor",
         "Vacuum first, then mop with appropriate cleaner and warm water."),
        ("how often should I do laundry",
         "Most people do laundry once or twice a week depending on usage."),
        ("how do I remove odor from the fridge",
         "Clean the inside and place baking soda to absorb odors."),
        ("how do I unclog a sink",
         "Try a plunger or baking soda and vinegar followed by hot water."),
        ("how do I clean glass windows",
         "Use glass cleaner or vinegar solution with a microfiber cloth."),
        ("how do I get rid of dust",
         "Wipe surfaces with a damp cloth and vacuum regularly."),
        ("how do I clean a sofa",
         "Vacuum the surface and use a fabric cleaner suitable for your sofa material."),
        ("how do I remove stains from carpet",
         "Blot the stain and apply carpet cleaner or mild detergent."),
        ("how do I clean a bathroom",
         "Use disinfectant on surfaces, scrub thoroughly, and rinse."),
        ("how do I prevent mold",
         "Keep areas dry, ventilate well, and clean moisture-prone surfaces."),
        ("how do I clean a coffee maker",
         "Run a cycle with equal parts water and white vinegar, then follow with plain water."),
        ("how do I clean a cutting board",
         "Wash with hot soapy water and occasionally disinfect with diluted bleach."),
        ("how do I remove grease from the stovetop",
         "Use dish soap and a scrubbing pad, then wipe clean."),
        ("how do I keep my room cool",
         "Close curtains during the day and use fans for air circulation."),
        ("how do I keep my room warm",
         "Close windows, use insulating curtains, and maintain steady heating."),
        ("how do I save electricity at home",
         "Turn off unused devices and reduce heating or cooling usage."),
        ("how do I reduce noise at home",
         "Use rugs, curtains, and soft materials to absorb sound."),
        ("how do I organize my room",
         "Declutter, group similar items together, and use storage containers."),
        ("how do I dry clothes faster",
         "Use proper airflow and hang clothes with space between them."),
        ("how do I make my room brighter",
         "Open curtains, use lighter colors, and add more light sources."),
        ("how do I make my room cozier",
         "Add warm lighting, soft textiles, and comfortable seating."),
        ("how do I reduce humidity in the room",
         "Use a dehumidifier and ensure good ventilation."),
        ("what time is it",
         "I do not have access to real-time data. Please check your phone or clock."),
        ("what is the capital of France",
         "The capital of France is Paris."),
        ("tell me a joke",
         "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("is it going to rain today",
         "I do not have weather access. Please check a weather app."),
        ("can I drink tap water",
         "It depends on your location. Check your local water quality report."),
        ("what is your name",
         "My name is Cathey, your smart home assistant."),
        ("who made you",
         "I was built by a student team as a smart home voice assistant."),
        ("what can you do",
         "I can control lights, the AC, curtains, and windows, and answer general questions."),
        ("what is 2 plus 2",
         "2 plus 2 equals 4."),
        ("how do I stay hydrated",
         "Drink about 8 glasses of water per day and eat water-rich foods."),
        ("how do I sleep better",
         "Keep a consistent schedule, limit screen time before bed, and keep the room cool."),
        ("how do I focus while studying",
         "Remove distractions, take short breaks, and study in a well-lit space."),
        ("how do I relieve stress",
         "Try deep breathing, light exercise, or a short walk outside."),
        ("what is a balanced diet",
         "A balanced diet includes fruits, vegetables, proteins, and whole grains."),
        ("how do I improve my posture",
         "Sit straight, take breaks from sitting, and strengthen your core muscles."),
        ("how do I save money",
         "Track your spending, set a budget, and avoid impulse purchases."),
        ("how do I read more books",
         "Set aside a regular reading time each day, even just 20 minutes."),
        ("how do I be more productive",
         "Prioritize tasks, minimize distractions, and take regular breaks."),
        ("what is photosynthesis",
         "Plants convert sunlight, water, and carbon dioxide into energy and oxygen."),
        ("how does a refrigerator work",
         "It uses a refrigerant cycle to absorb heat from inside and release it outside."),
        ("how does Wi-Fi work",
         "It transmits data wirelessly using radio waves between a router and devices."),
        ("what is the speed of light",
         "The speed of light is approximately 299,792 kilometers per second."),
        ("how many planets are in the solar system",
         "There are 8 planets in our solar system."),
        ("what is the boiling point of water",
         "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("how do I improve my English",
         "Practice daily, read English content, and speak whenever you get the chance."),
        ("how do I wake up early",
         "Set a consistent bedtime, place the alarm across the room, and sleep enough."),
        ("how do I reduce food waste",
         "Plan meals, use leftovers creatively, and store food properly."),
        ("how long should I exercise each day",
         "Aim for at least 30 minutes of moderate activity most days."),
        ("what is the largest ocean",
         "The Pacific Ocean is the largest ocean on Earth."),
        ("who was Albert Einstein",
         "Albert Einstein was a physicist best known for the theory of relativity."),
        ("what is gravity",
         "Gravity is the force that attracts objects toward each other, proportional to their mass."),
        ("how do computers work",
         "Computers process binary data using logic circuits and execute stored instructions."),
        ("how do I meditate",
         "Sit comfortably, close your eyes, and focus on your breath for a few minutes."),
        ("what is climate change",
         "Climate change refers to long-term shifts in global temperatures and weather patterns."),
        ("how do I plant a seed",
         "Place the seed in moist soil at the appropriate depth and keep it in sunlight."),
        ("how do I take care of a plant",
         "Water regularly, provide adequate sunlight, and ensure good drainage."),
        ("what is machine learning",
         "Machine learning is a method where computers learn patterns from data to make decisions."),
        ("how do I write a good essay",
         "Start with a clear thesis, organize your supporting points, and revise carefully."),
        ("what is the speed of sound",
         "Sound travels at about 343 meters per second in air at room temperature."),
        ("how do I reduce screen time",
         "Set usage limits, designate phone-free times, and replace scrolling with activities."),
        ("how do I deal with a cold",
         "Rest, stay hydrated, and take over-the-counter remedies as needed."),
        ("how do I remember things better",
         "Use spaced repetition, associate new info with what you know, and sleep well."),
        ("how do I make my bed properly",
         "Straighten the sheets, fluff the pillows, and smooth out any wrinkles."),
        ("how do I improve my handwriting",
         "Practice regularly, slow down, and focus on consistent letter shapes."),
        ("what is the largest country in the world",
         "Russia is the largest country in the world by land area."),
        ("how do I change a light bulb",
         "Turn off the power, wait for the bulb to cool, then twist it out and insert a new one."),
        ("how do I charge my phone faster",
         "Use a fast charger, enable airplane mode, and avoid using the phone while charging."),
        ("what is a smart home",
         "A smart home uses connected devices that can be controlled automatically or by voice."),
        ("how does voice recognition work",
         "Voice recognition converts speech into text using acoustic and language models."),
    ]

    qa_prefixes = [
        "{w}, {q}?",
        "{w}, can you tell me {q}?",
        "{w}, quick question: {q}?",
        "{w}, do you know {q}?",
        "{w}, please tell me {q}.",
        "{w}, any idea {q}?",
        "{w}, I want to know {q}.",
        "{w}, {q}, do you know?",
    ]

    for q, answer in qa_pairs:
        for prefix in qa_prefixes:
            for w in WAKES:
                pool.append((prefix.format(w=w, q=q), qa(answer)))

    return pool


# ── invalid pool ──────────────────────────────────────────────────────────────

def build_invalid_pool():
    templates = [
        "Hello.", "Turn on the light.", "How are you today?", "It is cold.",
        "What is going on?", "Open the window.", "Hey, can you help me?",
        "I feel hot.", "Thanks.", "Bob, turn on the light.", "Siri, turn on the light.",
        "No, I feel cold.", "Alexa, lights on.", "Hey Google, close the curtain.",
        "Set the AC to 22.", "Turn off everything.", "Um.", "Hmm.", "What?",
        "Never mind.", "Hi.", "Okay.", "Close the curtain.", "Open the curtain.",
        "Set brightness to 50.", "I need some light.", "It's cold.", "AC off.",
        "Lights off.", "Window open.", "Please.", "Yes.", "No.", "Uh.",
        "Echo, dim the lights.", "Jarvis, set temperature to 24.", "Computer, RGB mode.",
        "Hey, turn it on.", "Turn it off.", "Make it brighter.", "Make it warmer.",
        "I'm cold.", "Help.", "Stop.", "Go.", "Good morning.", "Good night.",
        "Whatever.", "Forget it.", "Yeah.", "Nope.", "Light on.", "AC on.",
        "Curtain open.", "Window close.", "Brightness 70.", "RGB cycle.",
        "Temperature 22.", "Hey.", "Okay then.", "All right.", "Mm.", "Ah.",
        "Oh.", "I see.", "Got it.", "Fine.", "Sure.",
        "Cortana, lights off.", "Amazon, temperature 24.",
        "Nova, turn on the light.", "Hey Nova, AC on.", "Nova, close the curtain.",
        "Nova, I feel cold.", "Nova, brightness 50.", "Nova, open the window.",
    ]
    return [(t, INVALID_JSON) for t in templates]


# ── sampling helper ───────────────────────────────────────────────────────────

def fill_to_target(pool, target):
    if len(pool) >= target:
        return RNG.sample(pool, target)
    result = list(pool)
    while len(result) < target:
        result.extend(pool)
    RNG.shuffle(result)
    return result[:target]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building pools...")
    direct_pool  = build_direct_pool()
    clarify_pool = build_clarification_pool()
    qa_pool      = build_qa_pool()
    invalid_pool = build_invalid_pool()

    print(f"  direct:  {len(direct_pool)}")
    print(f"  clarify: {len(clarify_pool)}")
    print(f"  qa:      {len(qa_pool)}")
    print(f"  invalid: {len(invalid_pool)}")

    direct_data  = fill_to_target(direct_pool,  7600)
    clarify_data = fill_to_target(clarify_pool, 6500)
    qa_data      = fill_to_target(qa_pool,      6000)
    invalid_data = fill_to_target(invalid_pool, 2400)

    all_data = direct_data + clarify_data + qa_data + invalid_data
    RNG.shuffle(all_data)
    print(f"Total examples: {len(all_data)}")

    out_path = Path(__file__).parent / "train_data.py"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('"""\n')
        f.write('Training data for Cathey LoRA fine-tuning.\n\n')
        f.write('RAW_TRAIN_DATA — list of (user_input, expected_json_output) tuples.\n')
        f.write('build_dataset() — formats into a HuggingFace Dataset ready for SFTTrainer.\n\n')
        f.write(f'Auto-generated by generate_train_data.py — {len(all_data)} examples.\n')
        f.write('"""\n\n')
        f.write('from datasets import Dataset\n')
        f.write('from llm_parser import UNIFIED_SYSTEM_PROMPT\n\n')
        f.write('RAW_TRAIN_DATA = [\n')
        for utterance, json_str in all_data:
            f.write(f'    ({repr(utterance)},\n')
            f.write(f'     {repr(json_str)}),\n')
        f.write(']\n\n\n')
        f.write('def build_dataset(tokenizer) -> Dataset:\n')
        f.write('    """\n')
        f.write('    Format RAW_TRAIN_DATA into a HuggingFace Dataset using the model\'s chat template.\n')
        f.write('    SFTTrainer computes loss only on the assistant turn.\n')
        f.write('    """\n')
        f.write('    def _fmt(inp: str, out: str) -> str:\n')
        f.write('        msgs = [\n')
        f.write('            {"role": "system",    "content": UNIFIED_SYSTEM_PROMPT},\n')
        f.write('            {"role": "user",      "content": f\'Text: "{inp}"\\nReturn JSON only.\'},\n')
        f.write('            {"role": "assistant", "content": out},\n')
        f.write('        ]\n')
        f.write('        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)\n\n')
        f.write('    return Dataset.from_dict({"text": [_fmt(i, o) for i, o in RAW_TRAIN_DATA]})\n')

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
