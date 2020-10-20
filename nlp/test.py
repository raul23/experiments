# Ref: Natural Language Processing in Action
import re
from collections import Counter
from itertools import permutations

import numpy as np


# 1.4.3: A simple chatbot, p.13
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|" \
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"
re_greeting = re.compile(r, flags=re.IGNORECASE)
print(re_greeting.match('Hello Rosa'))
print(re_greeting.match('Hello Rosa').groups())
print(re_greeting.match("Good morning Rosa"))
print(re_greeting.match("Good Manning Rosa"))
print(re_greeting.match('Good evening Rosa Parks').groups())
print(re_greeting.match("Good Morn'n Rosa"))
print(re_greeting.match("yo Rosa"))

my_names = set(['rosa', 'rose', 'chatty', 'chatbot', 'bot', 'chatterbot'])
curt_names = set(['hal', 'you', 'u'])
greeter_name = 'Marc'
match = re_greeting.match(input())

if match:
    at_name = match.groups()[-1]
    if at_name in curt_names:
        print("Good one.")
    elif at_name.lower() in my_names:
        print("Hi {}, How are you?".format(greeter_name))

print()

# 1.4.4: Another way, p.18
print(Counter("Guten Morgen Rosa".split()))
print(Counter("Good morning, Rosa!".split()))

# 1.6: Word order and grammar, p.21
p = [" ".join(combo) for combo in
     permutations("Good morning Rosa!".split(), 3)]
print(p)

np.arange(1, 12 + 1).prod()  # factorial(12) = arange(1, 13).prod()

# 2.2: Building your vocabulary with a tokenizer, p.35
sentence = """Thomas Jefferson began building Monticello at the age of 26."""
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
