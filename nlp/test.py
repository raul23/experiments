# Ref: Natural Language Processing in Action
import re
from collections import Counter


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

print()

# p.
print(Counter("Guten Morgen Rosa".split()))
print(Counter("Good morning, Rosa!".split()))
