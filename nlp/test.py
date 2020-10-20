# Ref: Natural Language Processing in Action
# https://github.com/totalgood/nlpia
# https://www.manning.com/books/natural-language-processing-in-action
import re
from collections import Counter
from itertools import permutations

import numpy as np
import pandas as pd

import ipdb


# 1.4.3: A simple chatbot, p.13
def _1_4_3():
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
def _1_4_4():
    print(Counter("Guten Morgen Rosa".split()))
    print(Counter("Good morning, Rosa!".split()))


# 1.6: Word order and grammar, p.21
def _1_6():
    p = [" ".join(combo) for combo in
         permutations("Good morning Rosa!".split(), 3)]
    print(p)

    np.arange(1, 12 + 1).prod()  # factorial(12) = arange(1, 13).prod()


# 2.2: Building your vocabulary with a tokenizer, p.35
def _2_2():
    sentence = """Thomas Jefferson began building Monticello at the age of 26."""
    token_sequence = str.split(sentence)
    vocab = sorted(set(token_sequence))
    num_tokens = len(token_sequence)
    vocab_size = len(vocab)
    onehot_vectors = np.zeros((num_tokens, vocab_size), int)

    for i, word in enumerate(token_sequence):
        onehot_vectors[i, vocab.index(word)] = 1

    print(' '.join(vocab))
    print()
    print(onehot_vectors)

    print()
    print(pd.DataFrame(onehot_vectors, columns=vocab))

    print()
    df = pd.DataFrame(onehot_vectors, columns=vocab)
    df[df == 0] = ''
    print(df)

    print()
    sentences = "Thomas Jefferson began building Monticello at the age of 26.\n"
    sentences += "Construction was done mostly by local masons and carpenters.\n"
    sentences += "He moved into the South Pavilion in 1770.\n"
    sentences += "Turning Monticello into a neoclassical masterpiece was " \
                 "Jefferson's obsession."
    corpus = {}
    for i, sent in enumerate(sentences.split('\n')):
        corpus['sent{}'.format(i)] = dict((tok, 1) for tok in sent.split())
    df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
    # Sort column names
    # Ref.: https://bit.ly/3oig94m (stackoverflow)
    # df = df.reindex(sorted(df.columns), axis=1)  # Method 1
    # df.sort_index(axis=1, inplace=True)  # Method 2
    df = df[sorted(df)]  # Method 3
    print(df[df.columns[:10]])
    # ipdb.set_trace()


def _2_2_1():
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 3, 4])
    print(v1.dot(v2))
    print((v1 * v2).sum())
    print(sum([x1 * x2 for x1, x2 in zip(v1, v2)]))
    print(v1.reshape(-1, 1).T @ v2.reshape(-1, 1))


if __name__ == '__main__':
    _2_2_1()
