# Ref: Natural Language Processing in Action
# https://github.com/totalgood/nlpia
# https://www.manning.com/books/natural-language-processing-in-action
import os
import re
from collections import Counter
from itertools import permutations

import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import ipdb


DATA_PATH = os.path.expanduser("~/PycharmProjects/nlpia-master/src/nlpia/data/")


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
    # p.35
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

    # Listing 2.2: One-hot vector sequence for the Monticello sentence, p.36
    print()
    print(pd.DataFrame(onehot_vectors, columns=vocab))

    # Listing 2.3: Prettier one-hot vectors, p.36
    print()
    df = pd.DataFrame(onehot_vectors, columns=vocab)
    df[df == 0] = ''
    print(df)

    # Listing 2.5: Construct a DataFrame of bag-of-words vectors, p.41
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


# 2.2.1: Dot product, p.41
def _2_2_1():
    # Listing 2.5: Example dot product calculation, p.42
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 3, 4])
    print(v1.dot(v2))
    print((v1 * v2).sum())
    print(sum([x1 * x2 for x1, x2 in zip(v1, v2)]))
    print(v1.reshape(-1, 1).T @ v2.reshape(-1, 1))


# 2.2.2: Measuring bag-of-words overlap, p.42
def _2_2_2():
    # From 2.2, p.41
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
    df = df[sorted(df)]

    # Listing 2.6: Overlap of word counts for two bag-of-words vectors, p.42
    df = df.T
    print(df.sent0.dot(df.sent1))
    print(df.sent0.dot(df.sent2))
    print(df.sent0.dot(df.sent3))

    # Find the word that is shared by sent0 and sent3, p.43
    print([(k, v) for (k, v) in (df.sent0 & df.sent3).items() if v])


# 2.2.3: A token improvement, p.43
def _2_2_3():
    # Listing 2.7: Tokenize the Monticello sentence with a regular expression, p.43
    sentence = """Thomas Jefferson began building Monticello at the age of 26."""
    tokens = re.split(r'[-\s.,;!?]+', sentence)
    print(tokens)

    print()

    # When to compile your regex patterns, p.45
    pattern = re.compile(r"([-\s.,;!?])+")
    tokens = pattern.split(sentence)
    print(tokens[-10:])  # just the last 10 tokens

    print()

    # Filter the whitespace and punctuation characters, p.45
    print([x for x in tokens if x and x not in '- \t\n.,;!?'])
    # Another method with lambda and filter()
    print(list(filter(lambda x: x if x and x not in '- \t\n.,;!?' else None, tokens)))

    print()

    # Use the NLTK function RegexpTokenizer to replicate your simple tokenizer example, p.46
    tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
    print(tokenizer.tokenize(sentence))

    print()

    # An even better tokenizer is the Treebank Word Tokenizer from the NLTK package, p.47
    sentence = "Monticello wasn't designated as UNESCO World Heritage Site until 1987."
    tokenizer = TreebankWordTokenizer()
    print(tokenizer.tokenize(sentence))

    print()

    # Tokenize informal text from social networks such as Twitter and Facebook, p.48
    message = "RT @TJMonticello Best day everrrrrrr at Monticello. Awesommmmmmeeeeeeee day :*)"
    print(casual_tokenize(message))
    print(casual_tokenize(message, reduce_len=True, strip_handles=True))


# 2.2.4: Extending your vocabulary with n-grams, 48
def _2_2_4():
    # From 2.2.3, p.45
    sentence = """Thomas Jefferson began building Monticello at the age of 26."""
    pattern = re.compile(r"([-\s.,;!?])+")
    tokens = pattern.split(sentence)
    tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
    print(tokens)

    print()

    # n-gram tokenizer from NLTK, p.50
    two_grams = list(ngrams(tokens, 2))
    print(two_grams)
    print(list(ngrams(tokens, 3)))
    print([" ".join(x) for x in two_grams])

    print()

    # Listing 2.8: NLTK list of stop words, p.53
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    print(len(stop_words))
    print(stop_words[:7])
    print([sw for sw in stop_words if len(sw) == 1])

    print()

    # Listing 2.9: NLTK list of stop words, p.54
    print(len(sklearn_stop_words))
    print(len(stop_words))
    print(len(set(stop_words).union(sklearn_stop_words)))
    print(len(set(stop_words).intersection(sklearn_stop_words)))


# 2.2.5: Normalizing your vocabulary, p.54
def _2_2_5():
    # Normalize the capitalization of your tokens, p.55
    tokens = ['House', 'Visitor', 'Center']
    normalized_tokens = [x.lower() for x in tokens]
    print(normalized_tokens)

    print()

    # Stemming, p.57
    words = ['Houses', 'house', 'housing', 'generously']
    # Porter stemmer
    stemmer = PorterStemmer()
    print([stemmer.stem(word) for word in words])
    # Snowball stemmer
    stemmer = SnowballStemmer("english")
    print([stemmer.stem(word) for word in words])

    print()

    # Regex-based stemmer, p.58
    def stem(phrase):
        return ' '.join([re.findall('^(.*ss|.*?)(s)?$', word)[0][0].strip("'") for word in phrase.lower().split()])

    print(stem('houses'))
    print(stem("Doctor House's calls"))

    print()

    # Porter stemmer, p.58
    stemmer = PorterStemmer()
    print(' '.join([stemmer.stem(w).strip("'") for w in "dish washer's washed dishes".split()]))

    print()

    # Lemmatizer, p.61
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    print(lemmatizer.lemmatize("better"))
    print(lemmatizer.lemmatize("better", pos="a"))
    print(lemmatizer.lemmatize("good", pos="a"))
    print(lemmatizer.lemmatize("goods", pos="a"))
    print(lemmatizer.lemmatize("goods", pos="n"))
    print(lemmatizer.lemmatize("goodness", pos="n"))
    print(lemmatizer.lemmatize("best", pos="a"))


# 2.3.1: VADER—A rule-based sentiment analyzer, p.64
def _2_3_1():
    # VADER, p.64
    sa = SentimentIntensityAnalyzer()
    print(sa.polarity_scores(text="Python is very readable and it's great for NLP."))
    print(sa.polarity_scores(text="Python is not a bad choice for most applications."))

    print()

    # VADER applied on other example statements, p.65
    corpus = ["Absolutely perfect! Love it! :-) :-) :-)",
              "Horrible! Completely useless. :(",
              "It was OK. Some good and some bad things."]
    for doc in corpus:
        scores = sa.polarity_scores(doc)
        print('{:+}: {}'.format(scores['compound'], doc))


# Ref.: https://github.com/totalgood/nlpia/blob/73c03f651e54e945f9a7eebe4714095dc3e5609a/src/nlpia/futil.py#L360
def looks_like_index(series, index_names=('Unnamed: 0', 'pk', 'index', '')):
    """ Tries to infer if the Series (usually leftmost column) should be the index_col
    >>> looks_like_index(pd.Series(np.arange(100)))
    True
    """
    if series.name in index_names:
        return True
    if (series == series.index.values).all():
        return True
    if (series == np.arange(len(series))).all():
        return True
    if (
        (series.index == np.arange(len(series))).all() and
        str(series.dtype).startswith('int') and
        (series.count() == len(series))
    ):
        return True
    return False


# Ref.: https://github.com/totalgood/nlpia/blob/73c03f651e54e945f9a7eebe4714095dc3e5609a/src/nlpia/futil.py#L381
def read_csv(*args, **kwargs):
    """Like pandas.read_csv, only little smarter: check left column to see if it should be the index_col
    """
    kwargs.update({'low_memory': False})
    if isinstance(args[0], pd.DataFrame):
        df = args[0]
    else:
        print('Reading CSV with `read_csv(*{}, **{})`...'.format(args, kwargs))
        df = pd.read_csv(*args, **kwargs)
    if looks_like_index(df[df.columns[0]]):
        df = df.set_index(df.columns[0], drop=True)
        if df.index.name in ('Unnamed: 0', ''):
            df.index.name = None
    if ((str(df.index.values.dtype).startswith('int') and (df.index.values > 1e9 * 3600 * 24 * 366 * 10).any()) or
            (str(df.index.values.dtype) == 'object')):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
            print('Unable to coerce DataFrame.index into a datetime using pd.to_datetime([{},...])'.format(
                df.index.values[0]))
    return df


# 2.3.2: Naive Bayes, p.65
def _2_3_2():
    # Load movie reviews dataset, p.66
    data_path = os.path.join(DATA_PATH, "hutto_ICWSM_2014/movieReviewSnippets_GroundTruth.csv.gz")
    movies = read_csv(data_path, nrows=None)
    print(movies.head().round(2))
    print()
    print(movies.describe().round(2))


if __name__ == '__main__':
    _2_3_2()
