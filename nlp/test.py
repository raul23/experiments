# Ref: Natural Language Processing in Action
# https://github.com/totalgood/nlpia
# https://www.manning.com/books/natural-language-processing-in-action
import copy
import os
import string
import re
from collections import Counter, OrderedDict
from itertools import permutations

import nltk
import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.tokenize.casual import casual_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from futil import read_csv

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


# 2.3.2: Naive Bayes, p.65
def _2_3_2():
    # Load movie review dataset, p.66
    data_path = os.path.join(DATA_PATH, "hutto_ICWSM_2014/movieReviewSnippets_GroundTruth.csv.gz")
    movies = read_csv(data_path, nrows=None)
    print(movies.head().round(2))
    print()
    print(movies.describe().round(2))

    print()

    # For debugging, use few examples
    # movies = movies[:100]

    # Create Pandas DataFrame BoW from these movie review texts, p.66
    pd.set_option('display.width', 75)
    bags_of_words = []
    for i, text in enumerate(movies.text):
        print(i)
        bags_of_words.append(Counter(casual_tokenize(text)))
    df_bows = pd.DataFrame.from_records(bags_of_words)
    df_bows = df_bows.fillna(0).astype(int)
    print(df_bows.shape)
    print(df_bows.head())
    print(df_bows.head()[list(bags_of_words[0].keys())])

    print()

    # Train a Naive Bayes model on movies, p.67
    nb = MultinomialNB()
    nb = nb.fit(df_bows, movies.sentiment > 0)
    # If working in the [-4, 4] range for the predicted values
    # movies['predicted_sentiment'] = nb.predict_proba(df_bows)[:, 1] * 8 - 4
    # If working with -4 or 4 for the predicted values
    movies['predicted_sentiment'] = np.where(nb.predict(df_bows) is True, 4, -4)
    movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
    print(movies.error.mean().round(1))
    movies['sentiment_ispositive'] = (movies.sentiment > 0).astype(int)
    movies['predicted_ispositive'] = (movies.predicted_sentiment > 0).astype(int)
    print(movies['''sentiment predicted_sentiment sentiment_ispositive predicted_ispositive'''.split()].head(8))
    print((movies.predicted_ispositive == movies.sentiment_ispositive).sum() / len(movies))


# 3.1: Bag of words, p.71
def _3_1():
    # Tokenize sentence, p.72
    sentence = """The faster Harry got to the store, the faster Harry, the faster, would get home."""
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(sentence.lower())
    print(tokens)

    print()

    # Count occurrences of words, p.72
    bag_of_words = Counter(tokens)
    print(bag_of_words)
    print(bag_of_words.most_common(4))

    print()

    # Calculate the term frequency of “harry”, p.73
    times_harry_appears = bag_of_words['harry']
    num_unique_words = len(bag_of_words)
    tf = times_harry_appears / num_unique_words
    print(round(tf, 4))

    # Tokenize kite text and compute term occurrences, p.75
    from data import kite_text
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(kite_text.lower())
    token_counts = Counter(tokens)
    print(token_counts)

    print()

    # Using spaCy to tokenize kite text
    # IMPORTANT: man-lifting is split into man and lifting
    # Same for free-falling and lighter-than-air
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(kite_text)
    tokens_spacy = [token.text for token in doc]
    token_counts_spacy = Counter(tokens_spacy)
    print(token_counts_spacy)

    print()

    # Remove stopwords from kite text, p.75
    nltk.download('stopwords', quiet=True)
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [x for x in tokens if x not in stopwords]
    kite_counts = Counter(tokens)
    print(kite_counts)


# 3.2: Vectorizing, p.76
def _3_2():
    # Previous code: 3.1, from p.75
    from data import kite_text
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(kite_text.lower())

    # Remove punctuation from tokens
    tokens_without_punctuation = [token for token in tokens if token not in string.punctuation]
    set1 = set(tokens)
    set2 = set(tokens_without_punctuation)
    diff = set1.difference(set2)
    print(diff)

    # For debugging
    # tokens = tokens_without_punctuation

    print()

    # Previous code: 3.1, from p.75
    nltk.download('stopwords', quiet=True)
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [x for x in tokens if x not in stopwords]
    kite_counts = Counter(tokens)

    # Make a vector out of the frequency dictionary, p.76
    document_vector = []
    doc_length = len(tokens)
    for key, value in kite_counts.most_common():
        document_vector.append(value / doc_length)
    print(document_vector)

    print()

    # Build harry_docs corpus, p.77
    docs = ["The faster Harry got to the store, the faster and faster Harry would get home."]
    docs.append("Harry is hairy and faster than Jill.")
    docs.append("Jill is not as hairy as Harry.")

    # Build lexicon from these 3 documents, p.77
    doc_tokens = []
    for doc in docs:
        doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
    print(len(doc_tokens[0]))
    all_doc_tokens = sum(doc_tokens, [])
    print(len(all_doc_tokens))
    lexicon = sorted(set(all_doc_tokens))
    print(len(lexicon))
    print(lexicon)

    # Build document vectors, p.78
    # Base vector
    zero_vector = OrderedDict((token, 0) for token in lexicon)
    print(zero_vector)

    # Update base vector for each document
    doc_vectors = []
    for doc in docs:
        vec = copy.copy(zero_vector)
        tokens = tokenizer.tokenize(doc.lower())
        token_counts = Counter(tokens)
        for key, value in token_counts.items():
            vec[key] = value / len(lexicon)
        doc_vectors.append(vec)


# 3.3: Zipf’s Law, p.83
def _3_3():
    # Download the Brown Cor- pus from NLTK, p.85
    nltk.download('brown')
    from nltk.corpus import brown
    print(brown.words()[:10])
    print(brown.tagged_words()[:5])

    print()

    # Count tokens, p.85
    # puncs = string.punctuation
    puncs = set((',', '.', '--', '-', '!', '?', ':', ';', '``', "''", '(', ')', '[', ']'))
    word_list = (x.lower() for x in brown.words() if x not in puncs)
    token_counts = Counter(word_list)
    print(token_counts.most_common(20))

    print()

    # Word frequencies in the Brown corpus follow the logarithmic relationship Zipf predicted, pp.85-86
    word_generator = (x.lower() for x in brown.words() if x not in puncs and '_' not in x and ' ' not in x)
    token_counts = pd.DataFrame(((w, c) for (w, c) in Counter(word_generator).items() if c > 10),
                                columns=['Token', 'Token Count'])
    token_counts.sort_values('Token Count', ascending=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    token_counts[['Token Count']].plot(style='+', logx=True, logy=True, ax=ax)
    plt.xlabel('Rank')
    plt.ylabel('Word Count')
    plt.show()


# 3.4: Topic modeling, p.86
def _3_4():
    # Load kite texts and get total word count for each document, p,87
    from data import kite_history, kite_text
    kite_intro = kite_text.lower()
    tokenizer = TreebankWordTokenizer()
    intro_tokens = tokenizer.tokenize(kite_intro)
    kite_history = kite_history.lower()
    history_tokens = tokenizer.tokenize(kite_history)
    intro_total = len(intro_tokens)
    print(intro_total)
    history_total = len(history_tokens)
    print(history_total)

    print()

    # Compute term frequency of “kite” in each document, p.87
    intro_tf = {}
    history_tf = {}
    intro_counts = Counter(intro_tokens)
    intro_tf['kite'] = intro_counts['kite'] / intro_total
    history_counts = Counter(history_tokens)
    history_tf['kite'] = history_counts['kite'] / history_total
    print('Term Frequency of "kite" in intro is: {:.4f}'.format(intro_tf['kite']))
    print('Term Frequency of "kite" in history is: {:.4f}'.format(history_tf['kite']))


if __name__ == '__main__':
    _3_4()
