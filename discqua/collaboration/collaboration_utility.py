import os
import re
from collections import defaultdict

import nltk

from .stopwords import stopwords as mallet_stopwords


class Lexicon(object):
    def __init__(self, wordlists):
        self.wordlists = wordlists
        self.regex = {
            cat: self.wordlist_to_re(wordlist) for cat, wordlist in wordlists.items()
        }

    def wordlist_to_re(self, wordlist):
        return re.compile(r"\b(?:{})\b".format("|".join(wordlist).lower()))

    def count_words(self, text, return_match=False):
        text_ = text.lower()
        match = {cat: reg.findall(text_) for cat, reg in self.regex.items()}
        count = {cat: len(m) for cat, m in match.items()}

        if return_match:
            return count, match
        else:
            return count


lexicons = {
    "pron_me": [
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "id",
        "im",
        "ive",
        "me",
        "mine",
        "my",
        "myself",
    ],
    "pron_we": [
        "let's",
        "lets",
        "our",
        "ours",
        "ourselves",
        "us",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "weve",
    ],
    "pron_you": [
        "y'all",
        "yall",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "youd",
        "youll",
        "your",
        "youre",
        "yours",
        "youve",
    ],
    "pron_3rd": [
        "he",
        "he'd",
        "he's",
        "hed",
        "her",
        "hers",
        "herself",
        "hes",
        "him",
        "himself",
        "his",
        "she",
        "she'd",
        "she'll",
        "she's",
        "shes",
        "their",
        "them",
        "themselves",
        "they",
        "they'd",
        "they'll",
        "they've",
        "theyd",
        "theyll",
        "theyve",
        "they're",
        "theyre",
    ],
}


script_dir = os.path.dirname(__file__)  # Directory of the current script
file_path = os.path.join(script_dir, "lexicons", "my_geo.txt")

with open(os.path.join(script_dir, "lexicons", "my_geo.txt"), encoding="utf-8") as f:
    lexicons["geo"] = [line.strip().lower() for line in f]

with open(os.path.join(script_dir, "lexicons", "my_meta.txt"), encoding="utf-8") as f:
    lexicons["meta"] = [line.strip().lower() for line in f]

with open(
    os.path.join(script_dir, "lexicons", "my_certain.txt"), encoding="utf-8"
) as f:
    lexicons["certain"] = [line.strip().lower() for line in f]

with open(os.path.join(script_dir, "lexicons", "my_hedges.txt"), encoding="utf-8") as f:
    lexicons["hedge"] = [line.strip().lower() for line in f]

lex_matcher = Lexicon(lexicons)

desired_tags = (
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "^",
    "JJ",
    "JJR",
    "JJS",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VPZ",
)


def get_content_tagged(words, tags):
    """Return content words based on tag"""
    return [
        w for w, tag in zip(words.lower().split(), tags.split()) if tag in desired_tags
    ]


def message_features(reasons, stopwords=mallet_stopwords):
    seen_words = set()
    introduced = defaultdict(set)
    where_introduced = defaultdict(list)

    reason_features = []

    for k, (user, tokens, tags) in enumerate(reasons):
        features = {}
        content_words = [
            w for w in get_content_tagged(tokens, tags) if w not in stopwords
        ]

        introduced[user].update(content_words)

        seen_words.update(content_words)
        for w in content_words:
            where_introduced[w].append(("reason", k))

        features["n_words"] = len(tokens.split())
        lex_counts = lex_matcher.count_words(tokens)
        features.update(lex_counts)

        features["n_introduced"] = len(content_words)
        features["n_introduced_w_certain"] = (
            features["n_introduced"] * features["certain"]
        )
        features["n_introduced_w_hedge"] = features["n_introduced"] * features["hedge"]

        reason_features.append(features)

    return reason_features


def deriving_collaboration_markers(utterance, speaker):
    pos_tags = nltk.pos_tag(utterance.split())
    tokens = utterance
    tags = " ".join([tag for (_, tag) in pos_tags])

    test_reasons = [(speaker, tokens, tags)]  # Create the 'reason' structure
    reason_feat = message_features(test_reasons)
    return reason_feat


class CollaborationUtility:
    def __init__(self, utterances):
        self.utterances = utterances

    def calculate_collaboration_features(self):
        conv_dict = {}
        for utt in self.utterances:
            text = utt.text
            speaker = utt.get_speaker().id
            g = deriving_collaboration_markers(text, speaker)[0]
            conv_dict[utt.get_id()] = g
        return conv_dict
