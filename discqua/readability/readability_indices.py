# -*- coding: utf-8 -*-
import math
import os
import sys
import time

import nltk
import regex as re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from discqua.utils import dprint, save_dict_2_json


def syllable_count(word, stopwords_list, pronouncing_dict):
    word = word.lower()
    if (
        word in pronouncing_dict
        and word not in stopwords.words("english")
        and word not in stopwords_list
    ):
        syllable_list = [
            len(list(y for y in x if y[-1].isdigit())) for x in pronouncing_dict[word]
        ]
        return max(syllable_list)
    else:
        return 0


def calculate_gunning_fog_smog_fleschkincaid_index(
    text, stopwords_list, pronouncing_dict
):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Filter out punctuation and stopwords
    complex_words = [
        word
        for word in words
        if re.match(r"[A-Za-z]+", word)
        and syllable_count(word, stopwords_list, pronouncing_dict) > 2
    ]
    syllable_words = [
        syllable_count(word, stopwords_list, pronouncing_dict) for word in words
    ]

    # Total number of words and sentences
    total_words = len(words)
    total_sentences = len(sentences)
    total_complex_words = len(complex_words)
    total_syllable_words = sum(syllable_words)

    if total_words == 0 or total_sentences == 0:
        return 0, 0, 0, 0

    # Gunning-Fog index calculation
    gunning_fog_index = 0.4 * (
        (total_words / total_sentences) + (100 * (total_complex_words / total_words))
    )

    smog_index = (
        1.0430 * math.sqrt(total_complex_words * (30 / total_sentences)) + 3.1291
    )
    Flesch_index = (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllable_words / total_words)
    )
    Flesch_Kincaid_index = (
        0.39 * (total_words / total_sentences)
        + 11.8 * (total_syllable_words / total_words)
        - 15.59
    )

    return gunning_fog_index, smog_index, Flesch_index, Flesch_Kincaid_index


def readability(message_list, msgsid_list, disc_id):
    """Calculates readability indices (including Gunning Fog Index, SMOG Index, Flesch Reading Ease, and Flesch-Kincaid) for each utterance in a discussion.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        msgsid_list (list[str]) : List of messages ids corresponding to each utterance.
        disc_id (str): Unique identifier for the discussion.

    Returns:
     dict: A dictionary containing the readability indices for each utterance in the discussion,
              structured as {disc_id: {msg_id: {readability_scores}}}.
    """
    if len(message_list) != len(msgsid_list):
        print("The lengths of 'message_list' and 'msgsid_list' do not match")
        sys.exit(1)
    nltk.download("cmudict")
    pronouncing_dict = nltk.corpus.cmudict.dict()

    stopwords_list = []
    script_dir = os.path.dirname(__file__)  # Directory of the current script
    file_path = os.path.join(script_dir, "stopwords", "StopWords_Generic.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        for word in f.read().splitlines():
            stopwords_list.append(word.lower())
    dprint("info", f"Building corpus of: {len(message_list)} utterances ")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    utt_dict = {}
    read_indic_per_disc = {}
    for i, utt in enumerate(message_list):
        try:
            dprint(
                "info",
                f"Readability Indices Per Utterance-Proccessing discussion: {disc_id}, utterance:utt_{i}",
            )
            #
            gunning_fog_index, smog_index, Flesch_index, Flesch_Kincaid_index = (
                calculate_gunning_fog_smog_fleschkincaid_index(
                    utt, stopwords_list, pronouncing_dict
                )
            )
            key_iter = str(msgsid_list[i])
            utt_dict[key_iter] = {
                "Gunning_Fog": gunning_fog_index,
                "Smog": smog_index,
                "Flesch": Flesch_index,
                "Flesch_Kincaid": Flesch_Kincaid_index,
            }
            read_indic_per_disc[disc_id] = utt_dict
            #
        except Exception as e:
            print("Error: ", e)
            print(disc_id, key_iter)

    save_dict_2_json(
        read_indic_per_disc,
        "readability_indices_per_disc_utt",
        disc_id,
        timestr,
    )

    return read_indic_per_disc
