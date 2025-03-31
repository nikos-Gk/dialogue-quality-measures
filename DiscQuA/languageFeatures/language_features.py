import json
import os
import sys
import time

import pandas as pd

from .stopwords import stopwords

###################################################################
stopwords = [w.strip() for w in stopwords]

###################################################################
MODERATOR = "moderator"
MESSAGE_THREASHOLD_IN_CHARS = 25
HARDCODED_MODEL = "hardcoded"


def processDiscussion(discussion, moderator_flag):
    with open(discussion, "r", encoding="utf-8") as file:
        data = json.load(file)

    conversation_id = data["id"]

    if not moderator_flag:
        data["logs"] = [
            (log[0], log[1], log[2], log[3])
            for log in data["logs"]
            if log[0] != MODERATOR
        ]

    utterances = []

    data["logs"] = [
        (log[0], log[1], log[2], log[3])
        for log in data["logs"]
        if ((isinstance(log[1], str)) and (len(log[1]) > MESSAGE_THREASHOLD_IN_CHARS))
    ]

    for i, log in enumerate(data["logs"]):
        speaker, text, model, message_id = log
        text = text.replace("\r\n", " ").replace("\n", " ").rstrip().lstrip()
        if i == 0 and model == HARDCODED_MODEL:
            conv_topic = text

        utterances.append((text, speaker, f"conv_{conversation_id}_utt_{i}"))
    return utterances, conversation_id


def get_metrics_allwords(u1, u2):
    words_u1 = [w for w in u1.split()]
    words_u2 = [w for w in u2.split()]
    common_words = [w for w in words_u1 if w in words_u2]
    number_of_common_words = len(common_words)

    number_of_words_u1 = len(words_u1)
    number_of_words_u2 = len(words_u2)
    unique_words_of_u1 = list(set(words_u1))
    unique_words_of_u2 = list(set(words_u2))
    unique_common_words = list(set(unique_words_of_u1 + unique_words_of_u2))

    reply_fraction = float(number_of_common_words) / float(number_of_words_u2)
    op_fraction = float(number_of_common_words) / float(number_of_words_u1)
    jaccard = float((number_of_common_words)) / float((len(unique_common_words)))

    return (number_of_common_words, reply_fraction, op_fraction, jaccard)


def get_metrics_stopwords(u1, u2):
    words_u1 = [w for w in u1.split() if w in stopwords]
    words_u2 = [w for w in u2.split() if w in stopwords]
    common_words = [w for w in words_u1 if w in words_u2]
    number_of_common_words = len(common_words)

    number_of_words_u1 = len(words_u1)
    number_of_words_u2 = len(words_u2)
    unique_words_of_u1 = list(set(words_u1))
    unique_words_of_u2 = list(set(words_u2))
    unique_common_words = list(set(unique_words_of_u1 + unique_words_of_u2))

    if number_of_words_u2 == 0:
        reply_fraction = 0
    else:
        reply_fraction = float(number_of_common_words) / float(number_of_words_u2)
    if number_of_words_u1 == 0:
        op_fraction = 0
    else:
        op_fraction = float(number_of_common_words) / float(number_of_words_u1)
    if len(unique_common_words) == 0:
        jaccard = 0
    else:
        jaccard = float((number_of_common_words)) / float((len(unique_common_words)))

    return (number_of_common_words, reply_fraction, op_fraction, jaccard)


def get_metrics_contentwords(u1, u2):
    words_u1 = [w for w in u1.split() if w not in stopwords]
    words_u2 = [w for w in u2.split() if w not in stopwords]
    common_words = [w for w in words_u1 if w in words_u2]
    number_of_common_words = len(common_words)

    number_of_words_u1 = len(words_u1)
    number_of_words_u2 = len(words_u2)
    unique_words_of_u1 = list(set(words_u1))
    unique_words_of_u2 = list(set(words_u2))
    unique_common_words = list(set(unique_words_of_u1 + unique_words_of_u2))

    if number_of_words_u2 == 0:
        reply_fraction = 0
    else:
        reply_fraction = float(number_of_common_words) / float(number_of_words_u2)
    if number_of_words_u1 == 0:
        op_fraction = 0
    else:
        op_fraction = float(number_of_common_words) / float(number_of_words_u1)
    if len(unique_common_words) == 0:
        jaccard = 0
    else:
        jaccard = float((number_of_common_words)) / float((len(unique_common_words)))

    return (number_of_common_words, reply_fraction, op_fraction, jaccard)


def calculate_language_features(input_directory, moderator_flag=True):
    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    language_feat_dict = {}
    for disc in discussions:
        utterances, conversation_id = processDiscussion(disc, moderator_flag)
        print("language_features-Proccessing disc: ", conversation_id)
        utt_pair_dict = {}
        features_allwords_dict = {}
        features_stopwords_dict = {}
        features_content_dict = {}

        for i, utt in enumerate(utterances):
            if i == 0:
                continue
            utt2_tuple = utterances[i]
            utt1_tuple = utterances[i - 1]
            utt2_text_iter, utt2_user_iter, utt2_id_iter = utt2_tuple
            utt1_text_iter, utt1_user_iter, utt1_id_iter = utt1_tuple
            pair_key = f"pair_{i-1,i}"
            number_of_common_words, reply_fraction, op_fraction, jaccard = (
                get_metrics_allwords(utt1_text_iter, utt2_text_iter)
            )
            (
                number_of_common_stopwords,
                reply_fraction_stopwords,
                op_fraction_stopwords,
                jaccard_stopwords,
            ) = get_metrics_stopwords(utt1_text_iter, utt2_text_iter)
            (
                number_of_content_words,
                reply_fraction_content,
                op_fraction_content,
                jaccard_content,
            ) = get_metrics_contentwords(utt1_text_iter, utt2_text_iter)
            features_allwords_dict = {
                "n_comwords": number_of_common_words,
                "reply_fra_comwords": reply_fraction,
                "op_fra_comwords": op_fraction,
                "jac_comwords": jaccard,
            }
            features_stopwords_dict = {
                "n_stopwords": number_of_common_stopwords,
                "reply_fra_stopwords": reply_fraction_stopwords,
                "op_fra_stopwords": op_fraction_stopwords,
                "jac_stopwords": jaccard_stopwords,
            }
            features_content_dict = {
                "n_contwords": number_of_content_words,
                "reply_fra_contwords": reply_fraction_content,
                "op_fra_contwords": op_fraction_content,
                "jac_contwords": jaccard_content,
            }
            utt_pair_dict[pair_key] = [
                features_allwords_dict,
                features_stopwords_dict,
                features_content_dict,
            ]
            language_feat_dict[conversation_id] = utt_pair_dict

    with open(
        "language_feat_per_pair_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(language_feat_dict, fout, ensure_ascii=False, indent=4)

    """            
    with open("language_feat_per_pair_.json", encoding="utf-8") as f:
        lang_scores = json.load(f)
    language_feat_dict=lang_scores
    """

    language_features_per_disc = {}

    # number_of_common_words_list = []
    for conversation_id, dict_iter in language_feat_dict.items():
        total_list = []
        for pair_key, features_list in dict_iter.items():
            total = {}
            for feature in features_list:
                total.update(feature)
            total_list.append(total)
        a = pd.DataFrame(total_list)
        mean = a.mean()
        language_features_per_disc[conversation_id] = mean.to_dict()

    flattened_data = []
    for key, value in language_features_per_disc.items():
        flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    language_features_per_disc["aggregate_mean"] = mean_values.to_dict()

    with open(
        "language_feat_per_disc_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(language_features_per_disc, fout, ensure_ascii=False, indent=4)
