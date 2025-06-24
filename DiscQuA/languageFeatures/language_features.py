import time

from DiscQuA.utils import save_dict_2_json

from .stopwords import stopwords

###################################################################
stopwords = [w.strip() for w in stopwords]


###################################################################
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


def calculate_language_features(message_list, disc_id):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    utterances = message_list
    language_feat_dict = {}
    print("language_features-Proccessing disc: ", disc_id)
    utt_pair_dict = {}
    features_allwords_dict = {}
    features_stopwords_dict = {}
    features_content_dict = {}
    for i, utt in enumerate(utterances):
        if i == 0:
            continue
        utt2_text_iter = utterances[i]
        utt1_text_iter = utterances[i - 1]
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
    language_feat_dict[disc_id] = utt_pair_dict
    save_dict_2_json(language_feat_dict, "language_feat_per_pair", disc_id, timestr)
    return language_feat_dict
