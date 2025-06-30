import time

import numpy as np

from DiscQuA.utils import save_dict_2_json


def calculate_controversy(message_list, disc_id, discussion_level):
    """Evaluates the controversy of a discussion based on sentiment variability.
       Sentiment scores are computed using a pretrained multilingual sentiment analysis model.
       Variability is quantified using the sample standard deviation of the sentiment scores, either at the discussion or utterance level.

    Args:
        message_list (list[str]): The list of utterances in the discussion.
        disc_id (str): Unique identifier for the discussion.
        discussion_level (bool): A boolean flag; if True, the annotations are applied at the discussion level; otherwise at the utterance level.

    Returns:
        - If discussion_level is True:
                - dict[str, float]: Unnormalized controversy score (sample std) per discussion.
                - dict[str, float]: Normalized controversy score per discussion.
        - If discussion_level is False:
                - dict[str, dict[str, float]]: Rolling standard deviation of the unormalized sentiment scores,
                                               where keys are discussion IDs and values are dicts indexed
                                               by utterance window (e.g., "utt_[0:2]").
                - dict[str, dict[str, float]]: Corresponding rolling standard deviation of the normalized sentiment scores.
    """

    from transformers import pipeline

    pipe = pipeline(
        "text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
    #
    print("Building corpus of ", len(message_list), "utterances")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #
    unorm_disc_controversy_dict = {}
    norm_disc_controversy_dict = {}
    disc_resultlist_dict = {}
    disc_productlist_dict = {}
    #
    utterances = []
    for i, utt in enumerate(message_list):
        utt = utt.replace("\r\n", " ").replace("\n", " ").rstrip().lstrip()
        utterances.append((utt, f"conv_{disc_id}_utt_{i}"))

    #
    print("Controversy-Proccessing disc: ", disc_id)
    utt_resultlist_iter_dict = {}
    utt_labelprobproduct_iter_dict = {}
    utt_unnormalized_factor_iter_dict = {}
    utt_normalized_factor_iter_dict = {}
    for utt in utterances:
        try:
            text_iter, id_iter = utt
            if len(text_iter) > 512:
                text_iter = text_iter[0:513]
            result_list = pipe([text_iter], return_all_scores=True)[0]
            utt_resultlist_iter_dict[id_iter] = result_list
            product_list = []
            for result in result_list:
                star = int(result["label"][0])
                prob = float(result["score"])
                product = star * prob
                product_list.append(product)
            utt_labelprobproduct_iter_dict[id_iter] = product_list
            sent_score = sum(product_list)
            utt_unnormalized_factor_iter_dict[id_iter] = sent_score
        except Exception as ex:
            print("Exception at atterance: ", utt)
            print(ex)
    disc_resultlist_dict[disc_id] = utt_resultlist_iter_dict
    disc_productlist_dict[disc_id] = utt_labelprobproduct_iter_dict
    #
    max_iter = max(utt_unnormalized_factor_iter_dict.values())
    min_iter = min(utt_unnormalized_factor_iter_dict.values())
    utt_normalized_factor_iter_dict = {
        key: (value - min_iter) / (max_iter - min_iter)
        for key, value in utt_unnormalized_factor_iter_dict.items()
    }
    unorm_disc_controversy_dict[disc_id] = utt_unnormalized_factor_iter_dict
    norm_disc_controversy_dict[disc_id] = utt_normalized_factor_iter_dict

    save_dict_2_json(
        unorm_disc_controversy_dict, "bert_sent_per_comment_unorm", disc_id, timestr
    )
    save_dict_2_json(
        norm_disc_controversy_dict, "bert_sent_per_comment_norm", disc_id, timestr
    )

    """            
        with open("bert_sent_per_comment_unorm_.json", encoding="utf-8") as f:
            sent_scores = json.load(f)
        unorm_disc_cotroversy_dict=sent_scores
    """

    """            
        with open("bert_sent_per_comment_norm_.json", encoding="utf-8") as f:
            sent_scores = json.load(f)
        norm_disc_controversy_dict=sent_scores
    """
    if discussion_level:
        controversy_per_disc_unorm = {}

        for conversation_id, dict_iter in unorm_disc_controversy_dict.items():
            std_iter = np.std(list(dict_iter.values()), ddof=1)
            controversy_per_disc_unorm[conversation_id] = std_iter

        save_dict_2_json(
            controversy_per_disc_unorm, "controversy_per_disc_unorm", disc_id, timestr
        )

        controversy_per_disc_norm = {}

        for conversation_id, dict_iter in norm_disc_controversy_dict.items():
            std_iter = np.std(list(dict_iter.values()), ddof=1)
            controversy_per_disc_norm[conversation_id] = std_iter

        save_dict_2_json(
            controversy_per_disc_norm, "controversy_per_disc_norm", disc_id, timestr
        )
        return controversy_per_disc_unorm, controversy_per_disc_norm
    else:

        controversy_perutt_unorm = {}

        for disc_id, sent_dict in unorm_disc_controversy_dict.items():
            utt_ids_sorted = sorted(
                sent_dict.keys(), key=lambda x: int(x.split("_")[-1])
            )
            scores_sorted = [sent_dict[utt_id] for utt_id in utt_ids_sorted]
            #
            rolling_std_dict = {}
            for i in range(len(scores_sorted)):
                key_iter = f"utt_[0:{i}]"
                if i == 0:
                    rolling_std_dict[key_iter] = 0.0
                else:
                    rolling_std_dict[key_iter] = float(
                        np.std(scores_sorted[: i + 1], ddof=1)
                    )
            controversy_perutt_unorm[disc_id] = rolling_std_dict

        save_dict_2_json(
            controversy_perutt_unorm, "controversy_per_utt_unorm", disc_id, timestr
        )

        controversy_perutt_norm = {}

        for disc_id, sent_dict in norm_disc_controversy_dict.items():
            utt_ids_sorted = sorted(
                sent_dict.keys(), key=lambda x: int(x.split("_")[-1])
            )
            scores_sorted = [sent_dict[utt_id] for utt_id in utt_ids_sorted]
            #
            rolling_std_dict = {}
            for i in range(len(scores_sorted)):
                key_iter = f"utt_[0:{i}]"
                if i == 0:
                    rolling_std_dict[key_iter] = 0.0
                else:
                    rolling_std_dict[key_iter] = float(
                        np.std(scores_sorted[: i + 1], ddof=1)
                    )
            controversy_perutt_norm[disc_id] = rolling_std_dict

        save_dict_2_json(
            controversy_perutt_norm, "controversy_per_utt_norm", disc_id, timestr
        )

        return controversy_perutt_unorm, controversy_perutt_norm
