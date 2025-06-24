import time

import numpy as np

from DiscQuA.utils import save_dict_2_json


def calculate_controversy(message_list, disc_id):
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
