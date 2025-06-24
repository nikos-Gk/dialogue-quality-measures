import time

import pandas as pd

from DiscQuA.utils import getUtterances, save_dict_2_json

from .collaboration_utility import CollaborationUtility


def calculate_collaboration(message_list, speakers_list, disc_id, discussion_level):
    utterances, speakers = getUtterances(message_list, speakers_list, disc_id)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if discussion_level:
        collaboration_features_per_utt = []
        colab = CollaborationUtility(utterances)
        collaboration_features = colab.calculate_collaboration_features()
        collaboration_features_per_utt.append(collaboration_features)

        collaboration_features_per_disc = []
        for col_features in collaboration_features_per_utt:
            disc_id = list(col_features.keys())[0].split("_")[2]
            col_features_df = pd.DataFrame(col_features)
            col_features_df = col_features_df.transpose()
            aggregate = col_features_df.sum()
            collaboration_features_per_disc.append({disc_id: aggregate.to_dict()})
        save_dict_2_json(
            collaboration_features_per_disc,
            "collaboration_per_disc",
            disc_id,
            timestr,
        )
        return collaboration_features_per_disc
    else:
        collaboration_features_per_utt = []
        colab = CollaborationUtility(utterances)
        collaboration_features = colab.calculate_collaboration_features()
        collaboration_features_per_utt.append(collaboration_features)
        save_dict_2_json(
            collaboration_features_per_utt, "collaboration_per_utt", disc_id, timestr
        )
        return collaboration_features_per_utt
