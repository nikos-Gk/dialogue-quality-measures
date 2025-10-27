import sys
import time

import pandas as pd

from discqua.utils import getUtterances, save_dict_2_json

from .collaboration_utility import CollaborationUtility


def collaboration(message_list, speakers_list, msgsid_list, disc_id, discussion_level):
    """Annotates a discussion with conversational markers indicative of collaboration, such as expressions of confidence, uncertainty, pronoun usage, and idea adoption, based on Niculae and Danescu-Niculescu-Mizil (2016).
    Args:
        message_list (list[str]): The list of utterances in the discussion.
        speakers_list (list[str]): The corresponding list of speakers for each utterance.
        msgsid_list (list[str]) : List of messages ids corresponding to each utterance.
        disc_id (str): Unique identifier for the discussion.
        discussion_level (bool): A boolean flag; if True, annotation at the discussion level; otherwise at the utterance level.

    Returns:
        list[dict[str, dict[str, float]]]: A list containing one dictionary per discussion.
        Each dictionary maps a discussion ID to its corresponding set of collaboration
        feature scores (aggregated if discussion-level, otherwise per-utterance).
    """
    if len(message_list) != len(msgsid_list):
        print("The lengths of 'message_list' and 'msgsid_list' do not match")
        sys.exit(1)
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
        collaboration_features_msgIDs = {}
        counter = 0
        for key, value in collaboration_features.items():
            collaboration_features_msgIDs[msgsid_list[counter]] = value
            counter += 1
        # collaboration_features_per_utt.append({disc_id: collaboration_features_msgIDs})
        save_dict_2_json(
            {disc_id: collaboration_features_msgIDs},
            "collaboration_per_utt",
            disc_id,
            timestr,
        )
        return collaboration_features_per_utt
