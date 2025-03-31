import json
import os
import sys
import time

import pandas as pd
from convokit import Speaker, Utterance

from .collaboration_utility import CollaborationUtility

MODERATOR = "moderator"
MESSAGE_THREASHOLD_IN_CHARS = 25
HARDCODED_MODEL = "hardcoded"


def processDiscussion(discussion, moderator_flag):
    with open(discussion, "r", encoding="utf-8") as file:
        data = json.load(file)

    conversation_id = data["id"]
    speakers = {user: Speaker(id=user) for user in data["users"]}

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
        text = text.rstrip().lstrip()
        if i == 0 and model == HARDCODED_MODEL:
            conv_topic = text
        utterances.append(
            Utterance(
                id=f"utt_{i}_{conversation_id}",
                speaker=speakers[speaker],
                conversation_id=str(conversation_id),
                text=text,
                meta={"timestamp": data["timestamp"]},
            )
        )
    return utterances, conv_topic


def calculate_collaboration(input_directory, moderator_flag=True):

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

    aggregate_utterances = []
    collaboration_features_per_disc = []
    for disc in discussions:
        utterances, conv_topic = processDiscussion(disc, moderator_flag)
        colab = CollaborationUtility(utterances)
        collaboration_features = colab.calculate_collaboration_features()
        collaboration_features_per_disc.append(collaboration_features)
        aggregate_utterances = aggregate_utterances + utterances

    average_collaboration_features_per_disc = []
    for col_features in collaboration_features_per_disc:
        disc_id = list(col_features.keys())[0].split("_")[2]
        col_features_df = pd.DataFrame(col_features)
        col_features_df = col_features_df.transpose()
        mean = col_features_df.mean()
        average_collaboration_features_per_disc.append({disc_id: mean.to_dict()})

    flattened_data = []
    for item in average_collaboration_features_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    average_collaboration_features_per_disc.append(
        {"aggregate_mean": mean_values.to_dict()}
    )

    with open("collaboration_" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(
            average_collaboration_features_per_disc, fout, ensure_ascii=False, indent=4
        )

    with open(
        "collaboration_features_per_utterance_" + timestr + ".json",
        "w",
        encoding="utf-8",
    ) as fout:
        json.dump(collaboration_features_per_disc, fout, ensure_ascii=False, indent=4)
