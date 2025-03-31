import json
import os
import sys
import time

import pandas as pd
from convokit import Speaker, Utterance

from .DisputeTacticsCl import DisputeTactics

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


def calculate_dispute_tactics(input_directory, openAIKEY, moderator_flag=True):

    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    if not openAIKEY:
        print(
            "OpenAI API key does not exist. Dispute tactics labels will not be annotated."
        )
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    dispute_tactics_llm_output_dict = {}

    for disc in discussions:
        utterances, conv_topic = processDiscussion(disc, moderator_flag)
        disc_id = utterances[0].id.split("_")[2]
        print("Proccessing disc: ", disc_id, " with LLM")
        dispute = DisputeTactics(utterances, conv_topic, openAIKEY)
        dispute_features = dispute.calculate_dispute_tectics()
        dispute_tactics_llm_output_dict[disc_id] = dispute_features

    with open(
        "llm_dispute_tactics_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(dispute_tactics_llm_output_dict, fout, ensure_ascii=False, indent=4)

    """        
    with open("llm_dispute_tactics_20241123-205834.json", encoding="utf-8") as f:
        d = json.load(f)
    dispute_tactics_llm_output_dict=d
    """

    dispute_tactics_per_disc = {}
    for disc_id, turnAnnotations in dispute_tactics_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1 or not label.startswith("- Level 0:"):
                print("LLM output for utterance is ill-formatted, skipping utterance\n")
                print(label)
                continue
            parts = label.split("\n")
            feature = {}
            for j in parts:
                entries = j.split(":")
                key = entries[0]
                value = entries[1]
                key = key.replace("-", "")
                value = value.replace("[", "").replace("]", "")
                feature[key] = value
            if disc_id in dispute_tactics_per_disc:
                dispute_tactics_per_disc[disc_id].append(feature)
            else:
                dispute_tactics_per_disc[disc_id] = [feature]

    average_dispute_tactics_features_per_disc = []
    for disc_id, dt_features in dispute_tactics_per_disc.items():
        dt_df = pd.DataFrame(dt_features)
        cols = dt_df.columns
        dt_df[cols] = dt_df[cols].apply(pd.to_numeric, errors="coerce")
        mean = dt_df.mean()
        average_dispute_tactics_features_per_disc.append({disc_id: mean.to_dict()})

    flattened_data = []
    for item in average_dispute_tactics_features_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    average_dispute_tactics_features_per_disc.append(
        {"aggregate_mean": mean_values.to_dict()}
    )

    with open(
        "dispute_tactics_mean_" + timestr + ".json", "w", encoding="utf-8"
    ) as fout:
        json.dump(
            average_dispute_tactics_features_per_disc,
            fout,
            ensure_ascii=False,
            indent=4,
        )

    with open(
        "dispute_tactics_features_per_utterance_" + timestr + ".json",
        "w",
        encoding="utf-8",
    ) as fout:
        json.dump(dispute_tactics_per_disc, fout, ensure_ascii=False, indent=4)
