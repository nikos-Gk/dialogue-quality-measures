import json
import os
import sys
import time

import pandas as pd
from convokit import Speaker, Utterance

from .arg_qual_dimensions import AQualityDimensions

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


def calculate_arg_dim(input_directory, openAIKEY, moderator_flag=True):

    if not os.path.exists(input_directory):
        print(input_directory)
        print("input directory does not exist. Exiting")
        sys.exit(1)

    if not openAIKEY:
        print(
            "OpenAI API key does not exist. Argument quality scores will not be computed.Exiting"
        )
        sys.exit(1)

    discussions = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if os.path.isfile(os.path.join(input_directory, f))
    ]
    print("Building corpus of ", len(discussions), "discussions")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    argqualitydimensions_scores_llm_output_dict = {}
    for disc in discussions:
        utterances, conv_topic = processDiscussion(disc, moderator_flag)
        disc_id = utterances[0].id.split("_")[2]
        print("Argument Quality Dimensions-Proccessing disc: ", disc_id, " with LLM")
        argqualdimensions = AQualityDimensions(utterances, conv_topic, openAIKEY)
        argument_quality_dimensions_features = (
            argqualdimensions.argquality_dimensions_scores()
        )
        argqualitydimensions_scores_llm_output_dict[disc_id] = (
            argument_quality_dimensions_features
        )

    with open("llm_output_maq" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(
            argqualitydimensions_scores_llm_output_dict,
            fout,
            ensure_ascii=False,
            indent=4,
        )

    """        
    with open("llm_output_maq.json", encoding="utf-8") as f:
        maq = json.load(f)
    argqualitydimensions_scores_llm_output_dict=maq
    """
    arq_dim_per_disc = {}
    for disc_id, turnAnnotations in argqualitydimensions_scores_llm_output_dict.items():
        for label in turnAnnotations:
            if label == -1 or not label.startswith("-Level 1a:"):
                print("LLM output for utterance is ill-formatted, skipping utterance\n")
                print(label)
                continue
            parts = label.split("\n")
            if len(parts) != 14:
                print(
                    "LLM output with missing arg quality dimensions, skipping utterance\n"
                )
                print(label)
                continue
            feature = {}
            for j in parts:
                entries = j.split(":")
                key = entries[0]
                value = entries[1]
                key = key.replace("-", "")
                rightParenthesisIndex = value.find("]")
                leftParenthesisIndex = value.find("[")
                value = value[leftParenthesisIndex:rightParenthesisIndex]
                value = value.replace("[", "").replace("]", "")
                feature[key] = value
            if disc_id in arq_dim_per_disc:
                arq_dim_per_disc[disc_id].append(feature)
            else:
                arq_dim_per_disc[disc_id] = [feature]

    average_argqualdim_per_disc = []
    for disc_id, maq_features in arq_dim_per_disc.items():
        maq_df = pd.DataFrame(maq_features)
        cols = maq_df.columns
        maq_df[cols] = maq_df[cols].apply(pd.to_numeric, errors="coerce")
        mean = maq_df.mean()
        average_argqualdim_per_disc.append({disc_id: mean.to_dict()})

    flattened_data = []
    for item in average_argqualdim_per_disc:
        for key, value in item.items():
            flattened_data.append(value)
    df = pd.DataFrame(flattened_data)
    mean_values = df.mean()

    average_argqualdim_per_disc.append({"aggregate_mean": mean_values.to_dict()})

    with open("maq_mean_" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(average_argqualdim_per_disc, fout, ensure_ascii=False, indent=4)

    with open("maq_per_utterance_" + timestr + ".json", "w", encoding="utf-8") as fout:
        json.dump(arq_dim_per_disc, fout, ensure_ascii=False, indent=4)
